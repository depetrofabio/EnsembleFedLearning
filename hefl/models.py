"""
Model definitions: cluster backbones, per-cluster heads, and swappable combiners.

The central refactor relative to v1
-----------------------------------
v1 concatenated the K feature vectors into a ``(B, K*512)`` tensor and applied a
single ``Linear(K*512, 10)``.  Because a client only ever activates its own
512-slot, the gradient on every other column block is exactly zero, so that
"shared classifier" is really K disjoint heads that are never trained jointly.
The test-time combination (divide by K, sum) is a constant that no gradient ever
saw.

Here the same computation is written explicitly:

    logits_k = head_k(backbone_k(x))            per-expert logits, (B, C)
    logits   = sum_k w_k * logits_k + bias      combination

With ``w_k = 1/|A|`` this is *numerically identical* to v1, so nothing is lost.
But now ``w`` is produced by a ``Combiner`` module that can be swapped and, more
importantly, trained.

Train/test consistency
----------------------
A client trains through ``ClientView``: ``logits = head_c(backbone_c(x)) + bias``.
Every weight-producing combiner is defined so that a single active expert
receives weight exactly 1.0.  The local objective a client optimises is
therefore the *same function* the ensemble evaluates when only that expert is
active - no magnitude correction needed, in any active subset.  ``MLPCombiner``
is the one exception and is documented as such.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

FEATURE_DIM = 512


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
def _replace_bn_with_gn(module: nn.Module, groups: int = 32) -> nn.Module:
    """Swap every BatchNorm2d for GroupNorm.

    Averaging BatchNorm running statistics across clients that see different
    input distributions is the failure mode FedBN exists to prevent.  GroupNorm
    carries no cross-client state, which removes that confound entirely.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(min(groups, num_channels), num_channels))
        else:
            _replace_bn_with_gn(child, groups)
    return module


class CifarBackbone(nn.Module):
    """ResNet-18 adapted to 32x32 inputs, returning 512-d features (no head).

    The stem is the standard CIFAR modification: 3x3 stride-1 conv and no
    maxpool, otherwise a 32x32 image is downsampled to 1x1 far too early.
    """

    def __init__(self, norm: str = "gn", groups: int = 32) -> None:
        super().__init__()
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        if norm == "gn":
            net = _replace_bn_with_gn(net, groups)
        elif norm != "bn":
            raise ValueError(f"norm must be 'gn' or 'bn', got {norm}")
        self.net = net
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 512)


# --------------------------------------------------------------------------- #
# Combiners
# --------------------------------------------------------------------------- #
def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim, restricted to ``mask``.

    ``scores`` is ``(..., K)`` and ``mask`` is a ``(K,)`` bool tensor.  Inactive
    experts get exactly zero weight and the active ones renormalise to sum to 1,
    which is what makes any active subset magnitude-correct by construction.
    """
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask, neg_inf)
    return torch.softmax(scores, dim=-1)


def confidence_features(logits_stack: torch.Tensor) -> torch.Tensor:
    """Per-expert confidence summary used as the gate's input.

    ``logits_stack`` is ``(B, K, C)``.  For each expert we take three scalars -
    negative entropy, max probability, and the top-1/top-2 margin - giving a
    ``(B, 3K)`` input.  Feeding the gate a 3K-dimensional summary rather than
    the K*512 features keeps it tiny (it trains on very little data) and keeps
    it interpretable: you can read off which expert it trusts and why.
    """
    logp = torch.log_softmax(logits_stack, dim=-1)
    p = logp.exp()
    neg_entropy = (p * logp).sum(-1)                                   # (B, K)
    top2 = torch.topk(p, k=min(2, p.shape[-1]), dim=-1).values
    max_p = top2[..., 0]
    margin = max_p - top2[..., 1] if top2.shape[-1] > 1 else max_p
    return torch.cat([neg_entropy, max_p, margin], dim=-1)             # (B, 3K)


class Combiner(nn.Module):
    """Interface: map per-expert logits ``(B, K, C)`` + active mask -> ``(B, C)``."""

    #: whether ``forward`` can emit routing logits for cluster-id supervision
    supports_routing: bool = False

    def forward(
        self, logits_stack: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class UniformCombiner(Combiner):
    """``w_k = 1/|A|``.  Zero parameters. This is exactly the v1 behaviour."""

    def forward(self, logits_stack, mask):
        w = mask.to(logits_stack.dtype)
        w = w / w.sum()
        combined = (logits_stack * w.view(1, -1, 1)).sum(1)
        return combined, {"weights": w.expand(logits_stack.size(0), -1)}


class BetaCombiner(Combiner):
    """One global scalar per expert, softmax-normalised over the active set.

    K parameters.  Fits on a few hundred samples and immediately removes the
    arbitrariness of uniform averaging; the learned beta is interpretable
    (it tracks cluster quality and size).
    """

    def __init__(self, num_clusters: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(num_clusters))

    def forward(self, logits_stack, mask):
        w = _masked_softmax(self.beta, mask)                     # (K,)
        combined = (logits_stack * w.view(1, -1, 1)).sum(1)
        return combined, {"weights": w.expand(logits_stack.size(0), -1)}


class GateCombiner(Combiner):
    """Input-dependent gate over experts: ``w(x) = softmax_A(g(conf(x)))``.

    This is the combiner that matters under feature shift, because the correct
    expert depends on the *sample*, not on a global constant.  It also accepts
    free supervision: a client knows its own cluster id, so the routing head can
    be trained with a cross-entropy term at no annotation cost.
    """

    supports_routing = True

    def __init__(self, num_clusters: int, hidden: int = 64, temperature: float = 1.0) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.net = nn.Sequential(
            nn.Linear(3 * num_clusters, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_clusters),
        )
        self.temperature = temperature

    def forward(self, logits_stack, mask):
        feats = confidence_features(logits_stack)                # (B, 3K)
        routing_logits = self.net(feats) / self.temperature      # (B, K)
        w = _masked_softmax(routing_logits, mask.view(1, -1))    # (B, K)
        combined = (logits_stack * w.unsqueeze(-1)).sum(1)
        return combined, {"weights": w, "routing_logits": routing_logits}


class MLPCombiner(Combiner):
    """Small MLP over the concatenated per-expert logits: ``(K*C) -> h -> C``.

    The only combiner with genuine cross-expert interaction - it can learn e.g.
    "when expert 0 says cat and expert 2 says dog, answer dog".  The trade-off
    is that it breaks the single-active equivalence with local client training,
    so it depends entirely on dense rounds.  The active mask is concatenated to
    the input so the model can condition on which experts are present.
    """

    def __init__(self, num_clusters: int, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(num_clusters * num_classes + num_clusters, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, logits_stack, mask):
        b = logits_stack.size(0)
        masked = logits_stack * mask.view(1, -1, 1).to(logits_stack.dtype)
        flat = masked.reshape(b, -1)
        inp = torch.cat([flat, mask.to(flat.dtype).view(1, -1).expand(b, -1)], dim=1)
        return self.net(inp), {}


def build_combiner(name: str, num_clusters: int, num_classes: int, **kwargs) -> Combiner:
    name = name.lower()
    if name == "uniform":
        return UniformCombiner()
    if name == "beta":
        return BetaCombiner(num_clusters)
    if name == "gate":
        return GateCombiner(num_clusters, hidden=kwargs.get("gate_hidden", 64))
    if name == "mlp":
        return MLPCombiner(num_clusters, num_classes, hidden=kwargs.get("mlp_hidden", 128))
    raise ValueError(f"unknown combiner '{name}'")


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #
class ClusterEnsemble(nn.Module):
    """K cluster backbones + K per-cluster heads + a shared bias + a combiner.

    The bias is shared across experts and updated by every client; keeping it
    outside the per-cluster heads means the label prior stays global instead of
    being re-learned (and re-skewed) per cluster.
    """

    def __init__(
        self,
        num_clusters: int,
        num_classes: int = 10,
        norm: str = "gn",
        combiner: str = "gate",
        feature_dim: int = FEATURE_DIM,
        **combiner_kwargs,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbones = nn.ModuleList([CifarBackbone(norm) for _ in range(num_clusters)])
        self.heads = nn.ModuleList(
            [nn.Linear(feature_dim, num_classes, bias=False) for _ in range(num_clusters)]
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.combiner = build_combiner(combiner, num_clusters, num_classes, **combiner_kwargs)
        self.combiner_name = combiner

    # -- helpers ----------------------------------------------------------- #
    def active_mask(self, active: Optional[Sequence[int]], device) -> torch.Tensor:
        mask = torch.zeros(self.num_clusters, dtype=torch.bool, device=device)
        if active is None:
            mask[:] = True
        else:
            if len(active) == 0:
                raise ValueError("active subset must contain at least one cluster")
            mask[list(active)] = True
        return mask

    def expert_logits(
        self, x: torch.Tensor, active: Optional[Sequence[int]] = None
    ) -> torch.Tensor:
        """Per-expert logits ``(B, K, C)``; inactive experts are not evaluated."""
        mask = self.active_mask(active, x.device)
        out = x.new_zeros(x.size(0), self.num_clusters, self.num_classes)
        for k in range(self.num_clusters):
            if mask[k]:
                out[:, k, :] = self.heads[k](self.backbones[k](x))
        return out

    # -- forward ----------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        active: Optional[Sequence[int]] = None,
        return_aux: bool = False,
    ):
        mask = self.active_mask(active, x.device)
        stack = self.expert_logits(x, active)
        combined, aux = self.combiner(stack, mask)
        combined = combined + self.bias
        if return_aux:
            aux["expert_logits"] = stack
            return combined, aux
        return combined

    # -- parameter groups -------------------------------------------------- #
    def expert_parameters(self):
        yield from self.backbones.parameters()
        yield from self.heads.parameters()
        yield self.bias

    def combiner_parameters(self):
        return self.combiner.parameters()

    def freeze_experts(self, frozen: bool = True) -> None:
        for p in self.expert_parameters():
            p.requires_grad_(not frozen)


class ClientView(nn.Module):
    """What a client actually holds and trains: one backbone, one head, the bias.

    ``logits = head_c(backbone_c(x)) + bias``.

    This is deliberately identical to ``ClusterEnsemble.forward(x, active=[c])``
    for every weight-producing combiner, since a lone active expert always gets
    weight 1.  Local training and single-expert evaluation optimise the same
    function - the train/test mismatch of v1 is gone by construction, not by a
    normalisation constant.
    """

    def __init__(self, norm: str = "gn", num_classes: int = 10, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.backbone = CifarBackbone(norm)
        self.head = nn.Linear(feature_dim, num_classes, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)) + self.bias


class DenseClientView(nn.Module):
    """Client-side view for a *dense* round: all K experts, combiner trainable.

    The experts arrive frozen; only the combiner receives gradient.  This is the
    only place where more than one expert is active during training, and it is
    what makes the combination a learned object rather than a hardcoded average.
    """

    def __init__(self, ensemble: ClusterEnsemble) -> None:
        super().__init__()
        self.ensemble = ensemble

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            stack = self.ensemble.expert_logits(x, active=None)
        mask = torch.ones(self.ensemble.num_clusters, dtype=torch.bool, device=x.device)
        combined, aux = self.ensemble.combiner(stack, mask)
        return combined + self.ensemble.bias.detach(), aux
