"""
Split-depth ensembles: personalise the early layers, share the rest.

Why this exists
---------------
The full-scale result (docs/REPORT.md 10.3) showed the cluster ensemble losing to
FedAvg by 14 points *while every component worked*. The diagnosis was arithmetic:
splitting 40 clients into 4 clusters gave each expert a quarter of the data, and
that cost more than specialisation bought back.

This module attacks that directly. Instead of K independent backbones, a
ResNet-18 is cut at a configurable depth:

    x -> [personal blocks, one copy per cluster] -> [shared trunk, ONE copy] -> head_k

The shared trunk is aggregated over **every** client, so it sees all 40k images —
no data splitting for the bulk of the network. Only the early layers, where
domain shift actually lives, are per-cluster.

``split_depth`` is a dial with both baselines as endpoints:

    0  nothing personal        -> a shared backbone; closest thing to FedAvg
    1  conv1 + first norm
    2  + layer1
    3  + layer2
    4  + layer3
    5  everything personal     -> equivalent to the independent-expert design

Sweeping it answers "how much personalisation do you actually need?", and the
answer is a curve rather than a yes/no.

Parameter cost at K=4 (ResNet-18, 11.2M params per backbone):

    split_depth=5 -> 4 full backbones      ~44.7M
    split_depth=2 -> 4 small stems + 1 trunk ~11.9M
    split_depth=0 -> 1 backbone            ~11.2M

So low split depths remove the K-times-capacity confound as well as the
data-splitting cost.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models import resnet18

from .datasets import FederatedData
from .models import FEATURE_DIM, _replace_bn_with_gn, build_combiner
from .utils import cpu_state, evaluate, label_log_prior, local_train, weighted_average

MAX_SPLIT_DEPTH = 5
PERSONAL_PREFIX = "personal."
SHARED_PREFIXES = ("shared.", "pool.", "head.", "bias")


def _resnet_blocks(norm: str = "gn") -> Tuple[List[nn.Module], nn.Module]:
    """ResNet-18 adapted for 32x32, as five sequential blocks + the pooling layer."""
    net = resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    if norm == "gn":
        net = _replace_bn_with_gn(net)
    elif norm != "bn":
        raise ValueError(f"norm must be 'gn' or 'bn', got {norm}")
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
    return [stem, net.layer1, net.layer2, net.layer3, net.layer4], net.avgpool


class SplitClientView(nn.Module):
    """What a client holds: its cluster's personal blocks + the shared remainder.

    Mirrors ``models.ClientView`` so the training loop is unchanged; the only
    difference is that its state dict is partitioned into personal and shared
    keys, which the aggregator treats differently.
    """

    def __init__(self, split_depth: int = 2, norm: str = "gn", num_classes: int = 10):
        super().__init__()
        if not 0 <= split_depth <= MAX_SPLIT_DEPTH:
            raise ValueError(f"split_depth must be in [0, {MAX_SPLIT_DEPTH}]")
        blocks, pool = _resnet_blocks(norm)
        self.split_depth = split_depth
        self.personal = nn.Sequential(*blocks[:split_depth])
        self.shared = nn.Sequential(*blocks[split_depth:])
        self.pool = pool
        self.head = nn.Linear(FEATURE_DIM, num_classes, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.shared(self.personal(x))), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)) + self.bias


def is_personal(key: str) -> bool:
    """Personal parameters are aggregated within a cluster; everything else globally."""
    return key.startswith(PERSONAL_PREFIX)


class SplitEnsemble(nn.Module):
    """K personal front-ends + one shared trunk + a shared head and bias.

    Experts differ **only** in their personal blocks: everything from
    ``split_depth`` onwards, including the classifier, is aggregated globally.
    That is the thesis of the design made literal - personalise where domain
    shift lives (early), share where the label space lives (late).
    """

    def __init__(
        self,
        num_clusters: int,
        split_depth: int = 2,
        num_classes: int = 10,
        norm: str = "gn",
        combiner: str = "uniform",
        **combiner_kwargs,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.split_depth = split_depth
        blocks, pool = _resnet_blocks(norm)

        self.personals = nn.ModuleList(
            [copy.deepcopy(nn.Sequential(*blocks[:split_depth])) for _ in range(num_clusters)]
        )
        self.shared = nn.Sequential(*blocks[split_depth:])
        self.pool = pool
        self.heads = nn.ModuleList(
            [nn.Linear(FEATURE_DIM, num_classes, bias=False) for _ in range(num_clusters)]
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.combiner = build_combiner(combiner, num_clusters, num_classes, **combiner_kwargs)
        self.combiner_name = combiner

    def expert_logits(self, x: torch.Tensor, active: Optional[Sequence[int]] = None) -> torch.Tensor:
        idx = range(self.num_clusters) if active is None else active
        out = x.new_zeros(x.size(0), self.num_clusters, self.num_classes)
        for k in idx:
            feats = torch.flatten(self.pool(self.shared(self.personals[k](x))), 1)
            out[:, k, :] = self.heads[k](feats)
        return out

    def forward(self, x: torch.Tensor, active: Optional[Sequence[int]] = None, return_aux: bool = False):
        mask = torch.zeros(self.num_clusters, dtype=torch.bool, device=x.device)
        mask[list(range(self.num_clusters)) if active is None else list(active)] = True
        stack = self.expert_logits(x, active)
        combined, aux = self.combiner(stack, mask)
        combined = combined + self.bias
        if return_aux:
            aux["expert_logits"] = stack
            return combined, aux
        return combined

    def parameter_count(self) -> Dict[str, int]:
        p = sum(q.numel() for q in self.personals.parameters())
        s = sum(q.numel() for q in self.shared.parameters())
        h = sum(q.numel() for q in self.heads.parameters()) + self.bias.numel()
        return {"personal": p, "shared": s, "heads": h, "total": p + s + h}


# --------------------------------------------------------------------------- #
def ensemble_to_client_state(ens: SplitEnsemble, cluster: int) -> Dict[str, torch.Tensor]:
    state = {PERSONAL_PREFIX + k: v.detach().cpu().clone()
             for k, v in ens.personals[cluster].state_dict().items()}
    state.update({"shared." + k: v.detach().cpu().clone() for k, v in ens.shared.state_dict().items()})
    state.update({"pool." + k: v.detach().cpu().clone() for k, v in ens.pool.state_dict().items()})
    state["head.weight"] = ens.heads[cluster].weight.detach().cpu().clone()
    state["bias"] = ens.bias.detach().cpu().clone()
    return state


def load_client_state_into(ens: SplitEnsemble, cluster: int, state: Dict[str, torch.Tensor]) -> None:
    dev = ens.bias.device
    ens.personals[cluster].load_state_dict(
        {k[len(PERSONAL_PREFIX):]: v for k, v in state.items() if is_personal(k)})
    ens.shared.load_state_dict({k[len("shared."):]: v for k, v in state.items() if k.startswith("shared.")})
    with torch.no_grad():
        ens.heads[cluster].weight.copy_(state["head.weight"].to(dev))
        ens.bias.copy_(state["bias"].to(dev))


class SplitTrainer:
    """Federated training with a two-way aggregation rule.

    Personal blocks are averaged **within** their cluster; the shared trunk, and
    the bias, are averaged over **every** participating client. That single
    difference is what lets the trunk benefit from all the data while the early
    layers still specialise.
    """

    def __init__(
        self,
        data: FederatedData,
        client_clusters: np.ndarray,
        num_clusters: int,
        device: torch.device,
        split_depth: int = 2,
        norm: str = "gn",
        combiner: str = "uniform",
        num_classes: int = 10,
        batch_size: int = 64,
        lr: float = 0.01,
        logit_adjust: bool = False,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.clusters = np.asarray(client_clusters)
        self.num_clusters = num_clusters
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.rng = random.Random(seed)

        self.ensemble = SplitEnsemble(
            num_clusters, split_depth=split_depth, num_classes=num_classes,
            norm=norm, combiner=combiner,
        ).to(device)
        self.scratch = SplitClientView(split_depth, norm, num_classes).to(device)
        self.history: List[dict] = []
        self._val_loader = DataLoader(ConcatDataset(list(data.val_sets.values())),
                                      batch_size=256, shuffle=False)
        self._priors = ([label_log_prior(h) for h in data.client_label_hist]
                        if logit_adjust else None)

    def round(self, round_num: int, fraction: float, local_epochs: int) -> dict:
        n = max(1, int(round(fraction * self.data.num_clients)))
        selected = self.rng.sample(range(self.data.num_clients), n)

        per_cluster: Dict[int, List[Tuple[Dict, int]]] = {k: [] for k in range(self.num_clusters)}
        global_states: List[Tuple[Dict, int]] = []
        losses = []

        for cid in selected:
            k = int(self.clusters[cid])
            self.scratch.load_state_dict(ensemble_to_client_state(self.ensemble, k))
            loader = DataLoader(self.data.client_sets[cid], batch_size=self.batch_size, shuffle=True)
            losses.append(local_train(
                self.scratch, loader, self.device, epochs=local_epochs, lr=self.lr,
                label_log_prior=self._priors[cid] if self._priors else None))
            state = cpu_state(self.scratch)
            size = len(self.data.client_sets[cid])
            per_cluster[k].append(({q: v for q, v in state.items() if is_personal(q)}, size))
            global_states.append(({q: v for q, v in state.items() if not is_personal(q)}, size))

        # personal blocks: within-cluster average (skipped entirely at split_depth=0,
        # where there are no personal parameters to average)
        for k, updates in per_cluster.items():
            if not updates or not updates[0][0]:
                continue
            merged = weighted_average([u[0] for u in updates], [u[1] for u in updates])
            self.ensemble.personals[k].load_state_dict(
                {q[len(PERSONAL_PREFIX):]: v for q, v in merged.items()})

        # shared trunk + head + bias: global average over every participant.
        # This is the point of the design - the trunk is trained by all 40 clients,
        # so it never pays the data-splitting cost that sank the independent experts.
        merged = weighted_average([g[0] for g in global_states], [g[1] for g in global_states])
        shared_sd = {q[len("shared."):]: v for q, v in merged.items() if q.startswith("shared.")}
        if shared_sd:
            self.ensemble.shared.load_state_dict(shared_sd)
        with torch.no_grad():
            self.ensemble.bias.copy_(merged["bias"].to(self.device))
            for k in range(self.num_clusters):
                self.ensemble.heads[k].weight.copy_(merged["head.weight"].to(self.device))

        rec = {"round": round_num, "kind": "split", "mean_local_loss": float(np.mean(losses)),
               "participants": n}
        return rec

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        return evaluate(self.ensemble, self._val_loader, self.device)

    def fit(self, rounds: int, fraction: float = 0.25, local_epochs: int = 2,
            eval_every: int = 10, verbose: bool = True) -> List[dict]:
        for r in range(1, rounds + 1):
            rec = self.round(r, fraction, local_epochs)
            if eval_every > 0 and (r % eval_every == 0 or r == rounds):
                v = self.validate()
                rec["val_accuracy"], rec["val_loss"] = v["accuracy"], v["loss"]
            self.history.append(rec)
            if verbose:
                msg = f"    round {r:>3} [split ] local loss {rec['mean_local_loss']:.4f}"
                if "val_accuracy" in rec:
                    msg += f" | val acc {rec['val_accuracy']:.4f}"
                print(msg)
        return self.history
