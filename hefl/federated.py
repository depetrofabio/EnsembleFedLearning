"""
Federated orchestration: warmup -> clustering -> expert rounds -> dense rounds.

Two things differ substantively from v1 beyond the model refactor.

Aggregation
-----------
v1 averaged the classifier over *all* participating clients.  A cluster-c client
only ever changes block c and returns everyone else's blocks unmodified, so
block c's effective update was silently scaled by ``n_c / n_total`` - an
unintended per-cluster learning-rate divisor that also confounds any sweep over
the number of clusters.  Here backbone_k and head_k are averaged over the
clients of cluster k only, and the shared bias over everyone, which is what the
math actually calls for.

Dense rounds
------------
Normal rounds keep one expert active, so they can never train the combination.
Every ``dense_every`` rounds the experts are frozen and the combiner is trained
with all K experts active - federated (clients receive the frozen experts) or
server-side (stacking on a held-out split).  This is the only place the
combination is learned rather than assumed.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from .datasets import FederatedData
from .models import ClientView, ClusterEnsemble
from .utils import cpu_state, evaluate, label_log_prior, local_train, weighted_average

BACKBONE_PREFIX = "backbone."


# --------------------------------------------------------------------------- #
# state-dict plumbing between the server-side ensemble and the client view
# --------------------------------------------------------------------------- #
def ensemble_to_client_state(ens: ClusterEnsemble, cluster: int) -> Dict[str, torch.Tensor]:
    state = {BACKBONE_PREFIX + k: v.detach().cpu().clone()
             for k, v in ens.backbones[cluster].state_dict().items()}
    state["head.weight"] = ens.heads[cluster].weight.detach().cpu().clone()
    state["bias"] = ens.bias.detach().cpu().clone()
    return state


def split_client_state(state: Dict[str, torch.Tensor]):
    backbone = {k[len(BACKBONE_PREFIX):]: v for k, v in state.items() if k.startswith(BACKBONE_PREFIX)}
    return backbone, state["head.weight"], state["bias"]


def init_ensemble_from_warmup(
    ens: ClusterEnsemble, warm_state: Dict[str, torch.Tensor]
) -> None:
    """Seed every expert from the same warm state.

    All K experts therefore start as identical functions.  Because the combiner
    weights sum to one, the ensemble's output at round 0 is *exactly* the warmup
    model's output for any active subset - the ensemble starts from a known
    reference point rather than from a random one.
    """
    backbone, head_w, bias = split_client_state(warm_state)
    for k in range(ens.num_clusters):
        ens.backbones[k].load_state_dict(backbone)
        with torch.no_grad():
            ens.heads[k].weight.copy_(head_w)
    with torch.no_grad():
        ens.bias.copy_(bias)


# --------------------------------------------------------------------------- #
@dataclass
class RoundRecord:
    round: int
    kind: str                      # "expert" | "dense"
    mean_local_loss: float
    participants: int
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


class EnsembleTrainer:
    """Owns the server-side ensemble and drives the federated schedule."""

    def __init__(
        self,
        data: FederatedData,
        client_clusters: np.ndarray,
        num_clusters: int,
        device: torch.device,
        norm: str = "gn",
        combiner: str = "gate",
        num_classes: int = 10,
        batch_size: int = 32,
        lr: float = 0.01,
        combiner_lr: float = 0.05,
        logit_adjust: bool = False,
        routing_weight: float = 0.5,
        seed: int = 42,
        **combiner_kwargs,
    ) -> None:
        self.data = data
        self.clusters = np.asarray(client_clusters)
        self.num_clusters = num_clusters
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.combiner_lr = combiner_lr
        self.logit_adjust = logit_adjust
        self.routing_weight = routing_weight
        self.rng = random.Random(seed)

        self.ensemble = ClusterEnsemble(
            num_clusters=num_clusters,
            num_classes=num_classes,
            norm=norm,
            combiner=combiner,
            **combiner_kwargs,
        ).to(device)

        # one reusable scratch client, so a round costs no extra allocations
        self.scratch = ClientView(norm=norm, num_classes=num_classes).to(device)
        self.history: List[RoundRecord] = []

        self._val_loader = DataLoader(
            ConcatDataset(list(data.val_sets.values())),
            batch_size=256,
            shuffle=False,
        )
        self._priors = (
            [label_log_prior(h) for h in data.client_label_hist] if logit_adjust else None
        )

    # -- expert round ------------------------------------------------------- #
    def expert_round(self, round_num: int, fraction: float, local_epochs: int) -> RoundRecord:
        """One standard round: each client trains its own expert + the bias."""
        n = max(1, int(round(fraction * self.data.num_clients)))
        selected = self.rng.sample(range(self.data.num_clients), n)

        per_cluster: Dict[int, List[Tuple[Dict[str, torch.Tensor], int]]] = {
            k: [] for k in range(self.num_clusters)
        }
        bias_states: List[Tuple[Dict[str, torch.Tensor], int]] = []
        losses = []

        for cid in selected:
            k = int(self.clusters[cid])
            self.scratch.load_state_dict(ensemble_to_client_state(self.ensemble, k))
            loader = DataLoader(self.data.client_sets[cid], batch_size=self.batch_size, shuffle=True)
            loss = local_train(
                self.scratch,
                loader,
                self.device,
                epochs=local_epochs,
                lr=self.lr,
                label_log_prior=self._priors[cid] if self._priors else None,
            )
            losses.append(loss)
            state = cpu_state(self.scratch)
            size = len(self.data.client_sets[cid])
            per_cluster[k].append((state, size))
            bias_states.append(({"bias": state["bias"]}, size))

        # backbone_k and head_k: cluster-local average (the fix)
        for k, updates in per_cluster.items():
            if not updates:
                continue
            states = [u[0] for u in updates]
            sizes = [u[1] for u in updates]
            merged = weighted_average(states, sizes)
            backbone, head_w, _ = split_client_state(merged)
            self.ensemble.backbones[k].load_state_dict(backbone)
            with torch.no_grad():
                self.ensemble.heads[k].weight.copy_(head_w.to(self.device))

        # bias: global average over every participant
        merged_bias = weighted_average([b[0] for b in bias_states], [b[1] for b in bias_states])
        with torch.no_grad():
            self.ensemble.bias.copy_(merged_bias["bias"].to(self.device))

        return RoundRecord(
            round=round_num,
            kind="expert",
            mean_local_loss=float(np.mean(losses)),
            participants=n,
        )

    # -- dense round -------------------------------------------------------- #
    def dense_round(
        self,
        round_num: int,
        mode: str = "clients",
        fraction: float = 0.2,
        epochs: int = 1,
        max_batches: Optional[int] = None,
    ) -> RoundRecord:
        """Train the combiner with all K experts active; experts stay frozen."""
        if self.ensemble.combiner_name == "uniform":
            return RoundRecord(round_num, "dense", float("nan"), 0, extra={"skipped": 1.0})

        self.ensemble.freeze_experts(True)
        try:
            if mode == "clients":
                rec = self._dense_federated(round_num, fraction, epochs, max_batches)
            elif mode == "server":
                rec = self._dense_server(round_num, epochs, max_batches)
            else:
                raise ValueError(f"dense mode must be 'clients' or 'server', got {mode}")
        finally:
            self.ensemble.freeze_experts(False)
        return rec

    def _combiner_step(
        self,
        loader: DataLoader,
        epochs: int,
        cluster_target: Optional[int],
        max_batches: Optional[int],
        rotation_to_cluster: Optional[Dict[int, int]] = None,
    ) -> float:
        """Optimise only the combiner.  Experts are evaluated under no_grad."""
        params = [p for p in self.ensemble.combiner.parameters() if p.requires_grad]
        if not params:
            return float("nan")
        opt = torch.optim.SGD(params, lr=self.combiner_lr, momentum=0.9)
        mask = torch.ones(self.num_clusters, dtype=torch.bool, device=self.device)
        total, seen = 0.0, 0

        self.ensemble.eval()          # frozen experts: keep norm layers in eval mode
        self.ensemble.combiner.train()

        for _ in range(epochs):
            for i, batch in enumerate(loader):
                if max_batches is not None and i >= max_batches:
                    break
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                with torch.no_grad():
                    stack = self.ensemble.expert_logits(x, active=None)
                combined, aux = self.ensemble.combiner(stack, mask)
                combined = combined + self.ensemble.bias.detach()
                loss = F.cross_entropy(combined, y)

                # free routing supervision: the client knows its own cluster id
                if "routing_logits" in aux and self.routing_weight > 0:
                    if cluster_target is not None:
                        tgt = torch.full((y.size(0),), cluster_target, device=self.device, dtype=torch.long)
                    elif rotation_to_cluster is not None and len(batch) > 2:
                        rot = batch[2]
                        tgt = torch.tensor(
                            [rotation_to_cluster.get(int(r), 0) for r in rot],
                            device=self.device, dtype=torch.long,
                        )
                    else:
                        tgt = None
                    if tgt is not None:
                        loss = loss + self.routing_weight * F.cross_entropy(aux["routing_logits"], tgt)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * y.size(0)
                seen += y.size(0)
        return total / max(seen, 1)

    def _dense_federated(self, round_num, fraction, epochs, max_batches) -> RoundRecord:
        n = max(1, int(round(fraction * self.data.num_clients)))
        selected = self.rng.sample(range(self.data.num_clients), n)
        base = copy.deepcopy(self.ensemble.combiner.state_dict())

        states, sizes, losses = [], [], []
        for cid in selected:
            self.ensemble.combiner.load_state_dict(base)      # every client starts from the server copy
            loader = DataLoader(self.data.client_sets[cid], batch_size=self.batch_size, shuffle=True)
            loss = self._combiner_step(
                loader, epochs, cluster_target=int(self.clusters[cid]), max_batches=max_batches
            )
            losses.append(loss)
            states.append({k: v.detach().cpu().clone()
                           for k, v in self.ensemble.combiner.state_dict().items()})
            sizes.append(len(self.data.client_sets[cid]))

        self.ensemble.combiner.load_state_dict(weighted_average(states, sizes))
        return RoundRecord(round_num, "dense", float(np.nanmean(losses)), n,
                           extra={"mode_clients": 1.0})

    def _dense_server(self, round_num, epochs, max_batches) -> RoundRecord:
        loader = DataLoader(self.data.combiner_set, batch_size=self.batch_size, shuffle=True)
        loss = self._combiner_step(loader, epochs, cluster_target=None, max_batches=max_batches,
                                   rotation_to_cluster=self.rotation_to_cluster())
        return RoundRecord(round_num, "dense", loss, 0, extra={"mode_server": 1.0})

    # -- diagnostics helpers ------------------------------------------------ #
    def cluster_to_rotation(self) -> Dict[int, int]:
        """Majority rotation group per cluster (analysis only, never used to train)."""
        out = {}
        for k in range(self.num_clusters):
            members = np.where(self.clusters == k)[0]
            if len(members) == 0:
                out[k] = -1
                continue
            rots = self.data.client_rotation[members]
            out[k] = int(np.bincount(rots).argmax())
        return out

    def rotation_to_cluster(self) -> Dict[int, int]:
        """Inverse map; ties broken by the largest cluster for that rotation."""
        c2r = self.cluster_to_rotation()
        out: Dict[int, int] = {}
        for g in range(len(self.data.rotations)):
            candidates = [k for k, r in c2r.items() if r == g]
            if candidates:
                out[g] = max(candidates, key=lambda k: int((self.clusters == k).sum()))
        turns_to_group = {t: g for g, t in enumerate(self.data.rotations)}
        return {t: out[g] for t, g in turns_to_group.items() if g in out}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        return evaluate(self.ensemble, self._val_loader, self.device)

    # -- schedule ----------------------------------------------------------- #
    def fit(
        self,
        rounds: int,
        fraction: float = 0.2,
        local_epochs: int = 1,
        dense_every: int = 10,
        dense_mode: str = "clients",
        dense_fraction: float = 0.2,
        dense_epochs: int = 1,
        dense_max_batches: Optional[int] = None,
        eval_every: int = 5,
        verbose: bool = True,
    ) -> List[RoundRecord]:
        for r in range(1, rounds + 1):
            rec = self.expert_round(r, fraction, local_epochs)

            if dense_every > 0 and (r % dense_every == 0 or r == rounds):
                dense = self.dense_round(
                    r, mode=dense_mode, fraction=dense_fraction,
                    epochs=dense_epochs, max_batches=dense_max_batches,
                )
                self.history.append(dense)
                if verbose and not dense.extra.get("skipped"):
                    print(f"    round {r:>3} [dense ] combiner loss {dense.mean_local_loss:.4f}")

            if eval_every > 0 and (r % eval_every == 0 or r == rounds):
                val = self.validate()
                rec.val_accuracy, rec.val_loss = val["accuracy"], val["loss"]

            self.history.append(rec)
            if verbose:
                msg = f"    round {r:>3} [expert] local loss {rec.mean_local_loss:.4f}"
                if rec.val_accuracy is not None:
                    msg += f" | val acc {rec.val_accuracy:.4f}"
                print(msg)
        return self.history
