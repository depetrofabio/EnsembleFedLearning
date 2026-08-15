"""
Warmup and client clustering.

Four clustering signals are implemented so the two heterogeneity axes can be
separated instead of conflated:

``delta_l4fc``
    Weight delta of ``layer4`` + head after local warmup.  This is the v1
    signal.  The head gradient under cross-entropy is close to a direct readout
    of the client's label histogram, so this signal is *label-dominated*: it
    works well under Dirichlet-only heterogeneity and degrades once label skew
    and feature shift are present together.

``delta_stem``
    Weight delta of the early layers (``conv1``, first norm, ``layer1``).
    Domain shift lives in low-level filters, so this is the feature-axis
    counterpart of ``delta_l4fc``.

``act_stats``
    Spatially-gridded activation statistics of an early block, measured with a
    *frozen* backbone on a sample of client data.  A direct summary of the
    client's input distribution: a few hundred dimensions instead of millions,
    and no local training at all.  Recovers the rotation groups at ARI 1.00
    (IID labels) and 0.88 (Dirichlet 0.5) while staying uncorrelated with the
    label axis.  The spatial grid is what makes this work - see
    ``activation_statistics`` for the measurements and the failure mode.

``delta_full``
    Every float parameter.  Included as a reference point.

Reporting ARI against *both* ground truths (rotation group and label-histogram
cluster) for each signal is the experiment that shows which axis a signal
actually recovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from .models import ClientView
from .utils import cpu_state, local_train, weighted_average

SIGNALS = ("delta_l4fc", "delta_stem", "act_stats", "delta_full")


def _match(key: str, prefixes: Sequence[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


# Prefixes are matched against ``ClientView`` state-dict keys, which look like
# ``backbone.net.layer4.0.conv1.weight`` / ``head.weight`` / ``bias``.  Prefix
# matching matters here: a substring test for "conv1" would also catch
# ``layer2.0.conv1`` and silently turn the stem signal into a full-network one.
_PATTERNS = {
    "delta_l4fc": ("backbone.net.layer4", "head."),
    "delta_stem": ("backbone.net.conv1", "backbone.net.bn1", "backbone.net.layer1"),
    "delta_full": (),
}


def _flat_delta(before: Dict[str, torch.Tensor], after: Dict[str, torch.Tensor], signal: str) -> np.ndarray:
    """Flatten ``before - after`` over the parameters selected by ``signal``."""
    patterns = _PATTERNS[signal]
    chunks: List[torch.Tensor] = []
    for key, ref in before.items():
        if not torch.is_floating_point(ref):
            continue
        if patterns and not _match(key, patterns):
            continue
        chunks.append((ref - after[key]).reshape(-1).to(torch.float32))
    if not chunks:
        raise RuntimeError(f"signal '{signal}' selected no parameters")
    return torch.cat(chunks).cpu().numpy()


@torch.no_grad()
def activation_statistics(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 4,
    grid: int = 2,
) -> np.ndarray:
    """Spatially-gridded activation statistics of the stem, on client data.

    No training involved: the backbone is used purely as a fixed feature probe,
    so this costs a handful of forward passes per client and works even at
    random initialisation.

    The ``grid`` argument is the part that matters, and it is not a detail.
    Averaging each channel over the *whole* feature map (``grid=1``) throws away
    precisely the information a rotation changes - a rotated image has the same
    global channel means, so the descriptor is nearly rotation-blind and what
    survives is class content, i.e. the label axis.  Pooling to a coarse
    ``grid x grid`` map instead keeps the spatial layout ("sky at the top" vs
    "sky on the left"), which is a direct signature of the domain.

    Measured on 24 clients / 4 rotation groups / untrained backbone / 8 batches
    per client, mean ARI vs the rotation ground truth over seeds {0, 1, 2}:

        alpha        grid=1      grid=2
        -----------------------------------
        100  (IID)    0.905       1.000
        0.5           0.093       0.884
        0.1           0.020       0.382

    Two things to read off this.  First, the spatial grid is what makes the
    signal *robust to label skew*: with IID labels even global pooling works,
    but as soon as the label distribution skews, ``grid=1`` collapses to noise
    while ``grid=2`` holds.  Second, the signal itself degrades at alpha=0.1 -
    a client holding almost one class has different stem statistics for reasons
    that have nothing to do with its rotation, so the two axes stop being
    separable from activations alone.  That is a real limitation of this signal,
    not a tuning problem.

    ARI against the label axis stays near zero throughout for ``grid=2``, so
    what it recovers is the feature axis and only the feature axis.
    """
    net = backbone.net
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
    stem.eval().to(device)

    acc, count = None, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        feats = stem(batch[0].to(device))                          # (B, C, H, W)
        cells = F.adaptive_avg_pool2d(feats, grid).flatten(1)      # (B, C*grid*grid)
        spread = feats.flatten(2).std(-1)                          # (B, C)
        vec = torch.cat([cells, spread], dim=1)
        acc = vec.sum(0) if acc is None else acc + vec.sum(0)
        count += vec.size(0)
    if acc is None:
        raise RuntimeError("empty loader: cannot compute activation statistics")
    return (acc / count).cpu().numpy()


# --------------------------------------------------------------------------- #
@dataclass
class WarmupResult:
    """Signals collected during warmup plus the shared initial weights."""

    init_state: Dict[str, torch.Tensor]
    signals: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (N, D)

    def matrix(self, name: str) -> np.ndarray:
        if name not in self.signals:
            raise KeyError(f"signal '{name}' not collected; available: {sorted(self.signals)}")
        return self.signals[name]


def run_warmup(
    client_sets,
    device: torch.device,
    norm: str = "gn",
    num_classes: int = 10,
    fedavg_rounds: int = 1,
    local_epochs: int = 2,
    lr: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
    signals: Sequence[str] = ("delta_l4fc", "delta_stem", "act_stats"),
    act_batches: int = 4,
    act_grid: int = 2,
) -> WarmupResult:
    """Warm up a shared model, then collect one clustering signal per client.

    Phase A (``fedavg_rounds`` > 0): plain FedAvg over all clients, producing a
    shared initialisation that is better than random.  v1 referenced a
    ``warmup_round`` method that was never defined, so the FedAvg warmup path
    crashed; it is implemented here.

    Phase B: every client starts from the *same* warm state and trains locally
    for ``local_epochs``.  Signals are differences from that common origin,
    which is what makes them comparable across clients.  The warm state itself
    is not advanced by phase B.
    """
    torch.manual_seed(seed)
    scratch = ClientView(norm=norm, num_classes=num_classes).to(device)
    shared = cpu_state(scratch)
    sizes = [len(s) for s in client_sets]

    # -- Phase A: FedAvg ---------------------------------------------------- #
    for rnd in range(fedavg_rounds):
        states = []
        for cid, ds in enumerate(client_sets):
            scratch.load_state_dict(shared)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
            local_train(scratch, loader, device, epochs=local_epochs, lr=lr)
            states.append(cpu_state(scratch))
        shared = weighted_average(states, sizes)
        print(f"  [warmup] FedAvg round {rnd + 1}/{fedavg_rounds} done")

    result = WarmupResult(init_state=shared)
    need_delta = [s for s in signals if s != "act_stats"]
    collected: Dict[str, List[np.ndarray]] = {s: [] for s in signals}

    # -- Phase B: per-client probe ------------------------------------------ #
    for cid, ds in enumerate(client_sets):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        if "act_stats" in signals:
            scratch.load_state_dict(shared)
            collected["act_stats"].append(
                activation_statistics(scratch.backbone, loader, device,
                                      max_batches=act_batches, grid=act_grid)
            )

        if need_delta:
            scratch.load_state_dict(shared)
            before = cpu_state(scratch)
            local_train(scratch, loader, device, epochs=local_epochs, lr=lr)
            after = cpu_state(scratch)
            for s in need_delta:
                collected[s].append(_flat_delta(before, after, s))

        if (cid + 1) % 10 == 0 or cid == len(client_sets) - 1:
            print(f"  [warmup] probed client {cid + 1}/{len(client_sets)}")

    result.signals = {k: np.asarray(v) for k, v in collected.items() if v}
    return result


# --------------------------------------------------------------------------- #
def cluster_clients(
    matrix: np.ndarray,
    num_clusters: int,
    seed: int = 42,
    center: bool = True,
    pca_dim: Optional[int] = 32,
) -> Dict[str, object]:
    """Centre, L2-normalise, optionally PCA-reduce, then K-means.

    Centring removes the component every client shares (the direction they all
    move in), leaving what makes them *different*.  L2 normalisation turns
    Euclidean K-means into spherical K-means so distance reflects direction, not
    update magnitude - otherwise clients simply cluster by dataset size.  PCA is
    not cosmetic: a raw ``layer4`` delta is ~8.4M-dimensional with only N points,
    where Euclidean distances concentrate and K-means becomes unstable.
    """
    x = np.asarray(matrix, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    x = normalize(x, norm="l2")
    if pca_dim and pca_dim < min(x.shape):
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)
        x = normalize(x, norm="l2")

    km = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(x)
    sil = float(silhouette_score(x, labels)) if 1 < num_clusters < len(x) else float("nan")
    return {"labels": labels, "silhouette": sil, "embedding": x}


def label_axis_ground_truth(hist: np.ndarray, num_clusters: int, seed: int = 42) -> np.ndarray:
    """Reference grouping along the label axis: K-means on true label histograms."""
    km = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    return km.fit_predict(hist)


def clustering_report(
    warmup: WarmupResult,
    rotation_truth: np.ndarray,
    label_hist: np.ndarray,
    num_clusters: int,
    seed: int = 42,
    pca_dim: Optional[int] = 32,
) -> Dict[str, Dict[str, float]]:
    """Score every collected signal against both ground-truth axes.

    ``ari_rotation`` measures recovery of the feature axis, ``ari_label`` the
    label axis.  A signal that scores high on one and low on the other is doing
    exactly what you want; a signal that scores middling on both is conflating
    them, which is the diagnosis for the v1 ``layer4 + fc`` choice.
    """
    label_truth = label_axis_ground_truth(label_hist, num_clusters, seed)
    report: Dict[str, Dict[str, float]] = {}
    for name, matrix in warmup.signals.items():
        out = cluster_clients(matrix, num_clusters, seed=seed, pca_dim=pca_dim)
        labels = out["labels"]
        report[name] = {
            "ari_rotation": float(adjusted_rand_score(rotation_truth, labels)),
            "ari_label": float(adjusted_rand_score(label_truth, labels)),
            "silhouette": out["silhouette"],
            "dim": int(matrix.shape[1]),
            "labels": labels.tolist(),
        }
    return report
