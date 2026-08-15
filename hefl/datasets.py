"""
Data pipeline for the rotation + Dirichlet federated benchmark.

Three things happen here, and they are deliberately kept separate:

1.  *Feature heterogeneity* is injected by rotating images by a multiple of 90
    degrees.  Every client is assigned to exactly one rotation group, and every
    sample that client holds is rotated by that amount.  Rotation is applied on
    the raw uint8 array with ``np.rot90`` so it is exact and lossless.

2.  *Label heterogeneity* is injected by a Dirichlet(alpha) split of the class
    indices across clients, independently of the rotation assignment.

3.  The evaluation sets are built with the *same rotation mixture* as the client
    population.  This is the single most important protocol detail: evaluating a
    rotation-specialised ensemble on an un-rotated test set guarantees that
    K - 1 experts are out of domain and makes uniform averaging look broken when
    the real problem is the test distribution.

Splits
------
The CIFAR-10 *train* file is cut into three disjoint pools:

    train      -> handed to the clients (non-IID)
    combiner   -> held out, used ONLY to fit the combiner in server-side mode
    val        -> held out, used ONLY for model selection / oracle expert choice

``combiner`` and ``val`` are disjoint on purpose.  Fitting the combination layer
on the same data used to pick the best configuration would leak.

The CIFAR-10 *test* file is never touched during training and is materialised
once per rotation group so that per-rotation accuracy can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RotatedArrayDataset(Dataset):
    """CIFAR-10 samples held as a raw uint8 array with a per-sample rotation.

    Parameters
    ----------
    data:
        ``(N, 32, 32, 3)`` uint8 array.
    targets:
        ``(N,)`` int array of class labels.
    rotations:
        ``(N,)`` int array in ``{0, 1, 2, 3}``; the number of 90-degree
        counter-clockwise turns applied to the corresponding image.
    augment:
        If True, apply random crop + horizontal flip.  Off by default so the
        protocol matches the original repository (no augmentation anywhere).
    return_rotation:
        If True, ``__getitem__`` returns ``(image, label, rotation)``.  Used by
        the routing diagnostics.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        rotations: np.ndarray,
        augment: bool = False,
        return_rotation: bool = False,
    ) -> None:
        assert len(data) == len(targets) == len(rotations)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.rotations = np.asarray(rotations, dtype=np.int64)
        self.return_rotation = return_rotation
        self.augment = augment

        # Rotation is applied ONCE here, in bulk, rather than per __getitem__.
        # Every sample in a client set shares one rotation, and even the mixed
        # combiner pool has only a handful of distinct values, so this is a
        # vectorised np.rot90 per group instead of one call per sample fetch.
        data = np.ascontiguousarray(data)
        for k in np.unique(self.rotations):
            if k:
                sel = self.rotations == k
                data[sel] = np.rot90(data[sel], int(k), axes=(1, 2))
        self.data = data

        if augment:
            # Augmentation needs the PIL pipeline; kept as an opt-in slow path.
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        else:
            # Fast path: keep uint8 NCHW in a tensor and normalise on access.
            # No PIL round-trip, which is what dominates the input pipeline.
            self.transform = None
            self._tensor = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
            self._mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
            self._std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        if self.transform is not None:
            img = self.transform(Image.fromarray(self.data[index]))
        else:
            img = (self._tensor[index].float().div_(255.0) - self._mean) / self._std
        label = int(self.targets[index])
        if self.return_rotation:
            return img, label, int(self.rotations[index])
        return img, label

    def label_histogram(self, num_classes: int = NUM_CLASSES) -> np.ndarray:
        """Normalised class histogram; used as ground truth for the label axis."""
        counts = np.bincount(self.targets, minlength=num_classes).astype(np.float64)
        total = counts.sum()
        return counts / total if total > 0 else counts


# --------------------------------------------------------------------------- #
# Splitting helpers
# --------------------------------------------------------------------------- #
def _three_way_split(
    targets: np.ndarray,
    fractions: Tuple[float, float, float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified split of indices into (train, combiner, val)."""
    f_train, f_comb, f_val = fractions
    total = f_train + f_comb + f_val
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")

    train_idx, comb_idx, val_idx = [], [], []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(f_train * n))
        n_comb = int(round(f_comb * n))
        train_idx.append(idx[:n_train])
        comb_idx.append(idx[n_train : n_train + n_comb])
        val_idx.append(idx[n_train + n_comb :])

    out = tuple(rng.permutation(np.concatenate(p)) for p in (train_idx, comb_idx, val_idx))
    return out  # type: ignore[return-value]


def dirichlet_partition(
    targets: np.ndarray,
    indices: np.ndarray,
    num_clients: int,
    alpha: float,
    rng: np.random.Generator,
    min_size: int = 16,
    max_tries: int = 200,
) -> List[np.ndarray]:
    """Split ``indices`` across clients with a Dirichlet(alpha) label skew.

    For every class we draw a proportion vector ``p ~ Dir(alpha * 1_N)`` and hand
    each client its share of that class.  Small ``alpha`` -> highly skewed.
    Resamples until every client holds at least ``min_size`` samples so that
    local training and the weighted aggregation stay well defined.
    """
    labels = targets[indices]
    for _ in range(max_tries):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in np.unique(labels):
            cls_idx = indices[labels == cls]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            for client_id, part in enumerate(np.split(cls_idx, cuts)):
                buckets[client_id].extend(part.tolist())
        sizes = [len(b) for b in buckets]
        if min(sizes) >= min_size:
            return [np.array(sorted(b), dtype=np.int64) for b in buckets]
    raise RuntimeError(
        f"could not build a Dirichlet split with min_size={min_size} after {max_tries} tries; "
        f"lower num_clients, raise alpha, or lower min_size"
    )


def assign_rotation_groups(num_clients: int, rotations: Sequence[int], rng) -> np.ndarray:
    """Assign clients to rotation groups in balanced round-robin order."""
    groups = np.array([i % len(rotations) for i in range(num_clients)], dtype=np.int64)
    return rng.permutation(groups)


# --------------------------------------------------------------------------- #
# Top-level builder
# --------------------------------------------------------------------------- #
@dataclass
class FederatedData:
    """Everything the orchestrator needs, plus the ground truth for diagnostics."""

    client_sets: List[RotatedArrayDataset]
    client_rotation: np.ndarray                     # (num_clients,) ground-truth group
    client_label_hist: np.ndarray                   # (num_clients, 10)
    combiner_set: RotatedArrayDataset               # held-out, server-side combiner fitting
    val_sets: Dict[int, RotatedArrayDataset]        # per rotation, model selection
    test_sets: Dict[int, RotatedArrayDataset]       # per rotation, final evaluation
    rotations: List[int] = field(default_factory=list)

    @property
    def num_clients(self) -> int:
        return len(self.client_sets)

    def client_sizes(self) -> List[int]:
        return [len(s) for s in self.client_sets]


def build_federated_data(
    root: str,
    num_clients: int,
    alpha: float,
    rotation_groups: Sequence[int],
    seed: int = 42,
    split_fractions: Tuple[float, float, float] = (0.80, 0.10, 0.10),
    augment: bool = False,
    download: bool = True,
    subsample: Optional[int] = None,
) -> FederatedData:
    """Build the full federated benchmark.

    ``rotation_groups`` lists the number of 90-degree turns per group, e.g.
    ``[0, 1, 2, 3]`` for 0/90/180/270 or ``[0]`` to disable feature shift.
    ``subsample`` keeps only that many training images (smoke tests).
    """
    rng = np.random.default_rng(seed)

    train_raw = datasets.CIFAR10(root=root, train=True, download=download)
    test_raw = datasets.CIFAR10(root=root, train=False, download=download)

    train_data = np.asarray(train_raw.data)
    train_targets = np.asarray(train_raw.targets, dtype=np.int64)
    test_data = np.asarray(test_raw.data)
    test_targets = np.asarray(test_raw.targets, dtype=np.int64)

    if subsample is not None:
        keep = rng.permutation(len(train_data))[:subsample]
        train_data, train_targets = train_data[keep], train_targets[keep]
        keep_t = rng.permutation(len(test_data))[: max(subsample // 5, len(rotation_groups) * 50)]
        test_data, test_targets = test_data[keep_t], test_targets[keep_t]

    train_idx, comb_idx, val_idx = _three_way_split(train_targets, split_fractions, rng)

    # --- clients: rotation group x Dirichlet label skew ------------------- #
    client_rotation = assign_rotation_groups(num_clients, rotation_groups, rng)
    parts = dirichlet_partition(train_targets, train_idx, num_clients, alpha, rng)

    client_sets, client_hist = [], []
    for cid, idx in enumerate(parts):
        turns = rotation_groups[client_rotation[cid]]
        ds = RotatedArrayDataset(
            data=train_data[idx],
            targets=train_targets[idx],
            rotations=np.full(len(idx), turns, dtype=np.int64),
            augment=augment,
        )
        client_sets.append(ds)
        client_hist.append(ds.label_histogram())

    # --- combiner pool: same rotation mixture as the clients --------------- #
    # Each held-out sample is assigned a rotation drawn from the empirical group
    # frequencies, so the combiner sees the deployment distribution.
    group_freq = np.bincount(client_rotation, minlength=len(rotation_groups)).astype(float)
    group_freq /= group_freq.sum()
    comb_groups = rng.choice(len(rotation_groups), size=len(comb_idx), p=group_freq)
    combiner_set = RotatedArrayDataset(
        data=train_data[comb_idx],
        targets=train_targets[comb_idx],
        rotations=np.array([rotation_groups[g] for g in comb_groups], dtype=np.int64),
        return_rotation=True,
    )

    # --- val / test: one materialised copy per rotation group -------------- #
    def per_rotation(data: np.ndarray, targets: np.ndarray) -> Dict[int, RotatedArrayDataset]:
        return {
            g: RotatedArrayDataset(
                data=data,
                targets=targets,
                rotations=np.full(len(data), turns, dtype=np.int64),
                return_rotation=True,
            )
            for g, turns in enumerate(rotation_groups)
        }

    val_sets = per_rotation(train_data[val_idx], train_targets[val_idx])
    test_sets = per_rotation(test_data, test_targets)

    return FederatedData(
        client_sets=client_sets,
        client_rotation=client_rotation,
        client_label_hist=np.asarray(client_hist),
        combiner_set=combiner_set,
        val_sets=val_sets,
        test_sets=test_sets,
        rotations=list(rotation_groups),
    )
