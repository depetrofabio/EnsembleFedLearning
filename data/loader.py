from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def _build_transform(mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    """Return a basic tensor + normalization transform."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _stratified_split(
    labels: Sequence[int],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split indices into train/val subsets preserving label distribution."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must belong to [0, 1).")

    rng = random.Random(seed)
    label_to_indices: Dict[int, List[int]] = {}

    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for label, indices in label_to_indices.items():
        rng.shuffle(indices)
        split = int(len(indices) * (1 - val_fraction))
        split = min(max(split, 0), len(indices))
        # ensure both splits contain samples if val_fraction > 0
        if val_fraction > 0 and split == len(indices):
            split -= 1

        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _extract_targets(dataset: Dataset) -> List[int]:
    """Return targets for a dataset or subset."""
    if isinstance(dataset, Subset):
        parent_targets = _extract_targets(dataset.dataset)
        return [parent_targets[i] for i in dataset.indices]

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "labels"):
        targets = dataset.labels
    else:
        raise AttributeError("Dataset must expose `targets` or `labels`.")

    if isinstance(targets, torch.Tensor):
        return targets.tolist()
    return list(targets)


def load_cifar10(
    mean: Sequence[float],
    std: Sequence[float],
    data_dir: str | Path = "experiments/data",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, datasets.CIFAR10]:
    """
    Download (if needed) CIFAR-10, normalize with provided statistics and
    return stratified train/val subsets plus the normalized test set.
    """
    data_root = Path(data_dir)
    transform = _build_transform(mean, std)

    full_train = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    train_idx, val_idx = _stratified_split(full_train.targets, val_fraction, seed)
    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)

    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    return train_subset, val_subset, test_set


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap datasets in PyTorch dataloaders with common defaults."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def dirichlet_split(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int = 0,
    min_samples_per_client: int = 1,
    max_retries: int = 20,
) -> List[Subset]:
    """
    Split a dataset into `num_clients` shards whose class proportions follow a
    Dirichlet(alpha) distribution. Useful to simulate non-i.i.d. FL clients.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if min_samples_per_client < 0:
        raise ValueError("min_samples_per_client cannot be negative.")

    targets = np.array(_extract_targets(dataset))
    unique_labels = np.unique(targets)
    rng = np.random.default_rng(seed)

    for attempt in range(max_retries):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for label in unique_labels:
            label_indices = np.where(targets == label)[0]
            rng.shuffle(label_indices)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            splits = (np.cumsum(proportions) * len(label_indices)).astype(int)
            splits = np.clip(splits, 0, len(label_indices))
            shards = np.split(label_indices, splits[:-1])

            for client_id, shard in enumerate(shards):
                client_indices[client_id].extend(shard.tolist())

        if min_samples_per_client == 0 or all(
            len(idxs) >= min_samples_per_client for idxs in client_indices
        ):
            for idxs in client_indices:
                rng.shuffle(idxs)
            return [Subset(dataset, idxs) for idxs in client_indices]

    raise RuntimeError(
        "Unable to sample Dirichlet split with the requested minimum size. "
        "Try lowering `min_samples_per_client` or reducing `num_clients`."
    )


def preview_samples(
    dataset: Dataset,
    mean: Sequence[float],
    std: Sequence[float],
    class_names: Iterable[str] | None = None,
    num_images: int = 9,
    seed: int = 0,
) -> None:
    """
    Display a grid of images from the dataset with their associated labels.

    The dataset is assumed to output normalized tensors; mean/std are used
    to undo normalization for visualization.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot preview samples from an empty dataset.")

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(num_images, len(dataset)))

    rows = int(np.ceil(len(indices) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    axes = np.array(axes).reshape(-1)

    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    for ax, idx in zip(axes, indices):
        image, label = dataset[idx]
        image = image.cpu() * std_tensor + mean_tensor
        image = image.clamp(0, 1)
        ax.imshow(image.permute(1, 2, 0).numpy())

        title = str(label)
        if class_names is not None:
            try:
                title = class_names[label]
            except Exception:
                title = f"{label}"
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(indices) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
