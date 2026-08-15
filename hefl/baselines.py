"""
Baselines the ensemble has to beat.

``fedavg``
    One global model, no clustering.  The reference point for "does any of this
    help".  Uses the identical backbone, optimiser, round count, participation
    fraction and local epochs as the ensemble, so the only difference is the
    method.

``random_clusters``
    Not a model but a *cluster assignment*: the same K experts with the same
    total capacity, assigned to clients at random.  This is the ablation that
    makes the result falsifiable - K ResNet-18s beating one ResNet-18 says
    nothing until random assignment is shown to be worse than learned
    assignment.  Built here so the two paths share one code route.
"""

from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import FederatedData
from .models import ClientView
from .utils import cpu_state, label_log_prior, local_train, weighted_average


def train_fedavg(
    data: FederatedData,
    device: torch.device,
    rounds: int = 50,
    fraction: float = 0.2,
    local_epochs: int = 1,
    lr: float = 0.01,
    batch_size: int = 32,
    norm: str = "gn",
    num_classes: int = 10,
    logit_adjust: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> ClientView:
    """Standard FedAvg over all clients with a single shared model."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = ClientView(norm=norm, num_classes=num_classes).to(device)
    global_state = cpu_state(model)
    priors = [label_log_prior(h) for h in data.client_label_hist] if logit_adjust else None

    for r in range(1, rounds + 1):
        n = max(1, int(round(fraction * data.num_clients)))
        selected = rng.sample(range(data.num_clients), n)
        states, sizes, losses = [], [], []
        for cid in selected:
            model.load_state_dict(global_state)
            loader = DataLoader(data.client_sets[cid], batch_size=batch_size, shuffle=True)
            losses.append(
                local_train(model, loader, device, epochs=local_epochs, lr=lr,
                            label_log_prior=priors[cid] if priors else None)
            )
            states.append(cpu_state(model))
            sizes.append(len(data.client_sets[cid]))
        global_state = weighted_average(states, sizes)
        if verbose and (r % 5 == 0 or r == rounds):
            print(f"    [fedavg] round {r:>3}/{rounds} local loss {np.mean(losses):.4f}")

    model.load_state_dict(global_state)
    return model


def train_centralized(
    data: FederatedData,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 64,
    norm: str = "gn",
    num_classes: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> ClientView:
    """Train one model on the pooled client data - the centralized ceiling.

    This is what you could achieve if federation were not required: every
    client's data (rotations included) in one place, one model, ordinary SGD.
    It bounds the whole federated family from above, so it belongs in the table
    even though it is not a federated method.

    Note the mixture matters: the pooled set contains all rotation groups, so
    this single model has to fit every domain at once. That is precisely the
    problem cluster specialisation claims to solve, which makes the gap between
    this row and the ensemble row interpretable rather than decorative.
    """
    from torch.utils.data import ConcatDataset

    torch.manual_seed(seed)
    model = ClientView(norm=norm, num_classes=num_classes).to(device)
    pooled = ConcatDataset(data.client_sets)
    loader = DataLoader(pooled, batch_size=batch_size, shuffle=True)

    for e in range(1, epochs + 1):
        loss = local_train(model, loader, device, epochs=1, lr=lr)
        if verbose and (e % 5 == 0 or e == epochs):
            print(f"    [centralized] epoch {e:>3}/{epochs}  loss {loss:.4f}")
    return model


def random_cluster_assignment(num_clients: int, num_clusters: int, seed: int = 42) -> np.ndarray:
    """Balanced random client-to-cluster assignment (capacity-matched control)."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % num_clusters for i in range(num_clients)], dtype=np.int64)
    return rng.permutation(labels)


def oracle_cluster_assignment(rotation_groups: np.ndarray) -> np.ndarray:
    """Ground-truth assignment: one cluster per rotation group (upper bound)."""
    return np.asarray(rotation_groups, dtype=np.int64).copy()
