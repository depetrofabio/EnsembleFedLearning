"""Small shared helpers: seeding, local training, evaluation, FedAvg."""

from __future__ import annotations

import random
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def local_train(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    params: Optional[Iterable[nn.Parameter]] = None,
    label_log_prior: Optional[torch.Tensor] = None,
) -> float:
    """Run ``epochs`` of local SGD and return the mean loss.

    ``label_log_prior`` enables logit adjustment: adding ``log p_client(y)`` to
    the logits during local training corrects for the client's label skew and is
    a near-free win under Dirichlet partitioning.  Pass ``None`` to disable.
    """
    model.train()
    trainable = list(params) if params is not None else [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    total, seen = 0.0, 0
    for _ in range(epochs):
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]
            if label_log_prior is not None:
                logits = logits + label_log_prior.to(logits.device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.size(0)
            seen += labels.size(0)
    return total / max(seen, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_kwargs: Optional[dict] = None,
) -> Dict[str, float]:
    """Return loss and accuracy on ``loader``."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    forward_kwargs = forward_kwargs or {}
    loss_sum, correct, seen = 0.0, 0, 0
    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        logits = model(inputs, **forward_kwargs)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)
    return {"loss": loss_sum / max(seen, 1), "accuracy": correct / max(seen, 1)}


def weighted_average(states: Sequence[Dict[str, torch.Tensor]], sizes: Sequence[int]) -> Dict[str, torch.Tensor]:
    """FedAvg over a list of state dicts, weighted by client sample count.

    Integer buffers (e.g. ``num_batches_tracked``) are copied from the first
    client rather than averaged - averaging them silently promotes them to float
    and they carry no useful information anyway.
    """
    if not states:
        raise ValueError("nothing to aggregate")
    total = float(sum(sizes))
    out: Dict[str, torch.Tensor] = {}
    for key, ref in states[0].items():
        if not torch.is_floating_point(ref):
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for state, size in zip(states, sizes):
            acc += state[key].to(torch.float32) * (size / total)
        out[key] = acc.to(ref.dtype)
    return out


def cpu_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def label_log_prior(hist: np.ndarray, eps: float = 1e-8) -> torch.Tensor:
    """``log p_client(y)`` for logit adjustment."""
    return torch.log(torch.tensor(hist, dtype=torch.float32) + eps)
