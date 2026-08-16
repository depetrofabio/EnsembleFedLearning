"""
Evaluation protocol.

The table this module produces is the point of the whole repository:

    method              | 0deg | 90deg | 180deg | 270deg | overall
    --------------------+------+-------+--------+--------+--------
    FedAvg (1 model)    |      |       |        |        |
    expert k (each)     |      |       |        |        |   <- diagonal check
    ensemble, uniform   |      |       |        |        |
    ensemble, beta      |      |       |        |        |
    ensemble, gate      |      |       |        |        |
    oracle expert       |      |       |        |        |   <- ceiling

Three numbers carry the argument: the *diagonal* of the per-expert block (does
expert k win on rotation k - which validates clustering and specialisation at
once), the *gate vs oracle* gap (what routing costs), and *gate vs FedAvg* (does
the method work at all).

Every set here is materialised per rotation group, so "overall" is a uniform
mixture over the groups - the same mixture the client population has.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import ClusterEnsemble


def _loaders(sets: Dict[int, object], batch_size: int = 256) -> Dict[int, DataLoader]:
    return {g: DataLoader(ds, batch_size=batch_size, shuffle=False) for g, ds in sets.items()}


@torch.no_grad()
def accuracy_per_rotation(
    model: torch.nn.Module,
    sets: Dict[int, object],
    device: torch.device,
    active: Optional[Sequence[int]] = None,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Accuracy on each rotation group plus the uniform-mixture overall."""
    model.eval()
    per_group, correct_all, seen_all = {}, 0, 0
    for g, loader in _loaders(sets, batch_size).items():
        correct, seen = 0, 0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            # Duck-typing, not isinstance: SplitEnsemble is not a ClusterEnsemble
            # subclass, and an isinstance check silently DROPPED `active` for it,
            # making every "expert k only" row identical to the full ensemble.
            supports_active = hasattr(model, "expert_logits")
            logits = model(x, active=active) if supports_active else model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        per_group[g] = correct / max(seen, 1)
        correct_all += correct
        seen_all += seen
    per_group_named = {f"rot_{g}": v for g, v in per_group.items()}
    per_group_named["overall"] = correct_all / max(seen_all, 1)
    return per_group_named


def expert_matrix(
    ensemble: ClusterEnsemble,
    sets: Dict[int, object],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """``(K, G)`` matrix of single-expert accuracy per rotation group.

    Row k is expert k evaluated alone.  If clustering worked and the experts
    specialised, this matrix is diagonally dominant after permuting columns by
    the cluster-to-rotation map.
    """
    groups = sorted(sets.keys())
    out = np.zeros((ensemble.num_clusters, len(groups)))
    for k in range(ensemble.num_clusters):
        res = accuracy_per_rotation(ensemble, sets, device, active=[k], batch_size=batch_size)
        for j, g in enumerate(groups):
            out[k, j] = res[f"rot_{g}"]
    return out


def oracle_from_validation(
    ensemble: ClusterEnsemble,
    val_sets: Dict[int, object],
    test_sets: Dict[int, object],
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[Dict[str, float], Dict[int, int]]:
    """Ceiling: pick the best expert per rotation on *validation*, apply to test.

    The choice is made on held-out validation data, never on test, so this is an
    honest upper bound for "you know the domain of each sample" rather than a
    number read off the test set.
    """
    val_mat = expert_matrix(ensemble, val_sets, device, batch_size)
    best = {g: int(val_mat[:, j].argmax()) for j, g in enumerate(sorted(val_sets.keys()))}

    correct_all, seen_all, per_group = 0, 0, {}
    for g, expert in best.items():
        res = accuracy_per_rotation(ensemble, {g: test_sets[g]}, device, active=[expert], batch_size=batch_size)
        acc = res[f"rot_{g}"]
        n = len(test_sets[g])
        per_group[f"rot_{g}"] = acc
        correct_all += acc * n
        seen_all += n
    per_group["overall"] = correct_all / max(seen_all, 1)
    return per_group, best


@torch.no_grad()
def routing_accuracy(
    ensemble: ClusterEnsemble,
    sets: Dict[int, object],
    device: torch.device,
    rotation_to_cluster: Dict[int, int],
    rotations: Sequence[int],
    batch_size: int = 256,
) -> Dict[str, float]:
    """How often the gate's argmax expert matches the sample's true domain.

    Only defined for combiners that expose routing logits.  ``rotation_to_cluster``
    maps a number of turns to the cluster that owns that domain, derived from the
    clustering result - it is a diagnostic, never a training signal here.
    """
    if not getattr(ensemble.combiner, "supports_routing", False):
        return {}
    ensemble.eval()
    per_group, hits_all, seen_all = {}, 0, 0
    for g, loader in _loaders(sets, batch_size).items():
        turns = rotations[g]
        target = rotation_to_cluster.get(turns)
        if target is None:
            continue
        hits, seen = 0, 0
        for batch in loader:
            x = batch[0].to(device)
            _, aux = ensemble(x, active=None, return_aux=True)
            choice = aux["weights"].argmax(1)
            hits += (choice == target).sum().item()
            seen += x.size(0)
        per_group[f"rot_{g}"] = hits / max(seen, 1)
        hits_all += hits
        seen_all += seen
    per_group["overall"] = hits_all / max(seen_all, 1)
    return per_group


def subset_analysis(
    ensemble: ClusterEnsemble,
    sets: Dict[int, object],
    device: torch.device,
    max_subsets: int = 63,
    batch_size: int = 256,
) -> List[Dict[str, object]]:
    """Accuracy of every active subset of experts, weights renormalised.

    This is v1's "cluster combination analysis", but now the renormalisation is
    part of the combiner (masked softmax) instead of a ``/ num_active`` constant,
    so every subset is magnitude-correct by construction.
    """
    ks = list(range(ensemble.num_clusters))
    rows = []
    for size in range(1, len(ks) + 1):
        for subset in combinations(ks, size):
            if len(rows) >= max_subsets:
                return rows
            res = accuracy_per_rotation(ensemble, sets, device, active=list(subset), batch_size=batch_size)
            rows.append({"subset": list(subset), "size": size, **res})
    return rows


def markdown_table(rows: List[Dict[str, object]], groups: Sequence[int], rotations: Sequence[int]) -> str:
    """Render the headline table as markdown."""
    header = ["method"] + [f"{rotations[g] * 90}deg" for g in groups] + ["overall"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        cells = [str(row["method"])]
        for g in groups:
            v = row.get(f"rot_{g}")
            cells.append(f"{v:.4f}" if isinstance(v, float) else "-")
        overall = row.get("overall")
        cells.append(f"{overall:.4f}" if isinstance(overall, float) else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
