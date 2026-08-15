"""
Invariant tests for the properties the report makes claims about.

These are cheap (seconds, CPU, tiny tensors) and they check the things that are
easy to break silently and hard to notice in an accuracy number:

1. The uniform combiner reproduces v1's forward pass exactly.
2. Single-active equivalence: what a client trains == what the ensemble
   evaluates with only that expert active.  This is the property that replaces
   v1's "Active Normalization".
3. Combination weights sum to 1 over every active subset.
4. FedAvg averaging is correct and does not corrupt integer buffers.
5. Cluster-local aggregation does not dilute the update the way v1's global
   classifier averaging did.
6. Both dense-round modes run and actually move the combiner.

    python -m hefl.test_invariants
"""

import numpy as np
import torch

from .datasets import build_federated_data
from .federated import (
    EnsembleTrainer,
    ensemble_to_client_state,
    init_ensemble_from_warmup,
)
from .models import ClientView, ClusterEnsemble, build_combiner
from .utils import cpu_state, set_seed, weighted_average

PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(ok)
    print(f"{PASS if ok else FAIL}  {name}{('  -> ' + detail) if detail else ''}")


# --------------------------------------------------------------------------- #
def test_uniform_matches_v1() -> None:
    """v1: logits = W @ (concat(z)/|A|) + b.  v2 uniform must equal that."""
    set_seed(0)
    K, C, D, B = 4, 10, 512, 3
    ens = ClusterEnsemble(K, C, norm="gn", combiner="uniform").eval()
    x = torch.randn(B, 3, 32, 32)

    with torch.no_grad():
        v2 = ens(x)
        # explicit v1 reconstruction
        z = [ens.backbones[k](x) for k in range(K)]
        W = torch.cat([ens.heads[k].weight for k in range(K)], dim=1)   # (C, K*D)
        concat = torch.cat(z, dim=1) / K                                # (B, K*D)
        v1 = concat @ W.T + ens.bias

    check("uniform combiner == v1 forward pass",
          torch.allclose(v1, v2, atol=1e-4),
          f"max |diff| = {(v1 - v2).abs().max():.2e}")


def test_single_active_equivalence() -> None:
    """ensemble(x, active=[c]) must equal the ClientView the client trains."""
    set_seed(0)
    K, C = 3, 10
    x = torch.randn(2, 3, 32, 32)
    for combiner in ("uniform", "beta", "gate"):
        ens = ClusterEnsemble(K, C, norm="gn", combiner=combiner).eval()
        client = ClientView(norm="gn", num_classes=C).eval()
        c = 1
        client.load_state_dict(ensemble_to_client_state(ens, c))
        with torch.no_grad():
            a = ens(x, active=[c])
            b = client(x)
        check(f"single-active equivalence ({combiner})",
              torch.allclose(a, b, atol=1e-5),
              f"max |diff| = {(a - b).abs().max():.2e}")


def test_weights_sum_to_one() -> None:
    """Every weight-producing combiner normalises over the active subset."""
    set_seed(0)
    K, C, B = 4, 10, 5
    stack = torch.randn(B, K, C)
    for name in ("uniform", "beta", "gate"):
        comb = build_combiner(name, K, C)
        ok = True
        for subset in ([0], [0, 2], [1, 2, 3], [0, 1, 2, 3]):
            mask = torch.zeros(K, dtype=torch.bool)
            mask[subset] = True
            _, aux = comb(stack, mask)
            w = aux["weights"]
            ok &= bool(torch.allclose(w.sum(-1), torch.ones(w.shape[:-1]), atol=1e-5))
            ok &= bool((w[..., ~mask].abs() < 1e-8).all())    # inactive get exactly 0
        check(f"weights sum to 1 on every subset ({name})", ok)


def test_weighted_average() -> None:
    """FedAvg maths, and integer buffers must survive unchanged."""
    a = {"w": torch.ones(3), "n": torch.tensor(5, dtype=torch.int64)}
    b = {"w": torch.zeros(3), "n": torch.tensor(9, dtype=torch.int64)}
    out = weighted_average([a, b], [3, 1])                # 3:1 weighting -> 0.75
    check("weighted average value",
          torch.allclose(out["w"], torch.full((3,), 0.75)),
          f"got {out['w'][0]:.4f}")
    check("integer buffers keep dtype and are not averaged",
          out["n"].dtype == torch.int64 and int(out["n"]) == 5,
          f"got {out['n']} ({out['n'].dtype})")


def test_no_lr_dilution() -> None:
    """Cluster-local aggregation applies the full update, not n_c/n of it.

    v1 averaged the classifier over ALL clients, so block c moved by
    (n_c / n_total) * mean_update.  With 2 of 4 clients in cluster c that is a
    silent halving.  Here the update must arrive intact.
    """
    K = 2
    base = torch.zeros(4)
    # cluster 0 has 2 of 4 clients; each applies the same +1.0 update
    v1_style = sum(0.25 * (base + 1.0) for _ in range(2)) + sum(0.25 * base for _ in range(2))
    v2_style = weighted_average([{"w": base + 1.0}, {"w": base + 1.0}], [1, 1])["w"]
    check("v1 aggregation dilutes the update", bool(torch.allclose(v1_style, base + 0.5)),
          f"v1 moved {v1_style[0]:.2f} instead of 1.00")
    check("v2 cluster-local aggregation does not dilute",
          bool(torch.allclose(v2_style, base + 1.0)),
          f"v2 moved {v2_style[0]:.2f}")


def test_dense_rounds_move_the_combiner() -> None:
    """Both dense modes must run and actually change the combiner parameters."""
    set_seed(0)
    data = build_federated_data(
        root="./data_cache", num_clients=4, alpha=100.0, rotation_groups=[0, 1],
        seed=0, subsample=600, download=False,
    )
    labels = np.array([0, 1, 0, 1])
    for mode in ("clients", "server"):
        trainer = EnsembleTrainer(
            data=data, client_clusters=labels, num_clusters=2,
            device=torch.device("cpu"), norm="gn", combiner="gate",
            batch_size=32, combiner_lr=0.5,
        )
        warm = cpu_state(ClientView(norm="gn"))
        init_ensemble_from_warmup(trainer.ensemble, warm)
        before = {k: v.clone() for k, v in trainer.ensemble.combiner.state_dict().items()}
        rec = trainer.dense_round(1, mode=mode, fraction=1.0, epochs=1, max_batches=2)
        after = trainer.ensemble.combiner.state_dict()
        moved = any(not torch.allclose(before[k], after[k]) for k in before)
        check(f"dense round moves the combiner (mode={mode})",
              moved and np.isfinite(rec.mean_local_loss),
              f"loss = {rec.mean_local_loss:.4f}")


def test_uniform_dense_is_a_noop() -> None:
    """A parameterless combiner must skip dense rounds rather than crash."""
    set_seed(0)
    data = build_federated_data(
        root="./data_cache", num_clients=2, alpha=100.0, rotation_groups=[0],
        seed=0, subsample=400, download=False,
    )
    trainer = EnsembleTrainer(
        data=data, client_clusters=np.array([0, 0]), num_clusters=1,
        device=torch.device("cpu"), norm="gn", combiner="uniform", batch_size=32,
    )
    rec = trainer.dense_round(1, mode="clients", fraction=1.0, epochs=1, max_batches=1)
    check("uniform combiner skips dense rounds", bool(rec.extra.get("skipped")))


def main() -> None:
    print("\nEnsemble FL v2 - invariant tests\n" + "-" * 48)
    test_uniform_matches_v1()
    test_single_active_equivalence()
    test_weights_sum_to_one()
    test_weighted_average()
    test_no_lr_dilution()
    test_uniform_dense_is_a_noop()
    test_dense_rounds_move_the_combiner()
    n, total = sum(_results), len(_results)
    print("-" * 48)
    print(f"{n}/{total} passed\n")
    raise SystemExit(0 if n == total else 1)


if __name__ == "__main__":
    main()
