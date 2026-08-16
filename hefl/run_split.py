"""
Run the split-depth ensemble and compare it against the recorded FedAvg baseline.

    python -m hefl.run_split --split_depth 2

Two shortcuts make this much cheaper than `hefl.run`:

* **Clustering needs no warmup.** `act_stats` reads a frozen, randomly initialised
  backbone (docs/REPORT.md 10.2), so the whole clustering phase is a handful of
  forward passes rather than a training loop.
* **FedAvg is not retrained.** It is a property of the data and the budget, not of
  this architecture, so the number from the reference run is reused. Pass
  `--fedavg_reference` to override it, or `--retrain_fedavg` to measure it again.

Everything else - data, seed, rounds, participation, local epochs, optimiser - is
identical to the reference run, so the comparison is like-for-like.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .clustering import activation_statistics, cluster_clients
from .datasets import build_federated_data
from .evaluation import accuracy_per_rotation, expert_matrix, markdown_table, oracle_from_validation
from .models import CifarBackbone
from .split import SplitTrainer
from .utils import pick_device, set_seed

# Reference run: configs/rotation_dirichlet_mps.json, seed 42, 100 rounds.
FEDAVG_REFERENCE = 0.4895
INDEPENDENT_EXPERTS_REFERENCE = 0.3478      # split_depth = 5, uniform combiner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split_depth", type=int, default=2)
    p.add_argument("--num_clients", type=int, default=40)
    p.add_argument("--num_clusters", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--data_root", type=str, default="./data_cache")
    p.add_argument("--fedavg_reference", type=float, default=FEDAVG_REFERENCE)
    p.add_argument("--output_dir", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device("auto")
    out = Path(args.output_dir or f"./hefl/results/split_depth{args.split_depth}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 72)
    print(f"Split-depth ensemble  |  depth={args.split_depth}  device={device}")
    print("=" * 72)

    # ---------------------------------------------------------------- data --
    data = build_federated_data(
        root=args.data_root, num_clients=args.num_clients, alpha=args.alpha,
        rotation_groups=[0, 1, 2, 3], seed=args.seed, subsample=args.subsample,
        download=False,
    )
    print(f"[1/4] data: {data.num_clients} clients, sizes "
          f"{min(data.client_sizes())}-{max(data.client_sizes())}, alpha={args.alpha}")

    # ---------------------------------------------------- clustering (free) --
    probe = CifarBackbone("gn").to(device).eval()
    feats = np.array([
        activation_statistics(probe, DataLoader(ds, batch_size=64), device, max_batches=8, grid=2)
        for ds in data.client_sets
    ])
    res_cl = cluster_clients(feats, args.num_clusters, seed=args.seed, pca_dim=32)
    labels = res_cl["labels"]
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(data.client_rotation, labels)
    print(f"[2/4] clustering: ARI vs rotation {ari:.4f}, sizes "
          f"{np.bincount(labels, minlength=args.num_clusters).tolist()}")
    del probe

    # ------------------------------------------------------------ training --
    trainer = SplitTrainer(
        data=data, client_clusters=labels, num_clusters=args.num_clusters, device=device,
        split_depth=args.split_depth, combiner="uniform", batch_size=args.batch_size,
        lr=args.lr, seed=args.seed,
    )
    pc = trainer.ensemble.parameter_count()
    print(f"[3/4] training: personal {pc['personal']/1e6:.2f}M + shared "
          f"{pc['shared']/1e6:.2f}M = {pc['total']/1e6:.2f}M params "
          f"(FedAvg is 11.19M; independent experts were 44.70M)")
    trainer.fit(rounds=args.rounds, fraction=args.fraction,
                local_epochs=args.local_epochs, eval_every=args.eval_every)

    # ---------------------------------------------------------- evaluation --
    print("\n[4/4] evaluation ...")
    ens_acc = accuracy_per_rotation(trainer.ensemble, data.test_sets, device)
    mat = expert_matrix(trainer.ensemble, data.test_sets, device)
    oracle, choice = oracle_from_validation(trainer.ensemble, data.val_sets, data.test_sets, device)
    groups = sorted(data.test_sets.keys())

    rows = [{"method": f"FedAvg (reference)", "overall": args.fedavg_reference},
            {"method": f"split-depth {args.split_depth} ensemble", **ens_acc},
            {"method": "oracle expert (ceiling)", **oracle}]
    for k in range(args.num_clusters):
        rows.append({"method": f"expert {k} only",
                     **{f"rot_{g}": mat[k, j] for j, g in enumerate(groups)},
                     "overall": float(mat[k].mean())})
    table = markdown_table(rows, groups, data.rotations)
    print("\n" + table)

    delta_fed = ens_acc["overall"] - args.fedavg_reference
    delta_ind = ens_acc["overall"] - INDEPENDENT_EXPERTS_REFERENCE
    print(f"\n  vs FedAvg              : {delta_fed:+.4f}")
    print(f"  vs independent experts : {delta_ind:+.4f}")

    payload = {
        "config": vars(args), "clustering_ari_rotation": float(ari),
        "cluster_sizes": np.bincount(labels, minlength=args.num_clusters).tolist(),
        "parameters": pc, "ensemble": {"test": ens_acc},
        "oracle": {"test": oracle, "expert_per_rotation": choice},
        "expert_matrix": mat.tolist(),
        "fedavg_reference": args.fedavg_reference,
        "delta_vs_fedavg": delta_fed, "delta_vs_independent_experts": delta_ind,
        "history": trainer.history, "runtime_seconds": time.time() - started,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    (out / "table.md").write_text(table + "\n")

    # Save weights: re-evaluating is cheap, retraining is six hours.
    md = out / "models"; md.mkdir(exist_ok=True)
    for k in range(args.num_clusters):
        torch.save(trainer.ensemble.personals[k].state_dict(), md / f"personal_{k}.pt")
        torch.save(trainer.ensemble.heads[k].state_dict(), md / f"head_{k}.pt")
    torch.save(trainer.ensemble.shared.state_dict(), md / "shared.pt")
    torch.save(trainer.ensemble.bias.detach().cpu(), md / "bias.pt")
    print(f"\nDone in {payload['runtime_seconds']/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
