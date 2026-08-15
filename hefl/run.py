"""
Entry point: one command runs the whole protocol and writes the headline table.

    python -m hefl.run --config hefl/configs/smoke.json

Phases
------
1. Data       - Dirichlet label split x rotation groups; per-rotation val/test.
2. Warmup     - FedAvg warm start, then one local probe per client.
3. Clustering - every signal scored against BOTH ground-truth axes (ARI vs
                rotation, ARI vs label histogram); one signal is used to assign.
4. Experts    - federated rounds; one active expert per client, cluster-local
                aggregation of backbone/head, global aggregation of the bias.
5. Combiners  - the experts are frozen and each combiner is fitted in dense
                rounds (all K active).  This is the step v1 never performed.
6. Evaluation - per-rotation table, per-expert matrix, oracle ceiling, routing
                accuracy, subset analysis, FedAvg baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .baselines import (oracle_cluster_assignment, random_cluster_assignment,
                        train_centralized, train_fedavg)
from .clustering import clustering_report, run_warmup
from .datasets import build_federated_data
from .evaluation import (
    accuracy_per_rotation,
    expert_matrix,
    markdown_table,
    oracle_from_validation,
    routing_accuracy,
    subset_analysis,
)
from .federated import EnsembleTrainer, init_ensemble_from_warmup
from .models import build_combiner
from .utils import pick_device, set_seed

DEFAULTS: Dict = {
    # data
    "data_root": "./data_cache",
    "num_clients": 40,
    "alpha": 0.5,
    "rotation_groups": [0, 1, 2, 3],
    "split_fractions": [0.8, 0.1, 0.1],
    "augment": False,
    "subsample": None,
    # model / optimisation
    "norm": "gn",
    "batch_size": 32,
    "lr": 0.01,
    "num_classes": 10,
    "logit_adjust": False,
    # warmup
    "warmup_fedavg_rounds": 1,
    "warmup_local_epochs": 2,
    "warmup_signals": ["delta_l4fc", "delta_stem", "act_stats"],
    "warmup_act_batches": 4,
    "warmup_act_grid": 2,
    # clustering
    "num_clusters": 4,
    "cluster_signal": "act_stats",
    "pca_dim": 32,
    "assignment": "learned",           # learned | random | oracle
    # expert training
    "rounds": 30,
    "fraction": 0.25,
    "local_epochs": 1,
    "eval_every": 5,
    "dense_every": 0,                  # 0 = fit combiners after training instead
    # combiner fitting
    "combiners": ["uniform", "beta", "gate", "mlp"],
    "combiner_lr": 0.05,
    "dense_mode": "clients",           # clients | server
    "dense_rounds": 5,
    "dense_fraction": 0.25,
    "dense_epochs": 1,
    "dense_max_batches": None,
    "routing_weight": 0.5,
    # baselines
    "run_fedavg": True,
    "run_centralized": False,
    "centralized_epochs": 30,
    # misc
    "seed": 42,
    "device": "auto",
    "output_dir": "./hefl/results/run",
    "save_models": True,
    "subset_analysis": True,
}


def load_config(path: str | None, overrides: Dict) -> Dict:
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as fh:
            cfg.update(json.load(fh))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ensemble FL v2 - routed cluster ensemble")
    p.add_argument("--config", type=str, default=None)
    for key in ("num_clients", "num_clusters", "rounds", "dense_rounds", "seed",
                "local_epochs", "warmup_fedavg_rounds", "warmup_local_epochs", "subsample"):
        p.add_argument(f"--{key}", type=int, default=None)
    for key in ("alpha", "lr", "fraction", "combiner_lr", "routing_weight", "dense_fraction"):
        p.add_argument(f"--{key}", type=float, default=None)
    for key in ("cluster_signal", "assignment", "dense_mode", "norm", "device", "output_dir"):
        p.add_argument(f"--{key}", type=str, default=None)
    p.add_argument("--combiners", type=str, default=None, help="comma-separated")
    p.add_argument("--no_fedavg", action="store_true")
    p.add_argument("--centralized", action="store_true")
    p.add_argument("--logit_adjust", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "no_fedavg", "combiners", "logit_adjust", "centralized")}
    if args.combiners:
        overrides["combiners"] = [c.strip() for c in args.combiners.split(",")]
    if args.no_fedavg:
        overrides["run_fedavg"] = False
    if args.centralized:
        overrides["run_centralized"] = True
    if args.logit_adjust:
        overrides["logit_adjust"] = True

    cfg = load_config(args.config, overrides)
    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=" * 72)
    print(f"Ensemble FL v2  |  device={device}  |  output={out}")
    print("=" * 72)

    # ------------------------------------------------------------------ 1. data
    print("\n[1/6] Building federated data ...")
    data = build_federated_data(
        root=cfg["data_root"],
        num_clients=cfg["num_clients"],
        alpha=cfg["alpha"],
        rotation_groups=cfg["rotation_groups"],
        seed=cfg["seed"],
        split_fractions=tuple(cfg["split_fractions"]),
        augment=cfg["augment"],
        subsample=cfg["subsample"],
    )
    sizes = data.client_sizes()
    print(f"  clients={data.num_clients}  sizes min/med/max={min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
    print(f"  rotation groups={cfg['rotation_groups']}  alpha={cfg['alpha']}")
    print(f"  combiner pool={len(data.combiner_set)}  val/rot={len(next(iter(data.val_sets.values())))}"
          f"  test/rot={len(next(iter(data.test_sets.values())))}")

    # ---------------------------------------------------------------- 2. warmup
    print("\n[2/6] Warmup ...")
    warm = run_warmup(
        data.client_sets, device,
        norm=cfg["norm"], num_classes=cfg["num_classes"],
        fedavg_rounds=cfg["warmup_fedavg_rounds"],
        local_epochs=cfg["warmup_local_epochs"],
        lr=cfg["lr"], batch_size=cfg["batch_size"], seed=cfg["seed"],
        signals=cfg["warmup_signals"], act_batches=cfg["warmup_act_batches"],
        act_grid=cfg["warmup_act_grid"],
    )

    # ------------------------------------------------------------ 3. clustering
    print("\n[3/6] Clustering ...")
    report = clustering_report(
        warm, data.client_rotation, data.client_label_hist,
        num_clusters=cfg["num_clusters"], seed=cfg["seed"], pca_dim=cfg["pca_dim"],
    )
    print(f"  {'signal':<14}{'dim':>10}{'ARI(rotation)':>16}{'ARI(label)':>13}{'silhouette':>13}")
    for name, m in report.items():
        print(f"  {name:<14}{m['dim']:>10}{m['ari_rotation']:>16.4f}{m['ari_label']:>13.4f}{m['silhouette']:>13.4f}")

    if cfg["assignment"] == "learned":
        labels = np.array(report[cfg["cluster_signal"]]["labels"])
        print(f"  using signal '{cfg['cluster_signal']}' for assignment")
    elif cfg["assignment"] == "random":
        labels = random_cluster_assignment(data.num_clients, cfg["num_clusters"], cfg["seed"])
        print("  using RANDOM assignment (capacity-matched control)")
    elif cfg["assignment"] == "oracle":
        labels = oracle_cluster_assignment(data.client_rotation)
        print("  using ORACLE assignment (ground-truth rotation groups)")
    else:
        raise ValueError(f"unknown assignment '{cfg['assignment']}'")
    counts = np.bincount(labels, minlength=cfg["num_clusters"])
    print(f"  cluster sizes: {counts.tolist()}")

    # ------------------------------------------------------- 4. expert training
    print("\n[4/6] Expert training ...")
    trainer = EnsembleTrainer(
        data=data, client_clusters=labels, num_clusters=cfg["num_clusters"],
        device=device, norm=cfg["norm"], combiner="uniform",
        num_classes=cfg["num_classes"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        combiner_lr=cfg["combiner_lr"], logit_adjust=cfg["logit_adjust"],
        routing_weight=cfg["routing_weight"], seed=cfg["seed"],
    )
    init_ensemble_from_warmup(trainer.ensemble, warm.init_state)
    trainer.fit(
        rounds=cfg["rounds"], fraction=cfg["fraction"], local_epochs=cfg["local_epochs"],
        dense_every=cfg["dense_every"], dense_mode=cfg["dense_mode"],
        dense_fraction=cfg["dense_fraction"], dense_epochs=cfg["dense_epochs"],
        dense_max_batches=cfg["dense_max_batches"], eval_every=cfg["eval_every"],
    )

    # ----------------------------------------------------- 5. combiner fitting
    print("\n[5/6] Fitting combiners on frozen experts ...")
    groups = sorted(data.test_sets.keys())
    rows: List[Dict] = []
    results: Dict[str, Dict] = {}
    combiner_states: Dict[str, Dict] = {}

    for name in cfg["combiners"]:
        trainer.ensemble.combiner = build_combiner(
            name, cfg["num_clusters"], cfg["num_classes"]
        ).to(device)
        trainer.ensemble.combiner_name = name

        for d in range(cfg["dense_rounds"]):
            rec = trainer.dense_round(
                round_num=d + 1, mode=cfg["dense_mode"], fraction=cfg["dense_fraction"],
                epochs=cfg["dense_epochs"], max_batches=cfg["dense_max_batches"],
            )
            if not rec.extra.get("skipped"):
                print(f"    {name:<8} dense {d + 1}/{cfg['dense_rounds']}  loss {rec.mean_local_loss:.4f}")

        acc = accuracy_per_rotation(trainer.ensemble, data.test_sets, device)
        route = routing_accuracy(
            trainer.ensemble, data.test_sets, device,
            trainer.rotation_to_cluster(), data.rotations,
        )
        results[f"ensemble_{name}"] = {"test": acc, "routing": route}
        rows.append({"method": f"ensemble ({name})", **acc})
        combiner_states[name] = {k: v.cpu() for k, v in trainer.ensemble.combiner.state_dict().items()}
        line = f"  {name:<8} overall {acc['overall']:.4f}"
        if route:
            line += f"   routing acc {route['overall']:.4f}"
        print(line)

    # --------------------------------------------------------- 6. evaluation
    print("\n[6/6] Baselines and diagnostics ...")
    mat = expert_matrix(trainer.ensemble, data.test_sets, device)
    for k in range(cfg["num_clusters"]):
        rows.append({"method": f"expert {k} only",
                     **{f"rot_{g}": mat[k, j] for j, g in enumerate(groups)},
                     "overall": float(mat[k].mean())})
    results["expert_matrix"] = mat.tolist()
    results["cluster_to_rotation"] = trainer.cluster_to_rotation()

    oracle, choice = oracle_from_validation(trainer.ensemble, data.val_sets, data.test_sets, device)
    rows.append({"method": "oracle expert (ceiling)", **oracle})
    results["oracle"] = {"test": oracle, "expert_per_rotation": choice}
    print(f"  oracle expert overall {oracle['overall']:.4f}  (per-rotation choice {choice})")

    if cfg["run_fedavg"]:
        fed = train_fedavg(
            data, device, rounds=cfg["rounds"], fraction=cfg["fraction"],
            local_epochs=cfg["local_epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
            norm=cfg["norm"], num_classes=cfg["num_classes"],
            logit_adjust=cfg["logit_adjust"], seed=cfg["seed"],
        )
        fed_acc = accuracy_per_rotation(fed, data.test_sets, device)
        rows.insert(0, {"method": "FedAvg (1 model)", **fed_acc})
        results["fedavg"] = {"test": fed_acc}
        print(f"  FedAvg overall {fed_acc['overall']:.4f}")

    if cfg["run_centralized"]:
        cen = train_centralized(
            data, device, epochs=cfg["centralized_epochs"], lr=cfg["lr"],
            batch_size=cfg["batch_size"], norm=cfg["norm"],
            num_classes=cfg["num_classes"], seed=cfg["seed"],
        )
        cen_acc = accuracy_per_rotation(cen, data.test_sets, device)
        rows.insert(0, {"method": "Centralized (ceiling)", **cen_acc})
        results["centralized"] = {"test": cen_acc}
        print(f"  Centralized overall {cen_acc['overall']:.4f}")

    if cfg["subset_analysis"]:
        results["subsets"] = subset_analysis(trainer.ensemble, data.test_sets, device)

    # ----------------------------------------------------------------- output
    table = markdown_table(rows, groups, data.rotations)
    print("\n" + table + "\n")

    results["clustering"] = report
    results["config"] = cfg
    results["client_clusters"] = labels.tolist()
    results["client_rotation"] = data.client_rotation.tolist()
    results["client_sizes"] = sizes
    results["history"] = [vars(r) for r in trainer.history]
    results["runtime_seconds"] = time.time() - started

    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "table.md").write_text(table + "\n")
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))

    if cfg["save_models"]:
        models_dir = out / "models"
        models_dir.mkdir(exist_ok=True)
        for k in range(cfg["num_clusters"]):
            torch.save(trainer.ensemble.backbones[k].state_dict(), models_dir / f"backbone_{k}.pt")
            torch.save(trainer.ensemble.heads[k].state_dict(), models_dir / f"head_{k}.pt")
        torch.save(trainer.ensemble.bias.detach().cpu(), models_dir / "bias.pt")
        for name, state in combiner_states.items():
            torch.save(state, models_dir / f"combiner_{name}.pt")

    print(f"Done in {results['runtime_seconds']:.1f}s -> {out}")


if __name__ == "__main__":
    main()
