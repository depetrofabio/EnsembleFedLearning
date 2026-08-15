"""
Collect every finished run into the tables a paper needs.

    python -m hefl.aggregate                       # all runs
    python -m hefl.aggregate --pattern "main_*"    # a subset

Reads ``results.json`` from each run directory and emits, to stdout and to
``hefl/results/_summary/``:

* ``main_table.md``      per-method accuracy, mean +/- std over seeds
* ``clustering.md``      ARI per signal against BOTH heterogeneity axes
* ``ablations.md``       one row per non-seed run
* ``summary.csv``        everything flat, for plotting

Runs are grouped by stripping a trailing ``_seed<N>`` from the directory name,
so ``main_seed42``/``main_seed43``/``main_seed44`` become one row with error
bars. Anything without that suffix is treated as a single-seed ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

RESULTS = Path("hefl/results")
SEED_RE = re.compile(r"_seed\d+$")


def load_runs(pattern: str) -> Dict[str, dict]:
    runs = {}
    for path in sorted(RESULTS.glob(pattern)):
        f = path / "results.json"
        if not f.is_file():
            continue
        try:
            runs[path.name] = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  ! skipping {path.name}: unreadable results.json")
    return runs


def group_key(name: str) -> str:
    return SEED_RE.sub("", name)


def mean_std(values: List[float]) -> str:
    if not values:
        return "-"
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return f"{m:.4f}"
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return f"{m:.4f} ± {var ** 0.5:.4f}"


def method_rows(run: dict) -> Dict[str, dict]:
    """method name -> {rot_k: acc, overall: acc} for one run."""
    out = {}
    for key, val in run.items():
        if key.startswith("ensemble_") and isinstance(val, dict) and "test" in val:
            out[key.replace("ensemble_", "ensemble (") + ")"] = val["test"]
    if "fedavg" in run:
        out["FedAvg (1 model)"] = run["fedavg"]["test"]
    if "centralized" in run:
        out["Centralized (ceiling)"] = run["centralized"]["test"]
    if "oracle" in run:
        out["oracle expert (ceiling)"] = run["oracle"]["test"]
    for k, row in enumerate(run.get("expert_matrix", [])):
        groups = sorted(run["config"]["rotation_groups"])
        out[f"expert {k} only"] = {f"rot_{g}": row[j] for j, g in enumerate(range(len(groups)))}
        out[f"expert {k} only"]["overall"] = sum(row) / len(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default="hefl/results/_summary")
    args = ap.parse_args()

    runs = load_runs(args.pattern)
    if not runs:
        print(f"No finished runs matching '{args.pattern}' under {RESULTS}/")
        return
    print(f"Found {len(runs)} finished run(s)\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- main --
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for name, run in runs.items():
        grouped[group_key(name)].append(run)

    lines = ["# Main results", ""]
    flat_rows = []
    for group, group_runs in sorted(grouped.items()):
        seeds = [r["config"]["seed"] for r in group_runs]
        lines += [f"## {group}  (n={len(group_runs)}, seeds {sorted(seeds)})", ""]
        per_method: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in group_runs:
            for method, accs in method_rows(r).items():
                for col, v in accs.items():
                    if isinstance(v, (int, float)):
                        per_method[method][col].append(float(v))

        cols = sorted({c for m in per_method.values() for c in m if c != "overall"})
        header = ["method"] + cols + ["overall"]
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] * len(header)) + "|"]
        for method in sorted(per_method):
            cells = [method] + [mean_std(per_method[method].get(c, [])) for c in cols]
            cells.append(mean_std(per_method[method].get("overall", [])))
            lines.append("| " + " | ".join(cells) + " |")
            flat_rows.append({
                "group": group, "method": method, "n_seeds": len(group_runs),
                **{c: mean_std(per_method[method].get(c, [])) for c in cols},
                "overall": mean_std(per_method[method].get("overall", [])),
            })
        lines.append("")

    (out / "main_table.md").write_text("\n".join(lines))

    # ---------------------------------------------------------- clustering --
    cl = ["# Clustering signals (ARI against both heterogeneity axes)", "",
          "| run | signal | dim | ARI rotation | ARI label | silhouette |",
          "|---|---|---|---|---|---|"]
    for name, run in sorted(runs.items()):
        for sig, m in run.get("clustering", {}).items():
            cl.append(f"| {name} | {sig} | {m['dim']} | {m['ari_rotation']:.4f} "
                      f"| {m['ari_label']:.4f} | {m['silhouette']:.4f} |")
    (out / "clustering.md").write_text("\n".join(cl))

    # ------------------------------------------------------------- ablations --
    ab = ["# Ablations (single-seed runs)", "",
          "| run | clusters | alpha | signal | assignment | gate overall | uniform overall | FedAvg |",
          "|---|---|---|---|---|---|---|---|"]
    for name, run in sorted(runs.items()):
        c = run["config"]
        g = run.get("ensemble_gate", {}).get("test", {}).get("overall")
        u = run.get("ensemble_uniform", {}).get("test", {}).get("overall")
        f = run.get("fedavg", {}).get("test", {}).get("overall")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-"
        ab.append(f"| {name} | {c['num_clusters']} | {c['alpha']} | {c['cluster_signal']} "
                  f"| {c['assignment']} | {fmt(g)} | {fmt(u)} | {fmt(f)} |")
    (out / "ablations.md").write_text("\n".join(ab))

    # ------------------------------------------------------------------ csv --
    if flat_rows:
        keys = sorted({k for r in flat_rows for k in r})
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(flat_rows)

    print("\n".join(lines[:40]))
    print(f"\nWrote {out}/main_table.md, clustering.md, ablations.md, summary.csv")


if __name__ == "__main__":
    main()
