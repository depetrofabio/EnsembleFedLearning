# Hierarchical Ensemble Federated Learning

**Federated learning when clients differ in *two* ways at once: what their
images look like, and which classes they hold.**

Cluster clients by domain, give each cluster its own feature extractor, and
learn how to combine those experts behind a shared classifier. Evaluated on
CIFAR-10 with combined **feature shift** (image rotations) and **label skew**
(Dirichlet partitioning).

```bash
pip install -r requirements.txt
python -m hefl.run --config hefl/configs/rotation_dirichlet.json
```

---

## The finding

Clients can be grouped by their **domain** using spatially-gridded activation
statistics from a **frozen, untrained** network — 320 dimensions, one forward
pass per client, **no local training at all**:

| clustering signal | dimension | ARI vs rotation | ARI vs label |
|---|---|---|---|
| `layer4 + fc` update *(the usual choice)* | 8,398,848 | 0.547 | 0.065 |
| stem weight update | 149,824 | 0.145 | 0.102 |
| **gridded activation statistics** | **320** | **1.000** | 0.025 |

*40 clients, 4 rotation groups, Dirichlet α = 0.5, full CIFAR-10, seed 42.*

Two things make this worth reporting:

- **The spatial grid is the whole trick.** Pooling each channel over the entire
  feature map collapses to ARI 0.09 once labels skew, because it discards
  exactly what a rotation changes. A 2×2 grid keeps the spatial layout — *sky at
  the top* vs *sky on the left* — which is a direct signature of the domain.
- **Measuring the input beats measuring the gradient.** The stem *update*
  targets the same layers the activation probe reads, yet reaches only 0.145:
  how weights respond to data is dominated by dataset size and label
  composition; the input distribution itself is not.

Reproduce it in about a minute, CPU only:
**[`notebooks/02_clustering_signals.ipynb`](notebooks/02_clustering_signals.ipynb)**

Honest caveat: ARI = 1.00 from an *untrained* network means 90° rotations are an
easy shift to detect. This is strong evidence that 8.4M-dimensional gradient
signatures are unnecessary, and weak evidence that the approach transfers to
subtler domain gaps. See [`docs/REPORT.md`](docs/REPORT.md) §10.2.

---

## Method

```
        ┌── backbone_1 ──► head_1 ──► ℓ_1 ┐
  x ────┼── backbone_2 ──► head_2 ──► ℓ_2 ├──► Σ_k w_k·ℓ_k + b
        └── backbone_K ──► head_K ──► ℓ_K ┘         ▲
                                                 combiner
```

A client trains through **one** expert. The combination is trained separately in
*dense rounds*, where the experts are frozen and all K are active — this is the
step that makes `w` a learned object rather than a hardcoded constant.

| combiner | params | weights | note |
|---|---|---|---|
| `uniform` | 0 | `1/\|A\|` | reproduces the classic formulation exactly |
| `beta` | K | `softmax(β)` | one global scalar per expert, interpretable |
| `gate` | ~1k | `softmax(g(conf(x)))` | **input-dependent** — the one that matters under feature shift |
| `mlp` | ~5k | — | genuine cross-expert interaction |

Every weight-producing combiner normalises over the active subset, so a lone
active expert receives weight exactly 1. The function a client optimises locally
is therefore *identical* to what the ensemble computes with only that expert
active — verified to zero error in the test suite. That removes the train/test
mismatch structurally instead of correcting it with a scale factor.

Full derivation and design rationale: **[`docs/REPORT.md`](docs/REPORT.md)**.

---

## Results

*40 clients · full CIFAR-10 · 4 rotation groups · Dirichlet α = 0.5 · ResNet-18 from
scratch · 100 rounds at 25 % participation · seed 42. No augmentation anywhere, so
absolute numbers are low by construction — the relative ordering is the result.*

| method | test accuracy | |
|---|---|---|
| **oracle expert** | **0.3587** | ceiling: per-sample domain known |
| `uniform` combination | 0.3478 | parameter-free |
| `beta` combination | 0.3442 | 4 learned params |
| `mlp` combination | 0.3409 | 7,050 learned params |
| `gate` combination | 0.3139 | routing accuracy **0.273** vs 0.25 chance |
| FedAvg (single model) | *pending* | identical backbone, rounds, participation |

Three things this establishes:

**Clustering recovers the domain structure exactly.** The oracle's best-expert-per-rotation
choice is a **bijection** — `{0°→e1, 90°→e2, 180°→e0, 270°→e3}`. Every expert wins on
exactly one domain and none wins twice. Undifferentiated experts could not produce that.

**The headroom for learned routing is only +1.1 points** (0.3478 → 0.3587). Parameter-free
averaging captures ~90 % of everything optimal expert selection could achieve, which makes
a learned router a poor trade before you even build one.

**The learned gate fails, and the failure is diagnosed.** It routes at chance because it is
fed *confidence summaries* (per-expert entropy, max-probability, margin) rather than the
input image — when experts are not confidently distinguishable, that summary carries no
signal and the gate degenerates. Choosing a cheap routing input was the wrong trade;
a small CNN on `x` would be robust to it, at the cost of one forward pass. Reported rather
than dropped, because the measured +1.1-point ceiling is the more useful finding.

---

## Notebooks

Read in order; the first two need no GPU.

| | | runtime |
|---|---|---|
| [`01_heterogeneity.ipynb`](notebooks/01_heterogeneity.ipynb) | build the benchmark, visualise both shift axes | ~1 min |
| [`02_clustering_signals.ipynb`](notebooks/02_clustering_signals.ipynb) | **reproduces the headline result**, grid ablation, α sweep | ~2 min |
| [`03_run_experiment.ipynb`](notebooks/03_run_experiment.ipynb) | architecture walkthrough + end-to-end pipeline run | ~15 min |
| [`04_results.ipynb`](notebooks/04_results.ipynb) | expert matrix, method comparison, curves, subset analysis | instant |
| [`05_colab_gpu.ipynb`](notebooks/05_colab_gpu.ipynb) | the full experimental matrix on a cloud GPU | hours |

---

## Reproducing everything

```bash
bash scripts/run_sweep.sh tier1     # ~15 runs
python -m hefl.aggregate            # -> mean ± std tables
```

**tier1** is the minimum defensible set: the main setting × 3 seeds, the two
single-axis controls (`rotation_only`, `dirichlet_only`), and the `random` /
`oracle` cluster-assignment ablations.

That last pair is not optional. *K* models beating one model is a **capacity
confound** until random assignment is shown to be worse than learned assignment.

**tier2** sweeps number of clusters, α, clustering signal, dense-round mode, and
logit adjustment.

Runs are independent and the script skips anything already finished, so the
matrix is resumable and splittable across machines. Budget ~1–1.5 h per run on a
T4, ~30 min on an A100.

---

## Layout

```
hefl/                  the package
├── datasets.py        rotation groups, Dirichlet split, per-rotation val/test
├── models.py          backbones, per-expert heads, four combiners
├── clustering.py      four clustering signals, K-means, ARI against both axes
├── federated.py       expert rounds, cluster-local aggregation, dense rounds
├── evaluation.py      per-rotation tables, expert matrix, oracle, routing
├── baselines.py       FedAvg, centralized, random/oracle assignment
├── run.py             one-command pipeline
├── aggregate.py       collects finished runs into paper-ready tables
├── test_invariants.py 14 structural invariants
└── configs/           smoke · fast · rotation_only · dirichlet_only · main

notebooks/             01–05, see above
scripts/run_sweep.sh   the full experimental matrix
docs/REPORT.md         implementation report: every step and its rationale
docs/METHOD.md         technical description of the original formulation
```

---

## Tests

```bash
python -m hefl.test_invariants
```

14 checks on the properties that are easy to break silently: that `uniform`
reproduces the classic forward pass, that single-expert evaluation equals local
client training exactly, that combination weights sum to 1 over every subset,
that aggregation does not dilute updates, and that both dense-round modes move
the combiner.

---

## Status

**Done** — clustering study (3 seeds, grid ablation, four signals compared); full-scale
run at α = 0.5 with all four combiners and the oracle ceiling; implementation with
14/14 invariants passing; harness, configs, aggregation and baselines.

**In progress** — the FedAvg row of the results table.

**Not done** — multiple seeds and the `random` / `oracle` *assignment* ablations
(both in `tier1` of the sweep, compute-bound not code-bound); α = 0.1; comparisons
against FedBN, IFCA, CFL, LG-FedAvg, FedRoD; any dataset beyond CIFAR-10 or any
non-synthetic feature shift.

Until the random-assignment control runs, any ensemble-vs-FedAvg gap is **not
attributable** to clustering — K experts carry K× the parameters, and that confound
has to be excluded before the comparison means anything.

Known limitations are documented rather than omitted —
[`docs/REPORT.md`](docs/REPORT.md) §12, including K× inference cost, the absence
of top-1 routing, and the gate's collapse mode when experts are undertrained.

---

## License

MIT — see [`LICENSE`](LICENSE).
