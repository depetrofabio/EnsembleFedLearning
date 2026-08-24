# Hierarchical Ensemble Federated Learning

**Federated learning when clients differ in *two* ways at once: what their
images look like, and which classes they hold.**

Cluster clients by domain, give each cluster its own feature extractor, and
learn how to combine those experts behind a shared classifier. Evaluated on
CIFAR-10 with combined **feature shift** (image rotations) and **label skew**
(Dirichlet partitioning).

> **Headline result: personalising the feature extractor loses to plain FedAvg on this
> benchmark — and every component worked.** Clustering recovers client domains perfectly
> (ARI 1.000), experts specialise cleanly, ensembling beats any single expert. Yet FedAvg
> wins by 14 points, and sharing more of the network monotonically closes the gap
> (44.7M params → 0.3478; 11.6M shared-trunk → 0.4036; FedAvg 11.2M → 0.4895).
> Partitioning clients starves each expert of data, and 90° rotations were never a hard
> enough shift to pay for that. [Full tables and analysis ↓](#results)

```bash
pip install -r requirements.txt
python -m hefl.run --config hefl/configs/rotation_dirichlet.json
```

---

## The clustering finding

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

| method | 0° | 90° | 180° | 270° | **overall** |
|---|---|---|---|---|---|
| **FedAvg** (single model) | 0.4927 | 0.5017 | 0.4751 | 0.4886 | **0.4895** |
| oracle expert *(ensemble ceiling)* | 0.4046 | 0.4352 | 0.2112 | 0.3839 | 0.3587 |
| ensemble, `uniform` | 0.3519 | 0.3834 | 0.3172 | 0.3386 | 0.3478 |
| ensemble, `beta` | 0.2696 | 0.4492 | 0.3007 | 0.3571 | 0.3442 |
| ensemble, `mlp` | 0.3543 | 0.4359 | 0.2261 | 0.3474 | 0.3409 |
| ensemble, `gate` | 0.2130 | 0.4445 | 0.2064 | 0.3918 | 0.3139 |
| expert 0 alone | 0.1317 | 0.1022 | **0.2112** | 0.1144 | 0.1399 |
| expert 1 alone | **0.4046** | 0.1400 | 0.2085 | 0.1478 | 0.2252 |
| expert 2 alone | 0.1977 | **0.4352** | 0.1920 | 0.2356 | 0.2651 |
| expert 3 alone | 0.1711 | 0.1927 | 0.1723 | **0.3839** | 0.2300 |

## FedAvg wins, by 14 points — and every component of the method worked

This is the headline, and it is a negative result worth reporting precisely because
nothing broke along the way:

- **Clustering was perfect.** ARI 1.000 against the rotation ground truth, clusters
  `[10,10,10,10]`.
- **The experts specialised.** Bold cells above are each expert's best domain — a clean
  **bijection**, every expert winning exactly one rotation, each ~2× better on its own
  domain than off it. Undifferentiated experts cannot produce that pattern.
- **Ensembling helped.** `uniform` (0.3478) beats the best single expert (0.2651).
- **The method still loses**, because the **oracle ceiling (0.3587) sits below FedAvg
  (0.4895)**. Even with each sample's true domain known, the specialists cannot match one
  model trained on everything. No combiner can rescue this — the ceiling is the ceiling.

The cause is arithmetic, not a bug: partitioning 40 clients into 4 clusters gives each
expert **a quarter of the data** (~10k images for a ResNet-18 trained from scratch), while
FedAvg's single model sees all 40k. **Data-splitting cost exceeded specialisation benefit.**
And 90° rotations are exactly what rotation augmentation does — a single CNN absorbs them
at modest capacity cost, so the gradient-conflict this method is designed to avoid was
never severe enough to pay for quartering the dataset.

### Two secondary findings

**Learned combination lost to parameter-free averaging.** Optimal routing is worth only
**+1.1 points** over `uniform` (0.3478 → 0.3587), and every learned combiner scored below
it. Measuring that ceiling *before* investing in a router is the transferable lesson.

**The gate failed, and the cause is identified.** Routing accuracy **0.273** vs 0.25
chance. It is fed per-expert *confidence summaries* rather than the input image, to keep
it at ~1k parameters; when experts are not confidently distinguishable that summary
carries no signal and the gate degenerates. A small CNN on `x` would fix it — but the
+1.1-point ceiling says it would not be worth the forward pass.

### Follow-up: how much personalisation does this problem actually need?

If data-splitting is the binding constraint, sharing more of the network should help.
`hefl/split.py` implements a **split-depth** architecture: per-cluster early blocks, one
globally-shared trunk aggregated over *all* clients. `split_depth` is a dial with both
baselines as endpoints — 0 is a shared backbone, 5 is fully independent experts.

| architecture | personalised | params | test accuracy | vs FedAvg |
|---|---|---|---|---|
| **FedAvg** (single model) | nothing | 11.19M | **0.4895** | — |
| split-depth 2 | conv1 + layer1 | 11.64M | 0.4036 | −0.086 |
| independent experts | everything | 44.70M | 0.3478 | −0.142 |

**Sharing the trunk recovered +5.6 points while cutting parameters 3.8×** — confirming the
diagnosis: reduce the data-splitting cost and accuracy rises, exactly as predicted.

But the trend keeps going in one direction. Less personalisation is monotonically better,
and extrapolating the dial to its endpoint gives FedAvg — which is precisely what the
measurements show. **On CIFAR-10 with 90° rotations, personalising the feature extractor
costs more than it returns at every depth tested.** That is the finding.

The likely reason is the shift itself: 90° rotations are exactly what rotation
augmentation does, and one CNN absorbs them at modest capacity cost. The architecture is
built to resolve gradient conflict that, in this benchmark, was never severe enough to pay
for splitting the data. Testing on a *natural* domain gap — PACS, Digit-5, Office-Caltech —
is the experiment that would give personalisation a fair chance.

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


