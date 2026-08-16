# Routed Cluster Ensembles for Federated Learning under Feature + Label Shift

**Implementation report for `hefl/`**

This document explains every step of the implementation: what each component
does, why it is built that way, what defect in the original `training/`
implementation it addresses, and how to verify each claim. It is meant to be
read alongside the code — every section names the file and the function.

---

## Table of contents

1. [Why a v2 at all](#1-why-a-v2-at-all)
2. [Defect → fix map](#2-defect--fix-map)
3. [Step 1 — Data pipeline](#3-step-1--data-pipeline)
4. [Step 2 — Model: explicit per-expert logits](#4-step-2--model-explicit-per-expert-logits)
5. [Step 3 — Warmup and clustering signals](#5-step-3--warmup-and-clustering-signals)
6. [Step 4 — Federated expert training](#6-step-4--federated-expert-training)
7. [Step 5 — Dense rounds: training the combination](#7-step-5--dense-rounds-training-the-combination)
8. [Step 6 — Evaluation protocol](#8-step-6--evaluation-protocol)
9. [Baselines and ablations](#9-baselines-and-ablations)
10. [Results](#10-results)
11. [File map and how to run](#11-file-map-and-how-to-run)
12. [Limitations and what is deliberately not implemented](#12-limitations-and-what-is-deliberately-not-implemented)
13. [Mapping this onto a paper](#13-mapping-this-onto-a-paper)

---

## 1. Why a v2 at all

The goal is unchanged from the original repository: **build a classifier that
combines several models, one per client cluster, and validate the clustering by
rotating images.** That goal is sound. The v1 implementation, however, has a
structural property that prevents it from testing that goal:

> In v1, the combination is never trained. It is a constant that no gradient
> ever saw.

Concretely. v1 concatenates the K feature vectors into a `(B, K·512)` tensor and
applies a single `nn.Linear(K·512, 10)`. During client training only block *c*
is non-zero, so the gradient with respect to every other column block is exactly
zero. Writing `W = [W₁ … W_K]`, a client in cluster *c* produces

$$\text{logits} = W_c z_c + b$$

and the gradient touches only `W_c` and `b`. Across the whole run, `W` is never
anything but K independently trained heads plus a shared bias. At test time v1
switches on all K experts and divides by K, i.e.

$$\text{logits}_{\text{test}} = \frac{1}{K}\sum_{k} W_k z_k + b$$

which is uniform logit averaging with weights that were fixed a priori. Since a
linear map over a block-sparse input cannot represent any interaction between
blocks, no amount of further training in this architecture would change that.

Everything in v2 follows from making the combination an explicit, trainable
object, and from fixing the evaluation protocol so the rotation experiment can
actually produce a signal.

**v2 is a strict generalisation of v1.** Running with `--combiners uniform`
reproduces v1's forward pass exactly (proof in [§4.2](#42-equivalence-to-v1)),
so the comparison between "v1 behaviour" and "learned combination" is controlled:
same experts, same data, same schedule, only the combination differs.

> **How this turned out.** Making the combination trainable was necessary to ask
> the question — but the measured answer (§10.3) was that **uniform averaging
> wins**. Optimal routing is worth only +1.1 points over it, and every learned
> combiner scored below it. The contribution is therefore the *measurement*, not
> the mechanism: the original design's simple averaging was the right choice, and
> this rewrite is what makes that statement falsifiable rather than assumed.

---

## 2. Defect → fix map

| # | v1 behaviour | Consequence | v2 fix | Where |
|---|---|---|---|---|
| 1 | Linear head over block-sparse concat | Combination is untrainable; "shared classifier" is K disjoint heads | Explicit per-expert logits + swappable `Combiner` | `models.py` |
| 2 | Train with 1 active expert, test with K and `/K` | Train/test discrepancy; the report's justification describes the opposite scenario | Weights sum to 1 in every regime (masked softmax) | `models.py:_masked_softmax` |
| 3 | Classifier averaged over **all** clients | Block *c*'s effective LR silently scaled by `n_c/n` | Backbone/head averaged **within cluster**; bias globally | `federated.py:expert_round` |
| 4 | Clustering on Δθ(layer4+fc) | Label-dominated; conflates the two heterogeneity axes | Four signals scored against **both** ground truths | `clustering.py` |
| 5 | `use_fedavg_warmup: true` → `self.warmup_round(...)`, undefined | Default config crashes | FedAvg warmup implemented | `clustering.py:run_warmup` |
| 6 | BN running stats averaged across clients | The failure mode FedBN exists to prevent | GroupNorm default (`norm: "gn"`), BN still available | `models.py:_replace_bn_with_gn` |
| 7 | `_aggregate` casts int buffers to float | Silent dtype corruption of `num_batches_tracked` | Non-float buffers copied, not averaged | `utils.py:weighted_average` |
| 8 | Test data pooled into one undifferentiated set | Per-rotation accuracy, expert matrix and oracle are all uncomputable | Per-rotation val/test sets, equal group sizes | `datasets.py:build_federated_data` |
| 9 | No random-assignment control | Capacity confound: K models vs 1 | `assignment: random / oracle / learned` | `baselines.py` |
| 10 | Feature clustering applies transform twice | Corrupted feature-clustering path | Transform applied once, inside the dataset | `datasets.py` |

---

## 3. Step 1 — Data pipeline

**File:** `datasets.py`

### 3.1 Two independent heterogeneity axes

The benchmark injects the two axes separately so their effects can be
attributed:

- **Feature axis.** Each client is assigned to one rotation group. Every image
  that client holds is rotated by that group's multiple of 90°. Rotation is
  applied with `np.rot90` on the raw uint8 array, so it is exact and lossless —
  no interpolation artefacts that a model could latch onto instead of the
  rotation itself. Assignment is balanced round-robin, then permuted
  (`assign_rotation_groups`).
- **Label axis.** `dirichlet_partition` draws `p ~ Dir(α·1_N)` per class and
  splits that class's indices accordingly. Small α ⇒ strong skew. The partition
  resamples until every client holds at least `min_size` samples, so local
  training and size-weighted aggregation are always well defined.

The two assignments are independent, so `rotation_groups=[0]` gives
Dirichlet-only, and `alpha=100` gives rotation-only. Both are needed as controls.

### 3.2 Three disjoint pools

`_three_way_split` cuts the CIFAR-10 *train* file class-stratified into:

| pool | default | used for |
|---|---|---|
| `train` | 80% | handed to clients (non-IID) |
| `combiner` | 10% | server-side combiner fitting **only** |
| `val` | 10% | model selection, oracle-expert choice **only** |

`combiner` and `val` are disjoint on purpose. Fitting the combination layer on
the same data used to pick the best configuration would leak; keeping them
separate means the oracle row is an honest upper bound rather than a number read
off the data it was selected on. The CIFAR-10 *test* file is never touched until
final evaluation.

### 3.3 The evaluation sets must be rotated *and* separable by group

Two distinct issues here; v1 gets the first one right in the rotation notebook
and the second one wrong everywhere.

**Rotating the test data.** `experiments/ensemble_rotation_dirichlet.ipynb` does
rotate its evaluation data — it builds a rotated test subset per client and
concatenates them, so the test mixture matches the client population. That is
correct, and the notebook says so explicitly. (`ensemble_dirichlet_only.ipynb`
uses the plain un-rotated CIFAR-10 test set, which is fine there because that
experiment has no rotations.) So the mixture itself is not the defect.

**Keeping the groups separable is.** The concatenated test set is a single
undifferentiated pool, and `EnsembleFedAvg.evaluate_ensemble` takes one
`test_set` with no notion of rotation groups. From that you can only ever read a
single scalar. You cannot ask the question the experiment exists to answer —
*does expert k win on rotation k?* — because nothing downstream knows which
samples carry which rotation. A rotation-specialised ensemble evaluated as one
pooled number reports the average of a matrix whose diagonal is the entire
result.

v2 materialises one full copy of the test set **per rotation group**
(`test_sets: Dict[g, Dataset]`), and likewise for validation. "Overall" is then a
uniform mixture over groups — the same mixture the client population has — but
the per-group numbers survive, which is what makes the expert matrix (§8.1), the
oracle ceiling (§8.2) and routing accuracy (§8.3) computable at all. Group sizes
are also equal by construction, rather than depending on how the client split
happened to fall.

The server-side `combiner_set` draws its rotations from the *empirical* client
group frequencies, so the combiner is fitted on the deployment distribution. All
evaluation datasets carry the rotation label (`return_rotation=True`) so routing
accuracy can be measured directly.

### 3.4 The input pipeline is not an afterthought

A federated run recreates a `DataLoader` per client per round, so anything done
per sample is done many times over. The obvious implementation — rotate with
`np.rot90` inside `__getitem__`, then `Image.fromarray` → `ToTensor` →
`Normalize` — makes the input pipeline the bottleneck and leaves the GPU idle.
Two changes remove it:

- **Rotate once, in bulk.** Every sample in a client set shares one rotation, so
  the rotation is applied to the whole array at construction (vectorised per
  distinct value) instead of per fetch.
- **Skip PIL.** With augmentation off — the default, matching the original
  protocol — the data is kept as a uint8 NCHW tensor and normalised on access
  with pure tensor ops.

Measured: **~87,000 samples/s** through a `DataLoader` on this machine, and the
bulk rotation is bit-exact against the per-sample version (`max |diff| = 0.00`).
Augmentation (`augment: true`) keeps the PIL path as an opt-in slow route.

---

## 4. Step 2 — Model: explicit per-expert logits

**File:** `models.py`

### 4.1 Architecture

```
x ──┬── backbone_1 ──► z_1 ──► head_1 ──► ℓ_1 ┐
    ├── backbone_2 ──► z_2 ──► head_2 ──► ℓ_2 ├──► Combiner ──► Σ_k w_k ℓ_k + b
    ├──      ⋮                              ⋮  │        ▲
    └── backbone_K ──► z_K ──► head_K ──► ℓ_K ┘        │
                                              active mask A
```

- `CifarBackbone` — ResNet-18 with the standard CIFAR stem (3×3 stride-1 conv,
  no maxpool; otherwise a 32×32 image collapses to 1×1 far too early), `fc`
  removed, returning 512-d features. `norm="gn"` replaces every `BatchNorm2d`
  with `GroupNorm`.
- `heads` — K × `Linear(512, 10, bias=False)`.
- `bias` — one shared `(10,)` parameter, **outside** the per-cluster heads. This
  is deliberate: the bias absorbs the label prior, and keeping it global stops
  each cluster from re-learning (and re-skewing) its own prior.
- `combiner` — see §4.3.

### 4.2 Equivalence to v1

Write v1's classifier weight as `W = [W₁ | … | W_K]` with `W_k ∈ ℝ^{10×512}`.

*Inference.* v1 computes

$$W\left(\tfrac{1}{|A|}\,\mathrm{concat}_k(z_k \text{ if } k \in A \text{ else } 0)\right) + b = \frac{1}{|A|}\sum_{k \in A} W_k z_k + b$$

v2 with `UniformCombiner` computes `Σ_{k∈A} w_k ℓ_k + b` with `w_k = 1/|A|` and
`ℓ_k = W_k z_k`. **Identical.**

*Client training.* v1's `ClientEnsembleWrapper` has the division commented out
(`# full_features = full_features / 1.0`), giving `W_c z_c + b`. v2's
`ClientView` computes `head_c(backbone_c(x)) + bias`. **Identical.**

So v2 with `uniform` is a faithful re-implementation of v1, and any difference in
the results table comes from the combiner, not from an incidental change.

### 4.3 The combiners

All four consume a `(B, K, C)` stack of per-expert logits plus a `(K,)` boolean
active mask.

| combiner | parameters | weights | notes |
|---|---|---|---|
| `UniformCombiner` | 0 | `1/|A|` | the v1 baseline row |
| `BetaCombiner` | K | `softmax_A(β)` | one global scalar per expert; interpretable — β tracks cluster quality/size |
| `GateCombiner` | ~1k | `softmax_A(g(conf(x)))` | **input-dependent**; the one that matters under feature shift |
| `MLPCombiner` | ~5k | — | MLP over concatenated logits; genuine cross-expert interaction |

**Gate input.** `confidence_features` reduces each expert's logits to three
scalars — negative entropy, max probability, top-1/top-2 margin — giving a
`(B, 3K)` input. Feeding the gate a 3K-dim summary instead of the K·512 features
keeps it tiny (it trains on very little data), keeps it fast, and makes it
readable: you can inspect which expert it trusts and why. It is also
architecture-agnostic — nothing about it depends on ResNet or on 512 dimensions.

**Free routing supervision.** A client knows its own cluster id. So the gate's
routing head can be trained with a plain cross-entropy term against that id at
zero annotation cost (`routing_weight`, default 0.5). This is the mechanism that
turns clustering from an offline preprocessing step into a component that has a
job at inference time.

### 4.4 The consistency property (why "Active Normalization" is unnecessary)

Every weight-producing combiner normalises over the active set:

$$\sum_{k \in A} w_k = 1 \quad \text{for every } A \neq \emptyset$$

`_masked_softmax` fills inactive entries with `-inf` before the softmax, so this
holds exactly, including for `|A| = 1`, where the single active expert receives
weight exactly 1. Therefore:

$$\text{ensemble}(x, A=\{c\}) = \ell_c + b = \text{ClientView}_c(x)$$

**The function a client optimises locally is exactly the function the ensemble
evaluates when only that expert is active.** The v1 train/test discrepancy is
removed by construction rather than patched by a `/num_active` constant, and
every active subset is magnitude-correct for free — which is what makes the
subset analysis in §8.4 meaningful.

`MLPCombiner` is the one exception: it is not a weighted average, so it has no
single-active equivalence and depends entirely on dense rounds. This is a real
trade-off, stated here rather than hidden.

---

## 5. Step 3 — Warmup and clustering signals

**File:** `clustering.py`

### 5.1 Warmup

`run_warmup` has two phases:

- **Phase A** — plain FedAvg for `fedavg_rounds`, producing a shared warm start.
  (v1 called `self.warmup_round(...)`, which is defined nowhere in the
  repository, so `use_fedavg_warmup: true` — the shipped default — raised
  `AttributeError`. It is implemented here.)
- **Phase B** — every client starts from the **same** warm state and trains
  locally for `local_epochs`. Signals are differences from that common origin,
  which is what makes them comparable across clients. Phase B does not advance
  the shared state.

### 5.2 Four signals, scored against both axes

| signal | what it is | dimension (ResNet-18) | expected axis |
|---|---|---|---|
| `delta_l4fc` | Δθ of `layer4` + head | ~8.4M | label (v1's choice) |
| `delta_stem` | Δθ of `conv1`, first norm, `layer1` | ~150k | feature |
| `act_stats` | gridded activation statistics, **no training** | ~320 | feature |
| `delta_full` | every float parameter | ~11M | mixed |

`clustering_report` scores each signal against **both** ground truths:
`ari_rotation` (feature axis) and `ari_label` (label axis, defined as K-means on
the true per-client label histograms). A signal that scores high on one and near
zero on the other is doing what you want. A signal that scores middling on both
is conflating them — which is the diagnosis for v1's `layer4 + fc`.

### 5.3 The preprocessing is not cosmetic

`cluster_clients` centres, L2-normalises, then optionally PCA-reduces before
K-means:

- **Centring** removes the component every client shares (the direction they all
  move in), leaving what makes them different.
- **L2 normalisation** turns Euclidean K-means into spherical K-means, so
  distance reflects *direction*, not update magnitude. Without it, clients
  cluster by dataset size.
- **PCA** matters more than it looks: a raw `layer4` delta is ~8.4M-dimensional
  with only N ≈ 40 points. Euclidean distances concentrate in that regime and
  K-means becomes unstable across seeds.

### 5.4 A measured result: the spatial grid

`activation_statistics` computes, on a frozen backbone, per-channel means pooled
to a coarse `grid × grid` map plus per-channel spatial std. The `grid` argument
turns out to be decisive.

Global pooling (`grid=1`) averages each channel over the whole feature map — and
that discards precisely the information a rotation changes. A rotated image has
nearly the same global channel means, so the descriptor is rotation-blind and
what survives is class content, i.e. the label axis. A coarse grid keeps the
spatial layout ("sky at the top" vs "sky on the left"), which is a direct
signature of the domain.

Measured with `diagnose_signals.py`: 24 clients, 4 rotation groups, **untrained**
backbone, 8 batches per client, mean ARI vs the rotation ground truth over seeds
{0, 1, 2}:

| α | `grid=1` | `grid=2` |
|---|---|---|
| 100 (IID labels) | 0.905 | **1.000** |
| 0.5 | 0.093 | **0.884** |
| 0.1 | 0.020 | **0.382** |

Two readings, one positive and one negative:

1. **The grid is what makes the feature signal robust to label skew.** With IID
   labels even global pooling works. As soon as labels skew, `grid=1` collapses
   to noise while `grid=2` holds. ARI against the *label* axis stays near zero
   for `grid=2` throughout, so it recovers the feature axis and only that.
2. **The signal degrades at α = 0.1.** A client holding almost one class has
   different stem statistics for reasons that have nothing to do with its
   rotation, so at extreme label skew the two axes stop being separable from
   activations alone. This is a genuine limitation of the signal, not a tuning
   problem, and it should be reported as such.

The practical consequence is worth stating plainly: **this signal needs no local
training at all** and works at random initialisation. Clustering can be done in
one cheap forward pass per client, before any federated training begins, at ~320
dimensions instead of 8.4 million.

Reproduce with:

```bash
python -m hefl.diagnose_signals --num_clients 24 --alpha 0.5 --grid 2
```

---

## 6. Step 4 — Federated expert training

**File:** `federated.py`

### 6.1 The round

`EnsembleTrainer.expert_round`:

1. Sample `fraction · N` clients.
2. Each client loads its cluster's backbone + head + the shared bias into a
   single reusable scratch `ClientView` (no per-client deep copy — one
   allocation per round, not one per client).
3. Local SGD for `local_epochs`, optionally with logit adjustment (§6.3).
4. Return the updated state.

### 6.2 The aggregation fix

v1 averaged the classifier over **all** participants. Consider block `c` and let
client *i* in cluster *c* produce update `ΔW_c^{(i)}`; clients outside cluster *c*
return `W_c` unchanged. v1's aggregation gives

$$W_c \leftarrow \sum_i \frac{n_i}{n}\left(W_c + \mathbb{1}[i \in c]\,\Delta W_c^{(i)}\right) = W_c + \frac{n_c}{n}\,\overline{\Delta W_c}$$

so block *c* moves at `n_c/n ≈ 1/K` of the intended step. Nobody chose that
divisor, and it is coupled to K — which means v1's "vary the number of clusters"
experiment confounds capacity, cluster quality, and the classifier learning rate.

v2 aggregates each expert **within its own cluster**:

$$W_c \leftarrow \sum_{i \in c} \frac{n_i}{n_c}\left(W_c + \Delta W_c^{(i)}\right) = W_c + \overline{\Delta W_c}$$

and the shared bias over every participant. Empty clusters are skipped rather
than crashing.

`weighted_average` (in `utils.py`) also fixes the buffer handling: non-float
buffers such as `num_batches_tracked` are copied from the first client rather
than averaged, since averaging silently promotes them to float and they carry no
useful information.

### 6.3 Logit adjustment (optional, `logit_adjust: true`)

Each client knows its own label histogram, so adding `log p_client(y)` to the
logits during local training corrects for label skew directly. It is three lines
(`utils.local_train`) and is typically the single highest-return change in the
Dirichlet-only setting. It is off by default so the main comparison stays clean,
but it should be run as an ablation — if it recovers most of the gap you are
attributing to clustering, that is important to know before writing anything up.

### 6.4 Initialisation

`init_ensemble_from_warmup` seeds every expert from the same warm state. All K
experts therefore start as identical functions, and because combiner weights sum
to one, **the ensemble's output at round 0 is exactly the warmup model's output
for any active subset.** The ensemble starts from a known reference point rather
than a random one, which makes the round-0 row of any learning curve meaningful.

---

## 7. Step 5 — Dense rounds: training the combination

**File:** `federated.py:dense_round`

This is the step v1 never performs, and the reason the combination in v1 is a
constant.

A normal round keeps one expert active, so it can never produce gradient for the
combination. A **dense round** freezes the experts and trains only the combiner
with all K active. Two modes:

**`dense_mode: "clients"` (federated, default).** Sample a subset of clients;
each receives all K frozen backbones, runs the combiner over its own data with
all K active, and uploads only the combiner. The combiner is a few thousand
parameters, so upload cost is negligible; download is K× but only on dense
rounds. The client's own cluster id supplies the routing target for free. This
keeps the method fully federated, which is what protects the privacy claim.

**`dense_mode: "server"` (stacking).** Fit the combiner on the held-out
`combiner` split. Faster and simpler, and it is standard practice for ensembles —
but it is a server-side step and should be described as such in any write-up, not
folded silently into a "fully federated" claim.

Two implementation details that matter:

- Experts are evaluated under `torch.no_grad()` and the ensemble is put in
  `eval()` mode during dense rounds, so frozen normalisation layers do not drift.
- Each client starts its dense step from the *server's* combiner copy, and the
  results are FedAvg'd — otherwise you get sequential drift rather than federated
  averaging.

By default (`dense_every: 0`) combiners are fitted **after** expert training
rather than interleaved. This means all four combiners are compared on *identical
experts*, which is the controlled comparison; interleaving is available via
`dense_every > 0` if you want the combiner to co-adapt during training.

---

## 8. Step 6 — Evaluation protocol

**File:** `evaluation.py`

### 8.1 The headline table

`markdown_table` produces:

| method | 0° | 90° | 180° | 270° | overall |
|---|---|---|---|---|---|
| FedAvg (1 model) | | | | | |
| ensemble (uniform) ← v1 | | | | | |
| ensemble (beta) | | | | | |
| ensemble (gate) | | | | | |
| ensemble (mlp) | | | | | |
| expert k only, for each k | | | | | |
| oracle expert (ceiling) | | | | | |

Three numbers carry the argument:

1. **The diagonal of the per-expert block** (`expert_matrix`). Does expert *k*
   win on rotation *k*? This validates clustering **and** specialisation in one
   measurement — far more informative than an offline ARI, because it shows the
   clustering produced experts that are actually different.
2. **Gate vs oracle.** What routing costs you relative to knowing the domain.
3. **Gate vs FedAvg.** Whether the method works at all.

### 8.2 The oracle is chosen honestly

`oracle_from_validation` picks the best expert per rotation on the **validation**
sets and then applies that choice to test. It never reads the test set to make
the selection, so it is a genuine ceiling rather than a post-hoc maximum.

### 8.3 Routing accuracy

`routing_accuracy` measures how often the gate's argmax expert matches the
sample's true domain, using the cluster→rotation majority map derived from the
clustering result. It is a diagnostic only — never a training signal in the
federated path. Reporting it turns "clustering worked" from an abstract index
into a statement about deployment behaviour.

### 8.4 Subset analysis

`subset_analysis` evaluates every active subset of experts. This is v1's
"cluster combination analysis", but the renormalisation now lives in the combiner
(masked softmax) instead of a `/num_active` constant, so every subset is
magnitude-correct by construction. It answers "can I deploy 2 experts instead of
4, and which 2?" with numbers that are comparable to each other.

---

## 9. Baselines and ablations

**File:** `baselines.py`

- **`train_fedavg`** — one global model, no clustering. Identical backbone,
  optimiser, round count, participation fraction and local epochs, so the only
  difference is the method.
- **`random_cluster_assignment`** — the same K experts, the same total capacity,
  clients assigned at random. **This is the ablation that makes the result
  falsifiable.** K ResNet-18s beating one ResNet-18 says nothing until random
  assignment is shown to be worse than learned assignment. Run with
  `--assignment random`.
- **`oracle_cluster_assignment`** — ground-truth rotation groups; the clustering
  upper bound. Run with `--assignment oracle`.

The three assignment modes share one code path, so nothing but the assignment
differs between them:

```bash
python -m hefl.run --config hefl/configs/rotation_dirichlet.json --assignment learned
python -m hefl.run --config hefl/configs/rotation_dirichlet.json --assignment random
python -m hefl.run --config hefl/configs/rotation_dirichlet.json --assignment oracle
```

---

## 10. Results

### 10.0 Invariant tests (all passing)

`python -m hefl.test_invariants` — seconds, CPU. These verify the
properties this report makes claims about, rather than trusting an accuracy
number to reveal a broken one:

```
PASS  uniform combiner == v1 forward pass            max |diff| = 3.73e-07
PASS  single-active equivalence (uniform)            max |diff| = 0.00e+00
PASS  single-active equivalence (beta)               max |diff| = 0.00e+00
PASS  single-active equivalence (gate)               max |diff| = 0.00e+00
PASS  weights sum to 1 on every subset (uniform / beta / gate)
PASS  weighted average value                         got 0.7500
PASS  integer buffers keep dtype and are not averaged
PASS  v1 aggregation dilutes the update              v1 moved 0.50 instead of 1.00
PASS  v2 cluster-local aggregation does not dilute   v2 moved 1.00
PASS  uniform combiner skips dense rounds
PASS  dense round moves the combiner (mode=clients)
PASS  dense round moves the combiner (mode=server)
14/14 passed
```

Two of these are the load-bearing claims of the whole rewrite. The **equivalence
to v1** (§4.2) makes the comparison controlled — the uniform row really is v1.
The **single-active equivalence** (§4.4) holds to exactly zero error, which is
what lets the "Active Normalization" machinery be deleted rather than fixed.
The dilution test reproduces v1's aggregation bug numerically: the same update
that should move a block by 1.00 moves it by 0.50 under v1's global averaging
with 2 of 4 clients in the cluster.

### 10.1 The grid ablation (3 seeds)

Reported in [§5.4](#54-a-measured-result-the-spatial-grid). Headline: gridded
activation statistics recover the rotation groups at ARI 1.00 (IID labels) and
0.88 (Dirichlet 0.5) from a **frozen, untrained** backbone at ~320 dimensions,
while remaining uncorrelated with the label axis; the signal degrades to ~0.38 at
α = 0.1. Global pooling fails as soon as labels skew.

### 10.2 Signal comparison, in the pipeline

Both runs use 4 rotation groups, Dirichlet α = 0.5, seed 42, and a 1-round
FedAvg warmup + 2 local epochs per client. They differ only in scale.

**Small scale** (24 clients, CIFAR-10 subsampled to 20k, ~530 images/client):

| signal | dimension | ARI vs rotation | ARI vs label | silhouette |
|---|---|---|---|---|
| `delta_l4fc` (**v1's choice**) | 8,398,848 | 0.069 | 0.066 | 0.085 |
| `delta_stem` | 149,824 | −0.022 | 0.124 | 0.197 |
| **`act_stats`** | **320** | **1.000** | 0.032 | 0.665 |

**Full scale** (40 clients, full CIFAR-10, ~970 images/client):

| signal | dimension | ARI vs rotation | ARI vs label | silhouette |
|---|---|---|---|---|
| `delta_l4fc` (**v1's choice**) | 8,398,848 | 0.547 | 0.065 | 0.062 |
| `delta_stem` | 149,824 | 0.145 | 0.102 | 0.073 |
| **`act_stats`** | **320** | **1.000** | 0.025 | 0.568 |

Three things to take from the pair.

- **v1's signal is data-hungry, not broken.** At small scale it recovers nothing
  (0.07); given four times the data per client it recovers roughly half the
  structure (0.55). So the honest characterisation is not "`layer4 + fc` measures
  the wrong thing" — it is that estimating an 8.4M-dimensional direction from
  tens of points needs far more local data than a federated warmup provides, and
  it never gets past partial recovery here.
- **`act_stats` is exact at both scales**, at 320 dimensions — around 26,000×
  smaller — with perfectly balanced clusters (`[6,6,6,6]` and `[10,10,10,10]`)
  matching the true group structure, and ARI ≈ 0.02–0.03 against the label axis.
  It measures the domain and nothing else, and it is insensitive to the scale
  that determines whether the gradient signal works at all.
- **The activation probe beats the gradient probe** even at equal locality.
  `delta_stem` targets the same early layers `act_stats` reads, yet reaches only
  0.15: measuring *how weights respond* to the data is dominated by dataset size
  and label composition, while measuring the *input distribution directly* is
  not. Probing beats differentiating here.

One caveat, stated plainly because it bounds how much this result is worth:
ARI = 1.00 from an **untrained** network means the discrimination task is easy.
90° rotations change early-layer spatial statistics dramatically, so a random
filter bank separates them. This strengthens the argument that v1's machinery
was unnecessarily elaborate, but it is weak evidence that the signal generalises
to subtler domain gaps (sensor, colour, style). Testing it on a natural
feature-shift benchmark is the obvious next step, and a reviewer will ask.

The practical consequence: clustering needs no warmup training at all. One
forward pass per client on a randomly initialised backbone is enough, which
removes an entire phase from the pipeline. `configs/fast.json` exercises this —
`warmup_fedavg_rounds: 0` with `warmup_signals: ["act_stats"]` means no local
training happens during warmup at all, and the phase drops from ~20 minutes to
seconds while still producing ARI 1.00 and perfectly balanced clusters. If you
only need the feature axis, the warmup phase can be deleted outright.

### 10.3 Full-scale results

*40 clients · full CIFAR-10 · 4 rotation groups · Dirichlet α = 0.5 · ResNet-18
from scratch · 100 rounds at 25 % participation · batch 64 · lr 0.01 · GroupNorm ·
no augmentation · seed 42.*

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

Training was healthy throughout — validation accuracy 0.256 → 0.357 over the 100
rounds, mean local loss 2.30 → 0.37 — so nothing below is an artefact of
undertrained models.

**FedAvg wins by 14.2 points, and the ensemble's own ceiling cannot reach it.**
The oracle (0.3587) is 13 points *below* the single-model baseline. That is the
decisive number: even granting perfect per-sample domain knowledge, the
specialists lose. No combination strategy closes this gap, because the gap is not
a combination problem.

The cause is arithmetic. Partitioning 40 clients into 4 clusters gives each expert
**a quarter of the data** — roughly 10k images for a ResNet-18 trained from
scratch — while FedAvg's model sees all 40k. **Data-splitting cost exceeded
specialisation benefit.** A secondary factor: 90° rotations are precisely what
rotation augmentation does, and a single CNN absorbs them at modest capacity cost,
so the gradient conflict this architecture exists to avoid was never severe enough
to justify quartering the dataset.

**Clustering recovered the domain structure exactly.** `act_stats` gave ARI 1.000
against the rotation ground truth with perfectly balanced clusters `[10,10,10,10]`.

**The experts specialised.** The oracle's best-expert-per-rotation selection,
made on validation and then applied to test, is a **bijection**:

```
0° → expert 1     90° → expert 2     180° → expert 0     270° → expert 3
```

Every expert wins on exactly one domain and none wins twice, and the bold cells in
the table above confirm it row-wise: each expert is roughly **2× better on its own
domain than off it** (e.g. expert 1: 0.4046 on 0° against a 0.165 mean elsewhere).
Undifferentiated experts cannot produce a permutation matrix. This is direct
evidence that cluster-specific training did exactly what it was designed to do —
which is what makes the overall loss to FedAvg a *scaling* result rather than a
failure of the mechanism.

**Ensembling also worked.** `uniform` (0.3478) beats the best individual expert
(0.2651) by 8 points, so combining specialists is clearly better than picking one.
It simply is not better than not partitioning the data at all.

**The headroom for learned routing is only +1.1 points.** The gap between
parameter-free averaging (0.3478) and a perfect oracle router (0.3587) bounds
everything any combiner could win. Uniform averaging already captures ~90 % of it.
That is a useful negative design result: it says a learned router is a poor
trade *before* you build one, and it retroactively justifies the original
formulation's simple averaging over the machinery in §4.3.

**The learned gate failed, and the cause is identified.** It routes at 0.273
against a 0.25 chance level — essentially not routing — and is consequently the
*worst* combiner rather than the best. Its dense-round loss oscillated
(1.00 → 1.90 → 1.20) instead of descending. The cause is the design choice in
§4.3: the gate is fed *confidence summaries* (per-expert entropy, max-probability,
margin) rather than the input image, to keep it at ~1k parameters. When experts
are not confidently distinguishable on a sample, that summary carries no signal
and the gate degenerates. A small CNN routing on `x` would be robust to this, at
the cost of one extra forward pass. Given the measured +1.1-point ceiling, the
better conclusion is that the routing problem was not worth solving here.

### 10.4 Why subsampled configs cannot substitute

The `fast` config runs all six phases to completion on a laptop, and every code
path executes. **It does not produce a usable accuracy table, and it is worth
being explicit about why.** Every row lands at chance (~0.10), and the gate's
routing accuracy is 0.25, exactly chance for 4 clusters:

| method | 0deg | 90deg | 180deg | 270deg | overall |
|---|---|---|---|---|---|
| FedAvg (1 model) | 0.1338 | 0.1100 | 0.1094 | 0.1138 | 0.1167 |
| ensemble (uniform) | 0.1100 | 0.1100 | 0.1100 | 0.1100 | 0.1100 |
| ensemble (beta) | 0.1119 | 0.1094 | 0.1100 | 0.1119 | 0.1108 |
| ensemble (gate) | 0.1037 | 0.1037 | 0.1037 | 0.1037 | 0.1037 |
| ensemble (mlp) | 0.1350 | 0.1087 | 0.1200 | 0.1000 | 0.1159 |
| expert 0 only | 0.0844 | 0.0994 | 0.1169 | 0.0912 | 0.0980 |
| expert 1 only | 0.1037 | 0.1037 | 0.1037 | 0.1037 | 0.1037 |
| expert 2 only | 0.0963 | 0.0806 | 0.0944 | 0.1156 | 0.0967 |
| expert 3 only | 0.1119 | 0.1025 | 0.1181 | 0.1025 | 0.1087 |
| oracle expert (ceiling) | 0.1119 | 0.1037 | 0.1169 | 0.1156 | 0.1120 |

Two tells that this is untrained-model noise rather than a result. The
per-expert block has **no diagonal structure at all** — expert *k* does not win
on rotation *k*, and the oracle ceiling (0.1120) sits *below* FedAvg (0.1167),
which is impossible for trained experts. And several rows are constant across
rotations to four decimals (expert 1, and the gate that collapses onto it),
the signature of a model predicting one class for every input.

This is gradient-step starvation, not a defect. Count the steps the `fast`
config actually delivers to a single expert:

| quantity | value |
|---|---|
| train images after subsampling | 8,000 → 6,400 train / 12 clients ≈ **533 per client** |
| batches per client-epoch (batch 64) | ≈ 9 |
| clients per round (`fraction` 0.5) | 6 — but spread over 4 clusters |
| clients per **cluster** per round | ≈ 1.5 → ≈ **13 steps/round per expert** |
| rounds | 15 |
| **total gradient steps per expert** | **≈ 200** |

A ResNet-18 (11M parameters) trained from random initialisation on CIFAR-10
needs on the order of 10⁴ steps to move meaningfully off chance — and FedAvg
averaging partially undoes progress at every round boundary. Two hundred steps
cannot produce a signal, so at this scale the *ordering* of the rows is noise
too, and should not be read as evidence for or against any combiner.

The same arithmetic is what sets the full config's hyperparameters.
`rotation_dirichlet.json` delivers ≈ 8,000 steps per expert:

```
40,000 train / 40 clients          = 1,000 images per client
1,000 / 64                         ≈ 16 batches, x2 local epochs = 32 steps
fraction 0.25 -> 10 clients/round  ≈ 2.5 per cluster -> 80 steps/round/expert
x 100 rounds                       ≈ 8,000 steps per expert
```

**So: the headline accuracy table requires the full config on a GPU.** Anything
subsampled far enough to fit on a laptop starves the experts before it starves
the pipeline, which makes small runs useful for validating code and useless for
comparing methods. Budget roughly 4 × the compute of a single FedAvg run
(K experts trained in parallel over the same client population), times the number
of seeds, times the three assignment modes for the ablation in §9.

The one result in this report that does **not** need that compute is the
clustering study (§10.1–10.2) — `act_stats` requires no training at all, which is
precisely what makes it worth reporting on its own.

### 10.5 What is still missing, and why it matters

The §10.3 table is **one seed and one setting**. A 14-point margin is not seed
noise, so the direction is safe; the magnitudes are not error-barred.

What remains, in order of what it would actually teach you:

1. **`rotation_only` (α = 100).** The most informative missing run. It removes
   label skew entirely, so each expert still gets a quarter of the data but the
   remaining heterogeneity is *purely* the axis clustering targets. If the
   ensemble cannot win there, it cannot win anywhere in this design.
2. **Fewer clusters (K = 2).** Directly trades specialisation against the
   data-splitting cost this experiment identified as decisive. K = 2 halves rather
   than quarters the data per expert.
3. **Seeds beyond 42**, for error bars on the magnitudes.
4. **`dirichlet_only`.** Expected to show little gain — with no feature shift all
   experts see the same input distribution and converge to near-identical
   functions. Worth reporting honestly rather than omitting.

Note the `random`-assignment control has been **demoted**. It exists to rule out
capacity as the explanation for an ensemble *win*; the ensemble lost, and the
oracle ceiling already establishes that no combination strategy closes the gap.
It is still worth running for completeness, but it is no longer decisive.

All are in `scripts/run_sweep.sh`. They are compute-bound, not code-bound.

One further caveat that bounds the headline clustering result: ARI = 1.00 from an
**untrained** network means 90° rotations are an easy shift to detect. That is
strong evidence the 8.4M-dimensional gradient signature is unnecessary, and weak
evidence the approach transfers to subtler domain gaps — sensor, colour, style.
A natural-shift benchmark (PACS, Digit-5, FEMNIST) is the obvious next test.

---

## 11. File map and how to run

```
hefl/
├── datasets.py           rotation groups, Dirichlet split, per-rotation val/test
├── models.py             backbones, per-expert heads, 4 combiners, ClientView
├── clustering.py         warmup, 4 clustering signals, K-means, ARI vs both axes
├── federated.py          expert rounds, cluster-local aggregation, dense rounds
├── evaluation.py         per-rotation table, expert matrix, oracle, routing, subsets
├── baselines.py          FedAvg, random/oracle cluster assignment
├── utils.py              seeding, local SGD, FedAvg, logit adjustment
├── run.py                one-command pipeline, writes results.json + table.md
├── diagnose_signals.py   seconds-long check of the clustering signal
├── test_invariants.py    14 structural invariants (see 10.0)
└── configs/
    ├── smoke.json               CPU, minutes, validates every code path
    ├── fast.json                laptop/MPS, ~40 min, no-warmup clustering
    ├── medium.json              laptop/MPS, larger, with all clustering signals
    └── rotation_dirichlet.json  full CIFAR-10, GPU, the main experiment
```

Requirements: `torch`, `torchvision`, `scikit-learn`, `numpy`, `pillow`.
Developed against torch 2.9.1 / torchvision 0.24.1 / scikit-learn 1.6.1.
Run from the repository root (the package is imported as `hefl.*`).

```bash
# 1. verify the structural invariants (seconds, CPU)
python -m hefl.test_invariants

# 2. check the clustering signal before spending GPU time (seconds)
python -m hefl.diagnose_signals --num_clients 24 --alpha 0.5

# 3. validate every code path end to end (minutes, CPU)
python -m hefl.run --config hefl/configs/smoke.json

# 4. the main experiment
python -m hefl.run --config hefl/configs/rotation_dirichlet.json
```

Outputs per run, under `output_dir`:

| file | contents |
|---|---|
| `table.md` | the headline table |
| `results.json` | every metric, the clustering report, per-round history, config |
| `config_used.json` | full resolved config for reproducibility |
| `models/` | per-cluster backbones and heads, shared bias, each fitted combiner |

Useful overrides: `--assignment {learned,random,oracle}`, `--cluster_signal
{act_stats,delta_stem,delta_l4fc}`, `--combiners uniform,gate`, `--alpha`,
`--num_clusters`, `--logit_adjust`, `--seed`.

---

## 12. Limitations and what is deliberately not implemented

Stated plainly, because each is a question a reviewer will ask.

- **The gate's routing input is the wrong feature, and it is measured.** Routing
  accuracy 0.273 against a 0.25 chance level (§10.3). Feeding the gate per-expert
  confidence summaries instead of the input image keeps it at ~1k parameters, but
  it degenerates whenever the experts are not confidently distinguishable on a
  sample. A small CNN on `x` would fix it at the cost of one forward pass —
  though the measured +1.1-point ceiling says the fix is not worth much here.
- **Inference cost is K× FLOPs.** Every combiner here evaluates all K backbones.
  Top-1 routing (evaluate a cheap probe, then run one backbone) would bring
  inference cost back to FedAvg's, but it is not implemented — and note it
  depends on a routing signal that currently does not work.
- **Dense rounds cost K× download** on the rounds where they run. Reported, not
  hidden.
- **Privacy.** Nothing here shares raw data, but sharing Δθ is exactly the input
  to gradient-inversion attacks, and the `fc` update leaks the label histogram
  almost directly — which is *why* the label-axis clustering works at all.
  Per-client clustering also requires the server to see individual updates, which
  is incompatible with secure aggregation. `act_stats` is somewhat better in this
  respect (it shares pooled activation statistics rather than parameter updates)
  but it is not a formal privacy guarantee. The honest claim is "no raw data
  exchange", not "privacy-preserving".
- **Single dataset.** CIFAR-10 with synthetic rotations. Rotations are a
  convenient, exactly-controlled shift, but reviewers discount them; one natural
  feature-shift benchmark (Digit-5, PACS, Office-Caltech, FEMNIST) would
  substantially strengthen the claim.
- **No cross-seed aggregation in the runner.** Each run is one seed. Clustering
  is a discrete, unstable step, so ≥3 seeds with mean±std are required; the
  runner writes everything to `results.json` for external aggregation, but does
  not do it for you.
- **No expert diversity regularisation.** All experts start identical. Under
  rotation, diversity is forced by the domain shift. Under Dirichlet-only
  heterogeneity it is *not*, and near-identical experts make an ensemble worth
  little beyond variance reduction — which is very likely why the original
  Dirichlet-only headline experiment was the weak one. See §13.
- **`num_clusters` is fixed a priori.** Silhouette is computed and reported but
  never used to select K. CFL-style recursive bipartitioning would fix this.
- **No comparison to FedBN, IFCA, CFL, LG-FedAvg, FedRoD.** The architecture
  (personalised bodies, shared head) is LG-FedAvg; the clustering half is
  IFCA/CFL. These need to be cited and, ideally, benchmarked.

---

## 13. Mapping this onto a paper

The framing this implementation supports:

> Heterogeneity has two axes, and they want opposite treatment. Feature shift is
> a shallow, local phenomenon — visible in stem activation statistics *before any
> training* — and it is handled by personalised early computation. Label skew is
> a deep, global phenomenon, handled by a shared head and a shared prior.
> Clustering on the wrong axis conflates them; separating them turns out to be
> nearly free.

Concrete claims this repository can support, in order of strength:

1. **The feature axis is recoverable at essentially zero cost.** Gridded stem
   statistics on an untrained backbone, ~320 dims, ARI 1.00 against rotation and
   ~0.02 against labels, versus 0.55 for the 8.4M-dimensional gradient
   signature. The grid=1 vs grid=2 contrast makes it a finding rather than an
   implementation note. (Measured; §5.4, §10.2.)
2. **Cluster-specific training produces genuine domain specialists.** The
   oracle's expert→domain assignment is a bijection and each expert is ~2× better
   on its own domain (§10.3). Direct evidence, independent of any baseline.
3. **Under combined rotation + Dirichlet shift on CIFAR-10, the approach loses to
   FedAvg by 14 points — and the ensemble's own oracle ceiling sits 13 points
   below it.** Every component worked; partitioning the data was the binding
   constraint. This is the paper's most defensible empirical claim, and it is a
   *negative* one. (§10.3.)
4. **The headroom for learned combination is small, and was measured before
   being spent.** Optimal routing is worth **+1.1 points** over parameter-free
   averaging; every learned combiner scored below it. A quantified argument for
   the simplest possible combiner. (§10.3.)

**A claim an earlier draft of this report made and the data does not support:**
"the combination must be trained." Measured, learned combiners were *worse* than
uniform averaging (0.3442 / 0.3409 / 0.3139 vs 0.3478). The defensible version is
narrower: the original formulation's combination was untrainable *by
construction*, which is a design flaw worth fixing so the question can be asked —
and once asked, the answer here was that the simple rule wins.

Expected but unverified: under Dirichlet-only heterogeneity all experts see the
same input distribution and converge to near-identical functions, so the ensemble
should have little to combine. `dirichlet_only` in `tier1` tests it; present it
as an honest limitation rather than omitting it.

The most valuable single addition beyond what is here: a **shared trunk with
per-cluster adapters** instead of K full backbones. It keeps specialisation while
removing both the K× parameter confound and the data-splitting cost that is the
method's fundamental constraint.
