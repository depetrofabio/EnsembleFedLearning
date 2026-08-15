# Ensemble Federated Learning: Complete Technical Explanation

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Architecture Components](#architecture-components)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Forward Pass in Detail](#forward-pass-in-detail)
5. [Training Phases](#training-phases)
6. [Normalization and Why It Matters](#normalization-and-why-it-matters)
7. [Complete Example Walkthrough](#complete-example-walkthrough)
8. [Implementation Deep Dive](#implementation-deep-dive)

---

## High-Level Overview

The **Ensemble Federated Learning** method addresses data heterogeneity in federated learning by:

1. **Clustering clients** based on their data similarity (via gradient patterns)
2. **Creating cluster-specific feature extractors** (one per cluster)
3. **Sharing a global classifier** across all clients
4. **Using an ensemble approach** where predictions combine features from all cluster models

### Key Insight
Instead of forcing all clients to use the same model (FedAvg), we recognize that clients with different data distributions need different feature extractors, but can still share a common classifier for the final prediction task.

---

## Architecture Components

### 1. ResNetFeatureExtractor

**Purpose**: Extract features from images without classification

**Code Location**: [`ensemble_model.py`, lines 73-93]

```python
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.features = nn.Sequential(
            original_model.conv1,     # Initial convolution
            original_model.bn1,       # Batch normalization
            original_model.relu,      # Activation
            original_model.maxpool,   # Pooling
            original_model.layer1,    # ResNet block 1
            original_model.layer2,    # ResNet block 2
            original_model.layer3,    # ResNet block 3
            original_model.layer4,    # ResNet block 4
            original_model.avgpool    # Global average pooling
        )
    
    def forward(self, x):
        x = self.features(x)       # Shape: (B, 512, 1, 1) for ResNet18
        x = torch.flatten(x, 1)    # Shape: (B, 512)
        return x
```

**What it does**:
- Takes input image: `x ∈ ℝ^(B×3×32×32)` (for CIFAR-10)
- Passes through all ResNet layers **except** the final FC layer
- Outputs feature vector: `z ∈ ℝ^(B×512)`

**Why remove FC layer?**
- FC layer is task-specific (10 classes for CIFAR-10)
- We want reusable features that can be combined
- The ensemble's **shared classifier** will handle the classification

---

### 2. EnsembleModel

**Purpose**: Combine multiple feature extractors with a shared classifier

**Code Location**: [`ensemble_model.py`, lines 5-71]

#### Architecture Diagram

```
Input Image (B, 3, 32, 32)
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         │                 │                 │                 │
    Feature Ext 0     Feature Ext 1     Feature Ext 2    ... Ext K-1
    (Cluster 0)       (Cluster 1)       (Cluster 2)         (Cluster K-1)
         │                 │                 │                 │
    (B, 512)          (B, 512)          (B, 512)          (B, 512)
         │                 │                 │                 │
         └─────────────────┴─────────────────┴─────────────────┘
                                 │
                          Concatenate
                                 │
                          (B, 512*K)
                                 │
                       Normalize by #active
                                 │
                      Shared Classifier
                          (Linear Layer)
                                 │
                          (B, num_classes)
```

#### Key Components

```python
class EnsembleModel(nn.Module):
    def __init__(self, models, feature_dim, num_classes):
        super().__init__()
        self.models = nn.ModuleList(models)      # K feature extractors
        self.num_models = len(models)             # K clusters
        self.feature_dim = feature_dim            # 512 for ResNet18
        
        # Shared classifier: (K × 512) → num_classes
        self.classifier = nn.Linear(
            self.num_models * self.feature_dim,   # Input: K*512
            num_classes                            # Output: 10 for CIFAR-10
        )
```

**Dimensions**:
- `self.models`: List of K feature extractors
- `self.feature_dim`: 512 (ResNet18 output dimension)
- `self.classifier.weight`: Shape `(num_classes, K × feature_dim)` = `(10, K×512)`
- `self.classifier.bias`: Shape `(num_classes,)` = `(10,)`

---

## Mathematical Formulation

### Notation

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| `x` | Input image | `(B, 3, 32, 32)` |
| `K` | Number of clusters | scalar |
| `d` | Feature dimension | 512 (ResNet18) |
| `C` | Number of classes | 10 (CIFAR-10) |
| `f_k(·)` | Feature extractor for cluster k | `ℝ^(3×32×32) → ℝ^d` |
| `z_k` | Features from cluster k | `(B, d)` |
| `Z` | Concatenated features | `(B, K×d)` |
| `A` | Set of active cluster indices | `{0, 1, ..., K-1}` subset |
| `W` | Classifier weight matrix | `(C, K×d)` |
| `b` | Classifier bias vector | `(C,)` |

### Forward Pass Equation

Given an input `x`, the ensemble prediction is:

```
Step 1: Extract features from each cluster model
    z_k = f_k(x)    for k ∈ {0, 1, ..., K-1}
    where z_k ∈ ℝ^(B×d)

Step 2: Concatenate features (with masking for active clusters)
    For each k:
        if k ∈ A (active):
            Z[:, k×d : (k+1)×d] = z_k
        else:
            Z[:, k×d : (k+1)×d] = 0
    
    Result: Z ∈ ℝ^(B×(K×d))

Step 3: Normalize by number of active clusters
    Z_norm = Z / |A|
    where |A| is the number of active clusters

Step 4: Apply classifier
    logits = Z_norm × W^T + b
    where logits ∈ ℝ^(B×C)

Step 5: Get predictions (during inference)
    predictions = argmax(logits, dim=1)
    or
    probabilities = softmax(logits, dim=1)
```

### Full Mathematical Expression

```
ŷ = W × (1/|A| × Concat[z_k for k ∈ A, 0 for k ∉ A]) + b

where:
  z_k = f_k(x)                    (feature extraction)
  Concat[·] creates (B, K×d)      (concatenation)
  1/|A| is scalar normalization   (averaging)
  W ∈ ℝ^(C×K×d), b ∈ ℝ^C          (learned parameters)
```

---

## Forward Pass in Detail

Let's trace through a concrete example with **real numbers**.

### Example Setup
- Batch size `B = 2` (two images)
- Number of clusters `K = 3`
- Feature dimension `d = 512` (ResNet18)
- Number of classes `C = 10` (CIFAR-10)
- Active clusters `A = {0, 2}` (only clusters 0 and 2 are active)

### Step-by-Step Execution

#### Step 1: Feature Extraction
```python
# Input: x.shape = (2, 3, 32, 32)

z_0 = f_0(x)  # Cluster 0 feature extractor
# z_0.shape = (2, 512)
# Example values: z_0[0, :] = [0.5, -0.3, 0.8, ..., 0.2]  (512 values)

z_1 = f_1(x)  # Cluster 1 feature extractor (NOT ACTIVE)
# z_1.shape = (2, 512) - computed but will be zeroed

z_2 = f_2(x)  # Cluster 2 feature extractor
# z_2.shape = (2, 512)
# Example values: z_2[0, :] = [0.1, 0.7, -0.4, ..., 0.9]  (512 values)
```

#### Step 2: Concatenation with Masking

```python
# Create empty tensor for ALL features
all_features = torch.zeros(B, K * d, device=x.device)
# all_features.shape = (2, 3*512) = (2, 1536)

# Fill in ONLY active clusters
# Cluster 0 (active, index 0):
all_features[:, 0*512:1*512] = z_0
# all_features[:, 0:512] = z_0

# Cluster 1 (inactive, index 1):
# all_features[:, 1*512:2*512] remains zeros
# all_features[:, 512:1024] = [0, 0, 0, ..., 0]

# Cluster 2 (active, index 2):
all_features[:, 2*512:3*512] = z_2
# all_features[:, 1024:1536] = z_2

# Result: all_features.shape = (2, 1536)
# all_features[0, :] = [z_0[0], zeros(512), z_2[0]]
#                    = [0.5, -0.3, ..., 0.2, 0, 0, ..., 0, 0.1, 0.7, ..., 0.9]
#                      └─────512 vals─────┘  └───512 zeros──┘  └─────512 vals─────┘
```

#### Step 3: Normalization

```python
num_active = len(A) = 2  # Clusters 0 and 2

all_features = all_features / num_active
# all_features = all_features / 2

# Effect on each position:
# - Features from cluster 0: divided by 2
# - Zeros from cluster 1: remain zeros
# - Features from cluster 2: divided by 2

# Example for first image:
# all_features[0, :] = [0.25, -0.15, ..., 0.1, 0, 0, ..., 0, 0.05, 0.35, ..., 0.45]
#                       └──z_0/2──┘             └──0's──┘         └──z_2/2──┘
```

**Why divide by 2?**
- We're averaging the contributions from active clusters
- If we used all 3 clusters, features would be divided by 3
- This normalization ensures the magnitude is consistent regardless of how many clusters are active

#### Step 4: Classifier Application

```python
# Classifier parameters
W.shape = (10, 1536)  # Weight matrix
b.shape = (10,)       # Bias vector

# Matrix multiplication
logits = all_features @ W.T + b
# Shape: (2, 1536) @ (1536, 10) + (10,) = (2, 10)

# For each sample, we get 10 logit values (one per class)
# Example for first image:
# logits[0, :] = [2.3, -1.2, 0.5, 3.1, -0.8, 1.4, 0.2, -2.1, 1.9, 0.7]
#                 └────────────────10 class scores────────────────┘
```

**Classifier Weight Structure**:
```
W = [w_0,0→0,511  | w_0,512→1023  | w_0,1024→1535]  ← weights for class 0
    [w_1,0→0,511  | w_1,512→1023  | w_1,1024→1535]  ← weights for class 1
    ...
    [w_9,0→0,511  | w_9,512→1023  | w_9,1024→1535]  ← weights for class 9
     └─cluster 0─┘  └─cluster 1─┘  └─cluster 2─┘
```

When cluster 1 is inactive (all zeros), the middle section `w_c,512→1023` doesn't contribute to the final score for any class `c`.

#### Step 5: Prediction

```python
# During training: Use logits with cross-entropy loss
loss = CrossEntropyLoss()(logits, labels)

# During inference: Get class predictions
predictions = torch.argmax(logits, dim=1)
# predictions.shape = (2,)
# Example: predictions = [3, 7]  # First image → class 3, Second image → class 7

# Or get probabilities
probabilities = torch.softmax(logits, dim=1)
# probabilities.shape = (2, 10)
# Each row sums to 1.0
```

---

## Training Phases

### Phase 1: Warmup

**Goal**: Train initial models and collect gradient information for clustering

**Two modes available**:

#### Mode A: FedAvg Warmup
```python
# All clients train a shared global model
for round in range(warmup_rounds):
    for each selected client:
        1. Download global model
        2. Train locally for local_epochs
        3. Compute gradients ∇L_i (loss gradient)
        4. Upload model updates
    
    5. Aggregate updates → new global model
    6. Store gradients for clustering
```

#### Mode B: Single Client Finetuning (Used in notebook)
```python
# Each client trains independently from scratch
for each client_i:
    1. Initialize model from global_model (or random)
    2. Train for local_epochs
    3. Collect gradients during training:
        For each epoch:
            For each batch:
                - Compute loss L
                - Backpropagate: ∇L
                - Extract gradients from layer4 and fc
                - Average gradients across batches → epoch_grad
        4. Store: warmup_gradients[client_i] = [epoch_1_grad, epoch_2_grad]

# NO aggregation - just gradient collection
```

**Gradient Extraction**:
```python
# During warmup, for each client, we extract:
gradient_vector = concat([
    ∇(layer4.conv1.weight),
    ∇(layer4.conv1.bias),
    ∇(layer4.conv2.weight),
    ∇(layer4.conv2.bias),
    ...
    ∇(fc.weight),
    ∇(fc.bias)
])

# This captures BOTH:
# - Feature-level heterogeneity (layer4 gradients)
# - Task-level heterogeneity (fc gradients)
```

**Why layer4 + FC?**
- `layer4`: High-level features, sensitive to data distribution
- `fc`: Task-specific, captures label distribution differences
- Together: Comprehensive view of client heterogeneity

---

### Phase 2: Clustering

**Goal**: Group clients with similar data distributions

#### Step 1: Prepare Gradient Matrix

```python
# Average gradients across warmup epochs for each client
gradient_matrix = np.zeros((num_clients, gradient_dim))

for client_i in range(num_clients):
    # Average across all warmup epochs
    avg_gradient = np.mean(warmup_gradients[client_i], axis=0)
    gradient_matrix[client_i, :] = avg_gradient

# gradient_matrix.shape = (60, ~2.6M)  for ResNet18 layer4+fc
```

#### Step 2: Normalize Gradients

```python
# Subtract mean (center the data)
mean_gradient = gradient_matrix.mean(axis=0)
gradient_differences = gradient_matrix - mean_gradient

# Normalize each client's gradient vector to unit length
for i in range(num_clients):
    norm = np.linalg.norm(gradient_differences[i])
    gradient_differences[i] = gradient_differences[i] / (norm + 1e-8)

# This makes clustering focus on DIRECTION, not magnitude
```

**Why normalize?**
- Clients with more data would have larger gradients
- We care about gradient **direction** (which features/classes are important)
- Not gradient **magnitude** (how much data the client has)

#### Step 3: K-Means Clustering

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
cluster_labels = kmeans.fit_predict(gradient_differences)

# cluster_labels[i] ∈ {0, 1, ..., K-1} indicates which cluster client i belongs to
```

**Mathematical formulation**:
```
Find cluster assignments C* that minimize:

J = Σ_{k=0}^{K-1} Σ_{i: client_i ∈ cluster_k} ||g_i - μ_k||²

where:
  g_i = normalized gradient vector for client i
  μ_k = centroid of cluster k
  || · || = Euclidean distance
```

**Result**:
```python
# Example with 60 clients, 3 clusters:
cluster_labels = [0, 0, 1, 0, 2, 1, 0, 2, ..., 1, 2, 0]
                  └─────────────60 values─────────────┘

# Distribution might be:
# Cluster 0: 25 clients
# Cluster 1: 20 clients  
# Cluster 2: 15 clients
```

---

### Phase 3: Ensemble Initialization

**Goal**: Create cluster-specific feature extractors and shared classifier

#### Step 1: Create Feature Extractors

```python
# Use the global model from warmup as base
base_feature_extractor = ResNetFeatureExtractor(global_model)

cluster_models = []
for k in range(K):
    # Deep copy for each cluster
    cluster_models.append(copy.deepcopy(base_feature_extractor))

# Result: K independent feature extractors, all initialized with same weights
```

#### Step 2: Initialize Shared Classifier

**Key Challenge**: Expand FC layer from `(512 → 10)` to `(K×512 → 10)`

```python
old_fc = global_model.fc  # Shape: (10, 512)
new_fc = nn.Linear(K * 512, 10)  # Shape: (10, K*512)

# Copy old weights K times
with torch.no_grad():
    for k in range(K):
        start = k * 512
        end = (k + 1) * 512
        new_fc.weight[:, start:end] = old_fc.weight
        # Each cluster's section gets a copy of the original weights
    
    # Bias remains the same
    new_fc.bias.copy_(old_fc.bias)
```

**Visualization of Weight Initialization**:
```
Old FC weights (10 × 512):
[w_0,0  w_0,1  ...  w_0,511]  ← class 0 weights
[w_1,0  w_1,1  ...  w_1,511]  ← class 1 weights
...
[w_9,0  w_9,1  ...  w_9,511]  ← class 9 weights

New FC weights (10 × K*512), for K=3:
[w_0,0 ... w_0,511 | w_0,0 ... w_0,511 | w_0,0 ... w_0,511]  ← class 0
[w_1,0 ... w_1,511 | w_1,0 ... w_1,511 | w_1,0 ... w_1,511]  ← class 1
...
[w_9,0 ... w_9,511 | w_9,0 ... w_9,511 | w_9,0 ... w_9,511]  ← class 9
 └────cluster 0────┘  └────cluster 1────┘  └────cluster 2────┘
```

**Why copy K times?**
- Ensures ensemble starts with the same capacity as the warmup model
- Each cluster's features initially contribute equally
- During training, cluster-specific sections will diverge

---

### Phase 4: Hierarchical Training

**Goal**: Train cluster-specific feature extractors while sharing the classifier

#### Training Strategy

**NOT Federated**: In the notebook, data from all clients in a cluster is **pooled**
```python
for round in range(training_rounds):
    for cluster_k in range(K):
        # Get all clients in this cluster
        cluster_clients = [i for i in range(num_clients) if cluster_labels[i] == k]
        
        # POOL all their data together (not federated!)
        cluster_dataset = ConcatDataset([
            train_subsets[client_i] for client_i in cluster_clients
        ])
        
        # Train ensemble model using ONLY features from cluster k
        for epoch in range(local_epochs):
            for batch in cluster_dataset:
                # Forward: use only f_k (cluster k's feature extractor)
                features = cluster_models[k](inputs)
                
                # Create full feature vector
                full_features = zeros(batch_size, K * 512)
                full_features[:, k*512:(k+1)*512] = features
                
                # Normalize (only 1 active cluster)
                full_features = full_features / 1
                
                # Classify
                logits = classifier(full_features)
                loss = CrossEntropyLoss()(logits, labels)
                
                # Backward: updates BOTH f_k and classifier
                loss.backward()
                optimizer.step()
```

**Alternative: True Federated Training** (in `ensemble_fl.py`)
```python
# Each client trains their cluster's feature extractor + global classifier
for round in range(training_rounds):
    selected_clients = sample(clients, fraction)
    
    for client_i in selected_clients:
        cluster_k = cluster_labels[client_i]
        
        # Client downloads:
        # 1. Feature extractor for their cluster: f_k
        # 2. Global classifier
        
        # Train locally
        for epoch in range(local_epochs):
            for batch in client_data:
                # Forward through client's cluster model
                features = f_k(inputs)
                full_features[k*512:(k+1)*512] = features
                logits = classifier(full_features / 1)
                loss = ...
                loss.backward()
                optimizer.step()
        
        # Upload updates for f_k and classifier
    
    # Server aggregates:
    # - f_k updates from all clients in cluster k
    # - Classifier updates from ALL clients
```

---

## Normalization and the Train/Test Regime

> **Corrected.** An earlier version of this section argued the case *train with
> all K clusters active, evaluate with 1*. The implementation does the
> **opposite**, and so does Algorithm 5 further down this document. The analysis
> below describes what the code actually does.

### What the code does

```python
all_features = all_features / num_active     # ensemble_model.py
```

| phase | active clusters | divisor | resulting logits |
|---|---|---|---|
| client training | 1 (the client's own cluster `c`) | 1 | `W_c z_c + b` |
| ensemble inference | all `K` | `K` | `(1/K) Σ_k W_k z_k + b` |

So the division by `|A|` is not a correction for a magnitude mismatch — it is
what makes inference a **uniform average of the per-cluster logits** rather than
a sum. Averaging keeps the scale independent of how many clusters are active,
which is what allows any subset `A` to be evaluated.

### The consequence this creates

Because a linear classifier over a block-sparse input satisfies

```
W · concat(z_1, …, z_K) = Σ_k W_k z_k
```

and because only block `c` is ever non-zero during client training, the
gradient with respect to every other column block is exactly zero. `W` is
therefore never anything but `K` independently trained heads plus a shared
bias, and the `1/K` combination at inference is a **constant that no gradient
ever saw**.

Two honest implications:

1. The combination is not learned. Calling it "the classifier learns optimal
   combinations of cluster-specific representations" is not supported by this
   architecture.
2. Feature-level fusion through a *linear* head is mathematically identical to
   prediction-level fusion, so it is not a departure from ensembles that
   "aggregate predictions".

Both are addressed in `hefl/`, where the combination is an explicit
module trained in dedicated rounds. See `docs/REPORT.md` §4 and §7.

## Complete Example Walkthrough

Let's trace a complete training and inference example with concrete numbers.

### Setup
- **Dataset**: CIFAR-10
- **Clients**: 60 total
- **Clusters**: K = 3
- **Architecture**: ResNet18
  - Feature dimension: d = 512
  - Classes: C = 10

### Phase 1: Warmup (Simplified)

```python
# Client 0: Trains on data with lots of "airplane" and "ship"
# - Gradients indicate strong learning for classes 0 and 8
# - Layer4 learns features good for these classes

# Client 1: Trains on data with lots of "cat" and "dog"
# - Gradients indicate strong learning for classes 3 and 5
# - Layer4 learns features good for these classes

# ... (all 60 clients)

# Gradient matrix (simplified to 2D for visualization):
gradients = [
    [0.8, 0.1],  # Client 0: high gradient for "airplane features"
    [0.2, 0.7],  # Client 1: high gradient for "animal features"
    [0.75, 0.15],# Client 2: similar to client 0
    ...
]
```

### Phase 2: Clustering

```python
# K-Means groups clients by gradient similarity
cluster_labels = [
    0, 1, 0, 2, 1, 0, ...  # 60 values
]

# Result:
# Cluster 0: Clients with vehicle-heavy data (25 clients)
# Cluster 1: Clients with animal-heavy data (20 clients)
# Cluster 2: Clients with mixed data (15 clients)
```

### Phase 3: Initialize Ensemble

```python
# Create 3 feature extractors
f_0 = ResNetFeatureExtractor(global_model)  # For cluster 0 (vehicles)
f_1 = ResNetFeatureExtractor(global_model)  # For cluster 1 (animals)
f_2 = ResNetFeatureExtractor(global_model)  # For cluster 2 (mixed)

# Create shared classifier: (3×512) → 10
classifier = Linear(1536, 10)

# Initialize with copied weights from warmup model
```

### Phase 4: Training (1 Round)

```python
# Train cluster 0 on aggregated vehicle-heavy data
cluster_0_data = concat([train_data[i] for i in [0, 2, 5, ...]])  # 25 clients

for batch in cluster_0_data:
    # Forward
    features_0 = f_0(batch)  # (32, 512)
    
    # Create full feature vector
    full = zeros(32, 1536)
    full[:, 0:512] = features_0
    full[:, 512:1024] = 0  # Cluster 1 inactive
    full[:, 1024:1536] = 0  # Cluster 2 inactive
    
    # Normalize
    full = full / 1  # Only 1 cluster active
    
    # Classify
    logits = classifier(full)  # (32, 10)
    loss = CrossEntropyLoss()(logits, labels)
    
    # Backward
    loss.backward()
    # Updates: f_0 and classifier
    optimizer.step()

# Similarly for clusters 1 and 2
```

After training:
- `f_0` specializes in vehicle features
- `f_1` specializes in animal features
- `f_2` handles mixed cases
- `classifier` learns to work with features from ANY cluster

### Phase 5: Evaluation

**Scenario 1**: Test on single cluster (e.g., cluster 0 only)
```python
test_image = load_image("airplane.jpg")  # Shape: (1, 3, 32, 32)

# Use only cluster 0
features_0 = f_0(test_image)  # (1, 512)

# Full features
full = zeros(1, 1536)
full[:, 0:512] = features_0
full = full / 1  # Normalize by 1 active

# Predict
logits = classifier(full)  # (1, 10)
prediction = argmax(logits)  # Result: class 0 (airplane)
```

**Scenario 2**: Test with ALL clusters (full ensemble)
```python
test_image = load_image("cat.jpg")  # Shape: (1, 3, 32, 32)

# Use all 3 clusters
features_0 = f_0(test_image)  # (1, 512)
features_1 = f_1(test_image)  # (1, 512)
features_2 = f_2(test_image)  # (1, 512)

# Full features
full = concat([features_0, features_1, features_2], dim=1)  # (1, 1536)
full = full / 3  # Normalize by 3 active

# Predict
logits = classifier(full)  # (1, 10)
# logits might be: [0.1, 0.3, 0.2, 5.7, 0.5, 2.3, 0.4, 0.1, 0.6, 0.2]
#                                      └─ highest (cat = class 3)
prediction = argmax(logits)  # Result: class 3 (cat)
```

**Comparison**:
```
Cluster 1 only (animal specialist): confidence = 0.92 (cat)
Cluster 0 only (vehicle specialist): confidence = 0.31 (cat)
Cluster 2 only (mixed): confidence = 0.68 (cat)
ENSEMBLE (all 3): confidence = 0.95 (cat)  ← Best!
```

---

## Implementation Deep Dive

### Code Location Map

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `ResNetFeatureExtractor` | `ensemble_model.py` | 73-93 | Extract features without classification |
| `EnsembleModel` | `ensemble_model.py` | 5-71 | Combine K extractors + shared classifier |
| `EnsembleFedAvg` | `ensemble_fl.py` | 16-504 | Orchestrate warmup, clustering, training |
| `ClientEnsembleWrapper` | `ensemble_fl.py` | 505-535 | Client-side model for federated training |

### Key Implementation Details

#### 1. Feature Concatenation with Masking

```python
# From ensemble_model.py, lines 40-62
all_features = torch.zeros(batch_size, self.num_models * self.feature_dim, device=x.device)

for idx in active_indices:
    features = self.models[idx](x)  # Run feature extractor
    
    # Place in correct position
    start_idx = idx * self.feature_dim
    end_idx = (idx + 1) * self.feature_dim
    all_features[:, start_idx:end_idx] = features
```

**Why zeros for inactive clusters?**
- Mathematically equivalent to not using them
- Simpler implementation than variable-length concatenation
- Allows for dynamic activation of different clusters

#### 2. Normalization Implementation

```python
# From ensemble_model.py, lines 64-66
num_active = len(active_indices)
all_features = all_features / num_active
```

**Critical**: This happens BEFORE the classifier
- Ensures feature magnitudes are consistent
- Independent of how many clusters are active

#### 3. Gradient Extraction During Warmup

```python
# From ensemble_fl.py, lines 156-161
current_batch_grad = []
for name, p in local_model.named_parameters():
    if p.grad is not None and ('layer4' in name or 'fc' in name):
        current_batch_grad.append(p.grad.view(-1))

flat_batch_grad = torch.cat(current_batch_grad)
```

**Why flatten?**
- Gradient tensors have different shapes (conv: 4D, fc: 2D, bias: 1D)
- Clustering needs a single vector per client
- `.view(-1)` flattens any shape to 1D

#### 4. Classifier Weight Initialization

```python
# From ensemble_fl.py, lines 375-388
old_fc = self.global_model.fc
new_fc = nn.Linear(self.feature_dim * self.num_clusters, 10)

with torch.no_grad():
    for i in range(self.num_clusters):
        start_col = i * self.feature_dim
        end_col = (i + 1) * self.feature_dim
        new_fc.weight[:, start_col:end_col] = old_fc.weight
    
    new_fc.bias.copy_(old_fc.bias)
```

**Ensures**:
- Ensemble starts at same performance as warmup model
- Each cluster contributes equally initially
- Training can specialize each section

---

## Summary

### Architecture Summary

```
Ensemble = K Feature Extractors + 1 Shared Classifier

Input → [f_0, f_1, ..., f_{K-1}] → Concat → Normalize → Classifier → Output
        └─────────────────────────┘
         Cluster-specific features
                                              └──────────┘
                                            Shared across clusters
```

### Mathematical Summary

```
ŷ = softmax(W × (1/|A| × Z) + b)

where:
  Z = concat([z_0, z_1, ..., z_{K-1}])
  z_k = f_k(x) if k ∈ A else 0
  |A| = number of active clusters
  W, b = learned classifier parameters
```

### Training Summary

1. **Warmup**: Collect gradients → understand client heterogeneity
2. **Cluster**: Group similar clients → K clusters
3. **Initialize**: Create K feature extractors + 1 classifier
4. **Train**: Specialize extractors, share classifier
5. **Ensemble**: Combine all for prediction

### Key Advantages

✅ **Handles heterogeneity**: Different feature extractors for different data types  
✅ **Shared knowledge**: Classifier learns from all clients  
✅ **Flexible inference**: Can use 1, some, or all clusters  
✅ **Better than FedAvg**: Doesn't force one model to fit all data

---

## Additional Resources

- **Notebook**: `experiments/ensemble_dirichlet_only.ipynb`
- **Implementation**: `training/ensemble_model.py`, `training/ensemble_fl.py`
- **Cluster Analysis**: See previous documentation on cluster combinations

For questions about specific implementation details, refer to the inline comments in the source code.
