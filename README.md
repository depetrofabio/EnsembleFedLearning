
# Hierarchical Ensemble Federated Learning

This repository contains the implementation and experimental code for the paper **"Hierarchical Ensemble Federated Learning for Combined Feature and Label Heterogeneity"**.

Our framework proposes a novel **Cluster-Personalized Backbone, Shared Classifier** architecture designed to tackle the "conflicting gradient" problem in Federated Learning environments where clients suffer from both feature shifts (e.g., image rotations) and label skews (Dirichlet distribution).

## Key Features

*   **Weight-Based Clustering**: Groups clients based on the deep layer parameter updates ($\Delta \theta_{L4}, \Delta \theta_{FC}$) obtained during a local warmup phase.
*   **Hierarchical Ensemble**: Maintains $M$ specialized feature extractors (one per cluster) but shares a single global classifier to unify the label space.
*   **Active Normalization**: A specifically designed forward pass that allows the classifier to handle sparse inputs during training (1 active backbone) and dense inputs during inference (all active backbones) without magnitude mismatch.
*   **Privacy-Preserving**: Clustering and training are performed without ever sharing raw client data.

## Repository Structure

```
├── training/
│   ├── ensemble_fl.py      # Core logic for the Ensemble FL orchestrator (warmup, clustering, training)
│   ├── ensemble_model.py   # PyTorch definitions for the Ensemble and ResNet wrappers
│   └── utils.py            # Helper functions for training and evaluation
│
├── experiments/
│   ├── centralized_training.ipynb     # Baseline: Single model trained on central data (No Augmentation)
│   ├── fedavg_dirichlet_only.ipynb      # Baseline: Standard FedAvg under Label Heterogeneity
│   ├── ensemble_dirichlet_only.ipynb    # Main Method: Hierarchical Ensemble under Label Heterogeneity
│   ├── clustering_analysis.ipynb      # Validation: Analyzing clustering fidelity on Rotated CIFAR-10
│   └── config.json                    # Global configuration parameters
│
└── README.md               # This file
```

## Setup & Requirements

The code is implemented in **Python 3.8+** using **PyTorch**.

To install dependencies:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

## Running Experiments

### 1. Baselines
To establish performance benchmarks, we use two baselines:
*   **Centralized**: Run `experiments/centralized_training.ipynb` to train a ResNet-18 on the full dataset with a 90/10 Train/Val split and no data augmentation.
*   **FedAvg**: Run `experiments/fedavg_dirichlet_only.ipynb` to evaluate standard Federated Averaging under a Dirichlet ($\alpha=0.5$) non-IID distribution.

### 2. Clustering Validation
Run `experiments/clustering_analysis.ipynb` to verify the grouping mechanism.
*   **Scenario**: Clients are assigned to 4 fixed rotation groups ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
*   **Goal**: Confirm that K-Means on weight differences successfully recovers these ground-truth groups (High ARI score).

### 3. Ensemble Training (Notebooks)
Run `experiments/ensemble_dirichlet_only.ipynb` for the main method.
*   **Scenario**: Standard CIFAR-10 (No Rotations) with Dirichlet label heterogeneity.
*   **Process**:
    1.  **Warmup**: Clients train locally for 2 epochs.
    2.  **Clustering**: Server groups clients based on weight updates.
    3.  **Ensemble Training**: Clients train their cluster-specific backbone + shared classifier.

### 4. Running Experiments with `run_ensemble.py`

For systematic experimentation and reproducibility, use the `run_ensemble.py` script with flexible configuration options.

#### Quick Start

```bash
# Run with default configuration
python run_ensemble.py --config config.json

# Quick test with fewer clients and rounds
python run_ensemble.py --config config_quick_test.json

# Override specific parameters via CLI
python run_ensemble.py --config config.json --num_clients 50 --lr 0.001
```

#### Configuration Options

The script supports configuration via both JSON files and command line arguments (CLI overrides JSON):

**Key Parameters:**
- `num_clients`: Number of federated clients (default: 100)
- `num_clusters`: Number of clusters for ensemble (default: 5)
- `alpha`: Dirichlet distribution parameter (lower = more non-IID, default: 0.5)
- `batch_size`: Training batch size (default: 64)
- `lr`: Learning rate (default: 0.01)
- `warmup_rounds`: Warmup phase rounds (default: 1)
- `warmup_local_epochs`: Local epochs during warmup (default: 2)
- `use_fedavg_warmup`: Use FedAvg warmup (default: False)
- `use_weight_diff`: Use weight differences for clustering (default: True)
- `clustering_method`: Clustering approach - `kmeans` (default: kmeans)
- `ensemble_rounds`: Main training rounds (default: 50)
- `seed`: Random seed for reproducibility (default: 42)

**Complete parameter list**: Run `python run_ensemble.py --help`

#### Example Experiments

**Experiment 1: Varying Number of Clusters**
```bash
# Test different cluster configurations
python run_ensemble.py --config config.json --num_clusters 3 --output_dir ./results_k3
python run_ensemble.py --config config.json --num_clusters 5 --output_dir ./results_k5
python run_ensemble.py --config config.json --num_clusters 10 --output_dir ./results_k10
```

**Experiment 2: Different Non-IID Levels**
```bash
# High heterogeneity (very non-IID)
python run_ensemble.py --config config.json --alpha 0.1 --output_dir ./results_alpha_0.1

# Medium heterogeneity
python run_ensemble.py --config config.json --alpha 0.5 --output_dir ./results_alpha_0.5

# Low heterogeneity (more IID)
python run_ensemble.py --config config.json --alpha 10.0 --output_dir ./results_alpha_10
```

**Experiment 3: Clustering Method Comparison**
```bash
# Gradient-based clustering (KMeans on weight differences)
python run_ensemble.py --config config.json --clustering_method kmeans

# Feature-based clustering
python run_ensemble.py --config config.json --clustering_method features
```

**Experiment 4: Extended Training**
```bash
# More warmup and training rounds
python run_ensemble.py \
    --config config.json \
    --warmup_rounds 5 \
    --warmup_local_epochs 5 \
    --ensemble_rounds 100 \
    --output_dir ./results_extended
```

#### Output Structure

Results are saved to the specified output directory:
```
results/
├── training_history.json         # Loss & accuracy per round
├── config_used.json              # Configuration for reproducibility
└── models/
    ├── cluster_model_0.pth       # Cluster-specific feature extractors
    ├── cluster_model_1.pth
    ├── ...
    ├── ensemble_classifier.pth   # Shared global classifier
    └── client_clusters.json      # Client-to-cluster assignments
```

#### Google Colab

For cloud-based experimentation, use the provided Colab notebook:

1. Upload `ensemble_training_colab.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: Runtime → Change runtime type → GPU
3. Follow the notebook cells to setup and run experiments

See [`ENSEMBLE_TRAINING_GUIDE.md`](ENSEMBLE_TRAINING_GUIDE.md) for detailed documentation and troubleshooting.
