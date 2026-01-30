
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
├── paper_full_body.tex     # Draft of the research paper
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

### 3. Ensemble Training
Run `experiments/ensemble_dirichlet_only.ipynb` for the main method.
*   **Scenario**: Standard CIFAR-10 (No Rotations) with Dirichlet label heterogeneity.
*   **Process**:
    1.  **Warmup**: Clients train locally for 2 epochs.
    2.  **Clustering**: Server groups clients based on weight updates.
    3.  **Ensemble Training**: Clients train their cluster-specific backbone + shared classifier.
