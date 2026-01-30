# Clustering Methods Comparison - Configuration Guide

## Overview
This guide explains how to configure the clustering comparison experiments with different numbers of rotations and clusters.

## Configuration Parameters

### `config.json` Parameters

```json
{
    "num_clients": 50,              // Total number of federated clients
    "num_rotation_clusters": 4,     // Number of rotation angles (data heterogeneity)
    "num_clusters_model": 4,        // Number of clusters for the ensemble model
    "seed": 42,                     // Random seed for reproducibility
    "model_name": "resnet18",       // Model architecture
    "pretrained": false,            // Use pretrained weights
    "batch_size": 64,               // Training batch size
    "lr": 0.01,                     // Learning rate
    "warmup_epochs": 2,             // Warmup training epochs
    "training_rounds": 30,          // FL training rounds
    "client_fraction": 0.2          // Fraction of clients per round
}
```

## Key Parameters Explained

### `num_rotation_clusters` (Data Heterogeneity)
Controls how many different rotation angles are applied to create non-IID data.

**Examples:**
- `2` → [0°, 180°] (minimal heterogeneity)
- `3` → [0°, 120°, 240°] 
- `4` → [0°, 90°, 180°, 270°] (default)
- `6` → [0°, 60°, 120°, 180°, 240°, 300°]
- `8` → [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]

Rotation angles are **automatically generated** to be evenly distributed across 360°:
```
angle = 360° × i / num_rotation_clusters, where i ∈ [0, num_rotation_clusters-1]
```

### `num_clusters_model` (Ensemble Clusters)
Controls how many specialized models the ensemble should have.

**Relationship with `num_rotation_clusters`:**
- **Matching** (`num_clusters_model` = `num_rotation_clusters`): Ideal case, one model per rotation
- **Fewer** (`num_clusters_model` < `num_rotation_clusters`): Models must handle multiple rotations
- **More** (`num_clusters_model` > `num_rotation_clusters`): Potential over-clustering

**Best Practice:** Start with `num_clusters_model` = `num_rotation_clusters` for optimal specialization.

## Usage Examples

### Example 1: Binary Heterogeneity
```json
{
    "num_rotation_clusters": 2,
    "num_clusters_model": 2
}
```
- Data: 0° and 180° rotations
- Models: 2 specialized models
- Use case: Testing basic clustering

### Example 2: High Heterogeneity
```json
{
    "num_rotation_clusters": 8,
    "num_clusters_model": 8
}
```
- Data: 8 different rotations (45° intervals)
- Models: 8 specialized models  
- Use case: Testing fine-grained specialization

### Example 3: Cluster Mismatch
```json
{
    "num_rotation_clusters": 6,
    "num_clusters_model": 3
}
```
- Data: 6 rotations
- Models: 3 specialized models (each handles 2 rotations)
- Use case: Testing model capacity under-provisioning

### Example 4: Many Clients, Few Clusters
```json
{
    "num_clients": 100,
    "num_rotation_clusters": 4,
    "num_clusters_model": 4
}
```
- More data per rotation cluster
- Better statistical significance
- Use case: Production-like scenario

## Important Notes

1. **Client Distribution**: Clients are evenly distributed across rotation clusters
   - With 50 clients and 4 rotations: ~12-13 clients per rotation
   - With 50 clients and 8 rotations: ~6-7 clients per rotation

2. **Minimum Clients per Cluster**: Ensure `num_clients / num_rotation_clusters ≥ 5` for meaningful statistics

3. **Clustering Evaluation**: The code automatically computes:
   - Adjusted Rand Index (ARI): measures agreement with true rotations
   - Alignment Accuracy: % of clients correctly clustered
   - These metrics are only meaningful when comparing predictions vs ground truth

4. **Data Shuffling**: Indices are shuffled before distribution to avoid sequential bias

## Workflow

1. Edit `config.json` with desired parameters
2. Run notebook - rotation angles are **auto-generated**
3. Ground truth mapping (`rotation_to_id`) is **auto-created**
4. Clustering metrics compare predictions vs true rotation labels
5. Results show how well clustering captured rotation-based heterogeneity

## Troubleshooting

**Problem**: ARI = 1.0 (perfect clustering)
- **Cause**: Either legitimate (rotations very separable) or data bias
- **Solution**: Ensure data indices are shuffled (already implemented)

**Problem**: Poor clustering performance
- **Cause**: Too many clusters or rotations too similar
- **Solution**: Reduce `num_rotation_clusters` or use larger angles

**Problem**: Training fails
- **Cause**: Too few clients per cluster
- **Solution**: Increase `num_clients` or decrease `num_rotation_clusters`
