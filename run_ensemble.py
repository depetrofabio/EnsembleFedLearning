#!/usr/bin/env python3
"""
Ensemble Federated Learning Training Script

This script trains an ensemble federated learning model with hyperparameters
configurable via config.json file and/or command line arguments.
Command line arguments override config file values.

Example usage:
    # Using config file:
    python run_ensemble.py --config config.json
    
    # Override specific parameters:
    python run_ensemble.py --config config.json --num_clients 50 --lr 0.001
    
    # Without config file (use defaults):
    python run_ensemble.py --num_clients 100 --num_clusters 5
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

from training.ensemble_fl import EnsembleFedAvg
from data.loader import load_data_dirichlet, load_cifar10
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_config(config_path):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from {config_path}")
    return config


def merge_configs(file_config, cli_args):
    """
    Merge configurations from file and CLI arguments.
    CLI arguments override file configuration.
    """
    # Start with file config or empty dict
    config = file_config if file_config else {}
    
    # Override with CLI args (only if they were explicitly set)
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if value is not None and key != 'config':
            config[key] = value
    
    return config


def create_parser():
    """Create argument parser with all hyperparameters."""
    parser = argparse.ArgumentParser(
        description='Train Ensemble Federated Learning Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ensemble.py --config config.json
  python run_ensemble.py --num_clients 100 --num_clusters 10 --warmup_rounds 3
  python run_ensemble.py --config config.json --lr 0.001 --device cuda:1
        """
    )
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON configuration file')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store/load data')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha parameter for non-IID data distribution')
    
    # Federated Learning parameters
    parser.add_argument('--num_clients', type=int, default=100,
                        help='Total number of clients')
    parser.add_argument('--num_clusters', type=int, default=5,
                        help='Number of clusters for ensemble')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='resnet18',
                        choices=['resnet18'],
                        help='Base model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model weights')
    
    # Warmup parameters
    parser.add_argument('--warmup_rounds', type=int, default=1,
                        help='Number of warmup rounds for FedAvg')
    parser.add_argument('--warmup_local_epochs', type=int, default=2,
                        help='Number of local epochs during warmup')
    parser.add_argument('--use_fedavg_warmup', action='store_true', default=True,
                        help='Use FedAvg for warmup (default: True)')
    parser.add_argument('--no_fedavg_warmup', dest='use_fedavg_warmup', 
                        action='store_false',
                        help='Disable FedAvg warmup (use single client finetuning)')
    parser.add_argument('--use_weight_diff', action='store_true',
                        help='Use weight differences for clustering instead of gradients')
    
    # Clustering parameters
    parser.add_argument('--clustering_method', type=str, default='kmeans',
                        choices=['kmeans', 'features'],
                        help='Clustering method for clients')
    parser.add_argument('--num_feature_samples', type=int, default=100,
                        help='Number of samples per client for feature-based clustering')
    
    # Training parameters
    parser.add_argument('--ensemble_rounds', type=int, default=50,
                        help='Number of ensemble training rounds')
    parser.add_argument('--ensemble_local_epochs', type=int, default=1,
                        help='Number of local epochs during ensemble training')
    parser.add_argument('--client_fraction', type=float, default=0.1,
                        help='Fraction of clients to participate in each round')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'cuda:0', 'cuda:1'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results and models')
    parser.add_argument('--save_warmup_model', action='store_true',
                        help='Save model after warmup phase')
    parser.add_argument('--save_final_model', action='store_true', default=True,
                        help='Save final ensemble model')
    
    return parser


def perform_clustering(ensemble_fl, config):
    """
    Perform client clustering based on configuration.
    
    Args:
        ensemble_fl: EnsembleFedAvg instance
        config: Configuration dictionary
        
    Returns:
        client_clusters: Dictionary mapping client_idx -> cluster_idx
    """
    method = config.get('clustering_method', 'kmeans')
    num_clusters = config['num_clusters']
    
    print(f"\nPerforming {method} clustering into {num_clusters} clusters...")
    
    if method == 'kmeans':
        # Use gradient-based clustering
        client_gradients = ensemble_fl.get_client_gradients(average_across_rounds=True)
        gradient_matrix = np.array([client_gradients[i] for i in range(ensemble_fl.num_clients)])
        
        # Normalize gradients
        from sklearn.preprocessing import normalize
        gradient_matrix_norm = normalize(gradient_matrix, norm='l2')
        
        # K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=config['seed'], n_init=10)
        cluster_labels = kmeans.fit_predict(gradient_matrix_norm)
        
        # Compute silhouette score
        if num_clusters > 1:
            silhouette = silhouette_score(gradient_matrix_norm, cluster_labels)
            print(f"Silhouette Score: {silhouette:.4f}")
        
    elif method == 'features':
        # Use feature-based clustering
        num_samples = config.get('num_feature_samples', 100)
        client_features = ensemble_fl.compute_feature_representations(num_samples=num_samples)
        
        # Normalize features
        from sklearn.preprocessing import normalize
        client_features_norm = normalize(client_features, norm='l2')
        
        # K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=config['seed'], n_init=10)
        cluster_labels = kmeans.fit_predict(client_features_norm)
        
        # Compute silhouette score
        if num_clusters > 1:
            silhouette = silhouette_score(client_features_norm, cluster_labels)
            print(f"Silhouette Score: {silhouette:.4f}")
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Create client_clusters dictionary
    client_clusters = {i: int(cluster_labels[i]) for i in range(ensemble_fl.num_clients)}
    
    # Print cluster distribution
    cluster_counts = {}
    for cluster_id in cluster_labels:
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    print("\nCluster distribution:")
    for cluster_id in sorted(cluster_counts.keys()):
        print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} clients")
    
    return client_clusters


def save_results(ensemble_fl, config, history):
    """Save training results and models."""
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    # Save configuration
    config_path = output_dir / 'config_used.json'
    with open(config_path, 'w') as f:
        # Convert non-serializable objects to strings
        config_copy = config.copy()
        config_copy['device'] = str(config_copy['device'])
        json.dump(config_copy, f, indent=2)
    print(f"Saved configuration to {config_path}")
    
    # Save final model if requested
    if config.get('save_final_model', True):
        model_dir = output_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        
        # Save cluster models
        for i, cluster_model in enumerate(ensemble_fl.cluster_models):
            model_path = model_dir / f'cluster_model_{i}.pth'
            torch.save(cluster_model.state_dict(), model_path)
        
        # Save classifier
        classifier_path = model_dir / 'ensemble_classifier.pth'
        torch.save(ensemble_fl.ensemble_classifier.state_dict(), classifier_path)
        
        # Save cluster assignments
        clusters_path = model_dir / 'client_clusters.json'
        with open(clusters_path, 'w') as f:
            json.dump(ensemble_fl.client_clusters, f, indent=2)
        
        print(f"Saved models to {model_dir}/")


def main():
    """Main training function."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Load config file if specified
    file_config = load_config(args.config) if args.config else {}
    
    # Merge configs (CLI overrides file)
    config = merge_configs(file_config, args)
    
    # Set default values for any missing parameters
    defaults = {
        'dataset': 'cifar10',
        'data_dir': './data',
        'alpha': 0.5,
        'num_clients': 100,
        'num_clusters': 5,
        'batch_size': 64,
        'lr': 0.01,
        'model_name': 'resnet18',
        'pretrained': False,
        'warmup_rounds': 1,
        'warmup_local_epochs': 2,
        'use_fedavg_warmup': False,
        'use_weight_diff': True,
        'clustering_method': 'kmeans',
        'num_feature_samples': 100,
        'ensemble_rounds': 50,
        'ensemble_local_epochs': 1,
        'client_fraction': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'output_dir': './results',
        'save_warmup_model': False,
        'save_final_model': True,
    }
    
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
    
    # Setup device
    if isinstance(config['device'], str):
        if 'cuda' in config['device'] and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            config['device'] = torch.device('cpu')
        else:
            config['device'] = torch.device(config['device'])
    
    # Set random seeds for reproducibility
    import random
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Print configuration
    print("\n" + "="*60)
    print("ENSEMBLE FEDERATED LEARNING TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Load data
    print("Loading dataset...")
    if config['dataset'] == 'cifar10':
        train_subsets, test_set, val_set = load_data_dirichlet(
            num_clients=config['num_clients'],
            alpha=config['alpha'],
            batch_size=config['batch_size'],
            seed=config['seed']
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    print(f"Dataset loaded: {config['num_clients']} clients")
    
    # Initialize Ensemble FL
    print("\nInitializing Ensemble Federated Learning...")
    ensemble_fl = EnsembleFedAvg(
        train_subsets=train_subsets,
        test_set=test_set,
        num_clients=config['num_clients'],
        device=config['device'],
        model_name=config['model_name'],
        pretrained=config['pretrained'],
        num_clusters=config['num_clusters'],
        batch_size=config['batch_size'],
        lr=config['lr'],
        seed=config['seed']
    )
    
    # Warmup phase
    print("\n" + "="*60)
    print("WARMUP PHASE")
    print("="*60)
    ensemble_fl.run_warmup(
        use_fedavg=config['use_fedavg_warmup'],
        rounds=config['warmup_rounds'],
        local_epochs=config['warmup_local_epochs'],
        use_weight_diff=config['use_weight_diff']
    )
    
    # Save warmup model if requested
    if config['save_warmup_model']:
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        warmup_model_path = output_dir / 'warmup_model.pth'
        ensemble_fl.save_warmup_model(str(warmup_model_path))
    
    # Clustering phase
    print("\n" + "="*60)
    print("CLUSTERING PHASE")
    print("="*60)
    client_clusters = perform_clustering(ensemble_fl, config)
    ensemble_fl.client_clusters = client_clusters
    
    # Initialize ensemble
    print("\nInitializing ensemble models...")
    ensemble_fl._initialize_ensemble()
    
    # Training phase
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING PHASE")
    print("="*60)
    
    history = {
        'rounds': [],
        'losses': [],
        'accuracies': []
    }
    
    for round_num in range(1, config['ensemble_rounds'] + 1):
        loss, acc = ensemble_fl.train_ensemble_round(
            round_num=round_num,
            fraction=config['client_fraction'],
            local_epochs=config['ensemble_local_epochs']
        )
        
        history['rounds'].append(round_num)
        history['losses'].append(float(loss))
        history['accuracies'].append(float(acc))
        
        # Print progress every 10 rounds
        if round_num % 10 == 0:
            print(f"\nProgress: Round {round_num}/{config['ensemble_rounds']}")
            print(f"  Current Accuracy: {acc:.4f}")
            print(f"  Best Accuracy: {max(history['accuracies']):.4f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    final_loss, final_acc = ensemble_fl.evaluate_ensemble()
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Final Test Accuracy: {final_acc:.4f}")
    print(f"Best Test Accuracy: {max(history['accuracies']):.4f}")
    
    # Save results
    save_results(ensemble_fl, config, history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
