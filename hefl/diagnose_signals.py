"""
Fast standalone check of the clustering signals.

``act_stats`` needs no local training at all - it is a forward pass through a
frozen backbone - so its quality can be measured in seconds, long before
committing to a full federated run.  Use this to sanity-check that the feature
axis is recoverable in your setting before spending GPU hours on it.

    python -m hefl.diagnose_signals --num_clients 24 --alpha 0.5
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader

from .clustering import activation_statistics, cluster_clients, label_axis_ground_truth
from .datasets import build_federated_data
from .models import CifarBackbone
from .utils import pick_device, set_seed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_clients", type=int, default=24)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--num_clusters", type=int, default=4)
    p.add_argument("--rotations", type=str, default="0,1,2,3")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--act_batches", type=int, default=8)
    p.add_argument("--grid", type=int, default=2, help="spatial pooling grid; 1 destroys the rotation signal")
    p.add_argument("--pca_dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_root", type=str, default="./data_cache")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device("auto")
    rotations = [int(r) for r in args.rotations.split(",")]

    data = build_federated_data(
        root=args.data_root, num_clients=args.num_clients, alpha=args.alpha,
        rotation_groups=rotations, seed=args.seed, subsample=args.subsample,
    )
    backbone = CifarBackbone("gn").to(device).eval()

    feats = []
    for ds in data.client_sets:
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        feats.append(
            activation_statistics(backbone, loader, device,
                                  max_batches=args.act_batches, grid=args.grid)
        )
    matrix = np.asarray(feats)

    out = cluster_clients(matrix, args.num_clusters, seed=args.seed, pca_dim=args.pca_dim)
    label_truth = label_axis_ground_truth(data.client_label_hist, args.num_clusters, args.seed)

    print(f"\nact_stats  dim={matrix.shape[1]}  clients={args.num_clients}  "
          f"rotations={rotations}  alpha={args.alpha}")
    print(f"  ARI vs rotation : {adjusted_rand_score(data.client_rotation, out['labels']):.4f}")
    print(f"  ARI vs label    : {adjusted_rand_score(label_truth, out['labels']):.4f}")
    print(f"  silhouette      : {out['silhouette']:.4f}")
    print(f"  cluster sizes   : {np.bincount(out['labels'], minlength=args.num_clusters).tolist()}")
    print(f"  true rotation   : {data.client_rotation.tolist()}")
    print(f"  assigned cluster: {out['labels'].tolist()}")


if __name__ == "__main__":
    main()
