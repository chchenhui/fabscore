"""Train MiniBatchKMeans codebook on HuBERT last_hidden_state features from BIWI training audio.

Extracts HuBERT-base-ls960 features (768-dim, 50 Hz) from all training sequences,
fits K=200 clusters, verifies all clusters are non-empty, and saves the model.
"""
import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import HubertModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="scaffoldswap/data/biwi/processed/train.pt")
    parser.add_argument("--hubert_model", default="facebook/hubert-base-ls960")
    parser.add_argument("--cache_dir", default="pretrained_models")
    parser.add_argument("--n_clusters", type=int, default=200)
    parser.add_argument("--output_path", default="pretrained_models/hubert_kmeans_biwi_K200.pkl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading HuBERT from {args.hubert_model} (cache: {args.cache_dir})")
    hubert = HubertModel.from_pretrained(args.hubert_model, cache_dir=args.cache_dir)
    hubert = hubert.to(device)
    hubert.eval()
    for p in hubert.parameters():
        p.requires_grad = False

    print(f"Loading training data from {args.data_path}")
    train_data = torch.load(args.data_path, weights_only=False)
    print(f"  {len(train_data)} training sequences")

    all_features = []
    for i, sample in enumerate(train_data):
        audio = sample["audio"].unsqueeze(0).to(device)  # (1, n_samples)
        with torch.no_grad():
            out = hubert(audio).last_hidden_state  # (1, T, 768)
        feats = out.squeeze(0).cpu().numpy()  # (T, 768)
        all_features.append(feats)
        if (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(train_data)} sequences")

    all_features = np.concatenate(all_features, axis=0)  # (N_total_frames, 768)
    print(f"Total feature frames: {all_features.shape[0]}, dim: {all_features.shape[1]}")

    print(f"Fitting MiniBatchKMeans with K={args.n_clusters}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=4096,
        max_iter=300,
        random_state=42,
        n_init=3,
    )
    kmeans.fit(all_features)

    labels = kmeans.labels_
    cluster_counts = np.bincount(labels, minlength=args.n_clusters)
    n_empty = (cluster_counts == 0).sum()
    print(f"Cluster utilization: {args.n_clusters - n_empty}/{args.n_clusters} non-empty")
    if n_empty > 0:
        print(f"WARNING: {n_empty} empty clusters detected!")
        print(f"  Empty cluster IDs: {np.where(cluster_counts == 0)[0].tolist()}")
    else:
        print("All clusters are non-empty.")

    print(f"Cluster count stats: min={cluster_counts.min()}, max={cluster_counts.max()}, "
          f"mean={cluster_counts.mean():.1f}, median={np.median(cluster_counts):.1f}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"Saved k-means model to {args.output_path}")


if __name__ == "__main__":
    main()
