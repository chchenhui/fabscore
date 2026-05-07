"""Evaluate unweighted geodesic + Landmark MDS baseline on SIFT10K.

For each of 3 HNSW seeds, loads the precomputed adjacency list, runs geodesic
reconstruction with unit edge weights (L=2000 landmarks, d=128 components), computes
kNN from reconstructed coordinates, and evaluates Recall@{10,20} and Spearman
correlation (50k sampled pairs). Saves results to results/sift10k_unweighted_geodesic.json.
"""

import json
import os
import pickle
import sys
import time

import faiss
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reconstruction.geodesic_reconstruct import geodesic_reconstruct

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEEDS = [42, 123, 456]
K_VALUES = [10, 20]
N_LANDMARKS = 2000
N_COMPONENTS = 128
N_SPEARMAN_PAIRS = 50_000


def unit_weight(u, v, adj):
    return 1.0


def load_true_knn(k: int) -> dict[int, list[int]]:
    knn_array = np.load(os.path.join(OUTPUT_DIR, "true_knn_k20.npy"))
    true_knn = {}
    for i in range(knn_array.shape[0]):
        true_knn[i] = knn_array[i, :k].tolist()
    return true_knn


def compute_recall_at_k(predicted_neighbors, true_neighbors, k):
    n = len(true_neighbors)
    total = 0.0
    for i in range(n):
        true_set = set(true_neighbors[i][:k])
        pred_set = set(predicted_neighbors.get(i, [])[:k])
        total += len(true_set & pred_set) / k
    return total / n


def knn_from_embeddings(embeddings: np.ndarray, k: int) -> dict[int, list[int]]:
    n, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    _, indices = index.search(embeddings.astype(np.float32), k + 1)
    knn = {}
    for i in range(n):
        nbrs = [int(j) for j in indices[i] if j != i][:k]
        knn[i] = nbrs
    return knn


def compute_spearman(adjacency_list, vectors, embedding, n_pairs, seed):
    n = vectors.shape[0]
    rng = np.random.default_rng(seed)
    pairs_i = rng.integers(0, n, size=n_pairs)
    pairs_j = rng.integers(0, n, size=n_pairs)
    mask = pairs_i != pairs_j
    pairs_i = pairs_i[mask]
    pairs_j = pairs_j[mask]
    if len(pairs_i) > n_pairs:
        pairs_i = pairs_i[:n_pairs]
        pairs_j = pairs_j[:n_pairs]

    true_dists = np.sqrt(np.sum((vectors[pairs_i] - vectors[pairs_j]) ** 2, axis=1))

    recon_dists = np.sqrt(np.sum((embedding[pairs_i] - embedding[pairs_j]) ** 2, axis=1))

    corr, _ = spearmanr(recon_dists, true_dists)
    return float(corr)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vectors = np.load(os.path.join(OUTPUT_DIR, "vectors.npy"))
    n = vectors.shape[0]
    print(f"Loaded {n} vectors of dim {vectors.shape[1]}")

    results = {
        "dataset": "SIFT10K",
        "method": "unweighted_geodesic_lmds",
        "params": {"n_landmarks": N_LANDMARKS, "n_components": N_COMPONENTS, "edge_weight": "unit (1.0)"},
        "seeds": SEEDS,
        "k_values": K_VALUES,
        "per_seed": {},
        "summary": {},
    }

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        pkl_path = os.path.join(OUTPUT_DIR, f"adj_seed{seed}.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        adjacency_list = data["adjacency_list"]
        print(f"  Loaded adjacency list with {len(adjacency_list)} nodes")

        t0 = time.time()
        embedding = geodesic_reconstruct(
            adjacency_list,
            n_landmarks=N_LANDMARKS,
            n_components=N_COMPONENTS,
            edge_weight_fn=unit_weight,
            seed=seed,
        )
        elapsed = time.time() - t0
        print(f"  Geodesic reconstruction: {elapsed:.1f}s, embedding shape {embedding.shape}")

        seed_results = {"reconstruction_time_s": round(elapsed, 2)}

        for k in K_VALUES:
            true_knn = load_true_knn(k)
            pred_knn = knn_from_embeddings(embedding[:n], k)
            recall = compute_recall_at_k(pred_knn, true_knn, k)
            seed_results[f"recall@{k}"] = round(recall, 6)
            print(f"  Recall@{k}: {recall:.4f}")

        spearman = compute_spearman(adjacency_list, vectors, embedding[:n], N_SPEARMAN_PAIRS, seed)
        seed_results["spearman_correlation"] = round(spearman, 6)
        print(f"  Spearman correlation: {spearman:.4f}")

        results["per_seed"][str(seed)] = seed_results

    for k in K_VALUES:
        key = f"recall@{k}"
        vals = [results["per_seed"][str(s)][key] for s in SEEDS]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        results["summary"][key] = {
            "mean": round(mean_val, 6),
            "std": round(std_val, 6),
            "per_seed": {str(s): results["per_seed"][str(s)][key] for s in SEEDS},
        }
        print(f"\nRecall@{k}: {mean_val:.4f} +/- {std_val:.4f}")

    spearman_vals = [results["per_seed"][str(s)]["spearman_correlation"] for s in SEEDS]
    results["summary"]["spearman_correlation"] = {
        "mean": round(float(np.mean(spearman_vals)), 6),
        "std": round(float(np.std(spearman_vals)), 6),
        "per_seed": {str(s): results["per_seed"][str(s)]["spearman_correlation"] for s in SEEDS},
    }
    print(f"Spearman: {np.mean(spearman_vals):.4f} +/- {np.std(spearman_vals):.4f}")

    results_path = os.path.join(RESULTS_DIR, "sift10k_unweighted_geodesic.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
