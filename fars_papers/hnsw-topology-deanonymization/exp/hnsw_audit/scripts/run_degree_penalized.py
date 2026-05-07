"""Evaluate degree-penalized geodesic + Landmark MDS on SIFT10K.

For each of 3 HNSW seeds, loads the precomputed adjacency list, computes
degree-penalized edge weights with alpha=3.0, runs geodesic reconstruction
(L=2000 landmarks, d=128), evaluates Recall@{10,20} and Spearman correlation.
Saves results to results/sift10k_degree_penalized.json.
"""

import json
import os
import pickle
import sys
import time

import faiss
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reconstruction.edge_weights import degree_penalized_weight
from reconstruction.geodesic_reconstruct import geodesic_reconstruct

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEEDS = [42, 123, 456]
K_VALUES = [10, 20]
N_LANDMARKS = 2000
N_COMPONENTS = 128
N_SPEARMAN_PAIRS = 50_000
ALPHA = 3.0


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
        "method": "degree_penalized_geodesic_lmds",
        "params": {
            "n_landmarks": N_LANDMARKS,
            "n_components": N_COMPONENTS,
            "alpha": ALPHA,
            "edge_weight": "1 + alpha*(log(1+deg(u))+log(1+deg(v)))/2",
        },
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
        weight_matrix = degree_penalized_weight(adjacency_list, alpha=ALPHA)
        t_weight = time.time() - t0
        print(f"  Weight computation: {t_weight:.2f}s, nnz={weight_matrix.nnz}")

        t0 = time.time()
        embedding = geodesic_reconstruct(
            adjacency_list,
            n_landmarks=N_LANDMARKS,
            n_components=N_COMPONENTS,
            weight_matrix=weight_matrix,
            seed=seed,
        )
        elapsed = time.time() - t0
        print(f"  Geodesic reconstruction: {elapsed:.1f}s, embedding shape {embedding.shape}")

        seed_results = {
            "weight_computation_time_s": round(t_weight, 2),
            "reconstruction_time_s": round(elapsed, 2),
        }

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

    results_path = os.path.join(RESULTS_DIR, "sift10k_degree_penalized.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
