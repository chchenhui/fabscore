"""Evaluate the adjacency-only baseline on SIFT10K across 3 seeds and k values.

Loads precomputed adjacency lists and true kNN, computes Recall@k for
k in {5, 10, 20}, and saves results to results/sift10k_adjacency_only.json.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.adjacency_baseline import adjacency_only_recall

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEEDS = [42, 123, 456]
K_VALUES = [5, 10, 20]


def load_true_knn(k: int) -> dict[int, list[int]]:
    knn_array = np.load(os.path.join(OUTPUT_DIR, "true_knn_k20.npy"))
    true_knn = {}
    for i in range(knn_array.shape[0]):
        true_knn[i] = knn_array[i, :k].tolist()
    return true_knn


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {"dataset": "SIFT10K", "method": "adjacency_only", "seeds": SEEDS, "k_values": K_VALUES, "per_seed": {}, "summary": {}}

    for seed in SEEDS:
        pkl_path = os.path.join(OUTPUT_DIR, f"adj_seed{seed}.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        adjacency_list = data["adjacency_list"]

        seed_results = {}
        for k in K_VALUES:
            true_knn = load_true_knn(k)
            recall = adjacency_only_recall(adjacency_list, true_knn, k)
            seed_results[f"recall@{k}"] = round(recall, 6)
            print(f"  Seed {seed}, Recall@{k}: {recall:.4f}")

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
        print(f"  Recall@{k}: {mean_val:.4f} +/- {std_val:.4f}")

    results_path = os.path.join(RESULTS_DIR, "sift10k_adjacency_only.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
