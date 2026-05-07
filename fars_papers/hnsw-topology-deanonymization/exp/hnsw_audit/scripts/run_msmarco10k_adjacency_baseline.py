"""Evaluate adjacency-only baseline on MSMARCO-10K across 3 seeds and k values.

Loads precomputed adjacency lists and true kNN from outputs/msmarco10k/,
computes Recall@k for k in {5, 10, 20}, saves to results/msmarco10k_adjacency_only.json.
"""

import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.adjacency_baseline import adjacency_only_recall

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "msmarco10k")
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

    results = {
        "dataset": "MSMARCO-10K",
        "method": "adjacency_only",
        "n_vectors": 10000,
        "dim": 768,
        "hnsw_M": 32,
        "hnsw_efConstruction": 64,
        "seeds": SEEDS,
        "k_values": K_VALUES,
        "per_seed": {},
        "summary": {},
    }

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

        deg_vals = [len(adjacency_list[i]) for i in range(len(adjacency_list))]
        seed_results["avg_degree"] = round(float(np.mean(deg_vals)), 2)
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

    avg_degrees = [results["per_seed"][str(s)]["avg_degree"] for s in SEEDS]
    results["summary"]["avg_degree"] = {
        "mean": round(float(np.mean(avg_degrees)), 2),
        "std": round(float(np.std(avg_degrees)), 2),
    }

    results_path = os.path.join(RESULTS_DIR, "msmarco10k_adjacency_only.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
