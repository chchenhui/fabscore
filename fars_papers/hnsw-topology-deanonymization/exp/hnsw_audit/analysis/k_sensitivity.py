"""Recall@k sensitivity analysis for k in {5, 10, 20}.

Evaluates adjacency-only, unweighted geodesic + LMDS, and degree-penalized
geodesic + LMDS on both SIFT10K and MSMARCO-10K using seed=42. Saves a
combined table to results/analysis_k_sensitivity.json.
"""

import json
import os
import pickle
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reconstruction.edge_weights import degree_penalized_weight, unit_weight
from reconstruction.geodesic_reconstruct import geodesic_reconstruct
from evaluation.metrics import compute_recall_at_k

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEED = 42
K_VALUES = [5, 10, 20]

DATASET_CONFIGS = {
    "SIFT10K": {
        "output_dir": os.path.join(BASE_DIR, "outputs", "sift10k"),
        "n_landmarks": 2000,
        "n_components": 128,
        "alpha": 3.0,
    },
    "MSMARCO-10K": {
        "output_dir": os.path.join(BASE_DIR, "outputs", "msmarco10k"),
        "n_landmarks": 3000,
        "n_components": 128,
        "alpha": 4.0,
    },
}


def load_dataset(cfg):
    vectors = np.load(os.path.join(cfg["output_dir"], "vectors.npy"))
    knn_array = np.load(os.path.join(cfg["output_dir"], "true_knn_k20.npy"))
    true_knn = {}
    for i in range(knn_array.shape[0]):
        true_knn[i] = knn_array[i].tolist()
    with open(os.path.join(cfg["output_dir"], f"adj_seed{SEED}.pkl"), "rb") as f:
        data = pickle.load(f)
    return vectors, true_knn, data["adjacency_list"]


def knn_from_embeddings(embeddings, k):
    n, d = embeddings.shape
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    _, indices = index.search(embeddings.astype(np.float32), k + 1)
    knn = {}
    for i in range(n):
        nbrs = [int(j) for j in indices[i] if j != i][:k]
        knn[i] = nbrs
    return knn


def adjacency_recall(adjacency_list, true_knn, k):
    predicted = {}
    for node_id in range(len(true_knn)):
        nbrs = adjacency_list.get(node_id, [])
        predicted[node_id] = nbrs[:k]
    return compute_recall_at_k(predicted, true_knn, k)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "experiment": "k_sensitivity_analysis",
        "seed": SEED,
        "k_values": K_VALUES,
        "table": [],
    }

    for dataset_name, cfg in DATASET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        vectors, true_knn, adjacency_list = load_dataset(cfg)
        n = vectors.shape[0]
        print(f"Loaded {n} vectors, dim={vectors.shape[1]}")

        row_adj = {"method": "adjacency_only", "dataset": dataset_name}
        for k in K_VALUES:
            recall = adjacency_recall(adjacency_list, true_knn, k)
            row_adj[f"recall@{k}"] = round(recall, 6)
            print(f"  Adjacency-only Recall@{k} = {recall:.4f}")
        results["table"].append(row_adj)

        print(f"\n  Running unweighted geodesic (L={cfg['n_landmarks']}, d={cfg['n_components']})...")
        t0 = time.time()
        wm_unit = unit_weight(adjacency_list)
        emb_unit = geodesic_reconstruct(
            adjacency_list,
            n_landmarks=cfg["n_landmarks"],
            n_components=cfg["n_components"],
            weight_matrix=wm_unit,
            seed=SEED,
        )
        print(f"  Reconstruction took {time.time()-t0:.1f}s")

        max_k = max(K_VALUES)
        pred_unit = knn_from_embeddings(emb_unit[:n], max_k)
        row_unit = {"method": "unweighted_geodesic", "dataset": dataset_name}
        for k in K_VALUES:
            pred_k = {i: pred_unit[i][:k] for i in pred_unit}
            true_k = {i: true_knn[i][:k] for i in true_knn}
            recall = compute_recall_at_k(pred_k, true_k, k)
            row_unit[f"recall@{k}"] = round(recall, 6)
            print(f"  Unweighted geodesic Recall@{k} = {recall:.4f}")
        results["table"].append(row_unit)

        print(f"\n  Running degree-penalized geodesic (alpha={cfg['alpha']}, L={cfg['n_landmarks']}, d={cfg['n_components']})...")
        t0 = time.time()
        wm_pen = degree_penalized_weight(adjacency_list, alpha=cfg["alpha"])
        emb_pen = geodesic_reconstruct(
            adjacency_list,
            n_landmarks=cfg["n_landmarks"],
            n_components=cfg["n_components"],
            weight_matrix=wm_pen,
            seed=SEED,
        )
        print(f"  Reconstruction took {time.time()-t0:.1f}s")

        pred_pen = knn_from_embeddings(emb_pen[:n], max_k)
        row_pen = {"method": "degree_penalized_geodesic", "dataset": dataset_name}
        for k in K_VALUES:
            pred_k = {i: pred_pen[i][:k] for i in pred_pen}
            true_k = {i: true_knn[i][:k] for i in true_knn}
            recall = compute_recall_at_k(pred_k, true_k, k)
            row_pen[f"recall@{k}"] = round(recall, 6)
            print(f"  Degree-penalized geodesic Recall@{k} = {recall:.4f}")
        results["table"].append(row_pen)

    out_path = os.path.join(RESULTS_DIR, "analysis_k_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
