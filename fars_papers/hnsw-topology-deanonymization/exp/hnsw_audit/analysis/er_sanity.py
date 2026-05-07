"""Erdos-Renyi random graph sanity check on SIFT10K.

Generates an ER random graph matching the average degree of the HNSW graph
(seed=42, M=32). Runs adjacency-only and degree-penalized geodesic
reconstruction, comparing Recall@10 against chance level k/(n-1).
"""

import json
import os
import pickle
import sys
import time

import faiss
import matplotlib
matplotlib.use("Agg")
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reconstruction.edge_weights import degree_penalized_weight, unit_weight
from reconstruction.geodesic_reconstruct import geodesic_reconstruct

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEED = 42
K = 10
N_LANDMARKS = 256
N_COMPONENTS = 32
ALPHA = 1.0


def load_data():
    vectors = np.load(os.path.join(OUTPUT_DIR, "vectors.npy"))
    knn_array = np.load(os.path.join(OUTPUT_DIR, "true_knn_k20.npy"))
    true_knn = {}
    for i in range(knn_array.shape[0]):
        true_knn[i] = knn_array[i, :K].tolist()

    with open(os.path.join(OUTPUT_DIR, f"adj_seed{SEED}.pkl"), "rb") as f:
        data = pickle.load(f)
    return vectors, true_knn, data["adjacency_list"]


def compute_avg_degree(adjacency_list):
    total = sum(len(nbrs) for nbrs in adjacency_list.values())
    return total / len(adjacency_list)


def er_graph_to_adjacency(G):
    adj = {}
    for node in range(G.number_of_nodes()):
        adj[node] = sorted(G.neighbors(node))
    return adj


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


def compute_recall(predicted, true, k):
    n = len(true)
    total = 0.0
    for i in range(n):
        true_set = set(true[i][:k])
        pred_set = set(predicted.get(i, [])[:k])
        total += len(true_set & pred_set) / k
    return total / n


def compute_adjacency_recall(adjacency_list, true_knn, k):
    n = len(true_knn)
    total = 0.0
    for i in range(n):
        true_set = set(true_knn[i][:k])
        nbr_set = set(adjacency_list.get(i, []))
        total += len(true_set & nbr_set) / k
    return total / n


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    vectors, true_knn, hnsw_adj = load_data()
    n = vectors.shape[0]

    avg_degree = compute_avg_degree(hnsw_adj)
    p = avg_degree / (n - 1)
    chance_recall = K / (n - 1)

    print(f"HNSW avg degree: {avg_degree:.2f}")
    print(f"ER p = {p:.6f}")
    print(f"Chance Recall@{K} = {chance_recall:.6f}")

    print("\nGenerating ER random graph...")
    t0 = time.time()
    G_er = nx.erdos_renyi_graph(n, p, seed=SEED)
    t_gen = time.time() - t0
    er_adj = er_graph_to_adjacency(G_er)
    er_avg_degree = compute_avg_degree(er_adj)
    print(f"ER graph generated in {t_gen:.2f}s, avg degree: {er_avg_degree:.2f}")

    print("\n--- Adjacency-only baseline on ER graph ---")
    adj_recall = compute_adjacency_recall(er_adj, true_knn, K)
    print(f"  Recall@{K} = {adj_recall:.6f}")

    print("\n--- Degree-penalized geodesic on ER graph ---")
    t0 = time.time()
    wm = degree_penalized_weight(er_adj, alpha=ALPHA)
    embedding = geodesic_reconstruct(
        er_adj,
        n_landmarks=N_LANDMARKS,
        n_components=N_COMPONENTS,
        weight_matrix=wm,
        seed=SEED,
    )
    t_recon = time.time() - t0

    pred_knn = knn_from_embeddings(embedding[:n], K)
    recon_recall = compute_recall(pred_knn, true_knn, K)
    print(f"  Recall@{K} = {recon_recall:.6f} (time: {t_recon:.2f}s)")

    results = {
        "experiment": "er_sanity_check",
        "dataset": "SIFT10K",
        "seed": SEED,
        "n": n,
        "k": K,
        "hnsw_avg_degree": round(avg_degree, 4),
        "er_p": round(p, 8),
        "er_avg_degree": round(er_avg_degree, 4),
        "chance_recall_at_k": round(chance_recall, 6),
        "er_adjacency_only_recall_at_10": round(adj_recall, 6),
        "er_degree_penalized_recall_at_10": round(recon_recall, 6),
        "reconstruction_params": {
            "alpha": ALPHA,
            "n_landmarks": N_LANDMARKS,
            "n_components": N_COMPONENTS,
        },
        "reconstruction_time_s": round(t_recon, 2),
        "er_generation_time_s": round(t_gen, 2),
    }

    results_path = os.path.join(RESULTS_DIR, "analysis_er_sanity.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    print(f"\n=== Summary ===")
    print(f"Chance Recall@{K}: {chance_recall:.6f}")
    print(f"ER Adjacency-only Recall@{K}: {adj_recall:.6f}")
    print(f"ER Degree-penalized Recall@{K}: {recon_recall:.6f}")
    er_adj_ratio = adj_recall / chance_recall if chance_recall > 0 else float("inf")
    er_recon_ratio = recon_recall / chance_recall if chance_recall > 0 else float("inf")
    print(f"ER adj / chance ratio: {er_adj_ratio:.2f}x")
    print(f"ER recon / chance ratio: {er_recon_ratio:.2f}x")


if __name__ == "__main__":
    main()
