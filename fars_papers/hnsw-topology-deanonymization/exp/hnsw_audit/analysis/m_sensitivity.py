"""M sensitivity analysis: denser HNSW with M in {16, 32, 64} on SIFT10K.

Builds HNSW indices with varying M (efConstruction=64, seed=42), runs
adjacency-only and degree-penalized geodesic reconstruction (alpha=1.0,
L=256, d=32), and produces a grouped bar chart of Recall@10 vs M.
"""

import json
import os
import pickle
import sys
import time

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph.hnsw_builder import build_hnsw_index, extract_layer0_adjacency
from reconstruction.edge_weights import degree_penalized_weight
from reconstruction.geodesic_reconstruct import geodesic_reconstruct

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SEED = 42
K = 10
N_LANDMARKS = 256
N_COMPONENTS = 32
ALPHA = 1.0
EF_CONSTRUCTION = 64
M_VALUES = [16, 32, 64]


def load_data():
    vectors = np.load(os.path.join(OUTPUT_DIR, "vectors.npy"))
    knn_array = np.load(os.path.join(OUTPUT_DIR, "true_knn_k20.npy"))
    true_knn = {}
    for i in range(knn_array.shape[0]):
        true_knn[i] = knn_array[i, :K].tolist()
    return vectors, true_knn


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


def remap_adjacency(adjacency_list, perm):
    remapped = {}
    inv_perm = {int(perm[i]): i for i in range(len(perm))}
    for internal_id, nbrs in adjacency_list.items():
        orig_id = int(perm[internal_id])
        remapped[orig_id] = sorted([int(perm[v]) for v in nbrs])
    return remapped


def build_and_extract(vectors, M, efConstruction, seed):
    print(f"  Building HNSW index M={M}, efConstruction={efConstruction}...")
    t0 = time.time()
    index, perm = build_hnsw_index(vectors, M=M, efConstruction=efConstruction, seed=seed)
    t_build = time.time() - t0
    print(f"  Index built in {t_build:.2f}s")

    adj_internal, degrees = extract_layer0_adjacency(index)
    adj = remap_adjacency(adj_internal, perm)
    avg_deg = sum(len(v) for v in adj.values()) / len(adj)
    print(f"  Avg degree: {avg_deg:.2f}")
    return adj, avg_deg


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    vectors, true_knn = load_data()
    n = vectors.shape[0]
    print(f"Loaded {n} vectors, dim={vectors.shape[1]}")

    results = {
        "experiment": "m_sensitivity",
        "dataset": "SIFT10K",
        "seed": SEED,
        "efConstruction": EF_CONSTRUCTION,
        "n_landmarks": N_LANDMARKS,
        "n_components": N_COMPONENTS,
        "alpha": ALPHA,
        "k": K,
        "m_values": M_VALUES,
        "runs": [],
    }

    for M in M_VALUES:
        print(f"\n=== M={M} ===")
        adj, avg_deg = build_and_extract(vectors, M, EF_CONSTRUCTION, SEED)

        print("  Running adjacency-only baseline...")
        adj_recall = compute_adjacency_recall(adj, true_knn, K)
        print(f"  Adjacency-only Recall@{K} = {adj_recall:.6f}")

        print("  Running degree-penalized geodesic...")
        t0 = time.time()
        wm = degree_penalized_weight(adj, alpha=ALPHA)
        embedding = geodesic_reconstruct(
            adj,
            n_landmarks=N_LANDMARKS,
            n_components=N_COMPONENTS,
            weight_matrix=wm,
            seed=SEED,
        )
        t_recon = time.time() - t0

        pred_knn = knn_from_embeddings(embedding[:n], K)
        recon_recall = compute_recall(pred_knn, true_knn, K)
        print(f"  Degree-penalized Recall@{K} = {recon_recall:.6f} (time: {t_recon:.2f}s)")

        run = {
            "M": M,
            "avg_degree": round(avg_deg, 4),
            "adjacency_only_recall_at_10": round(adj_recall, 6),
            "degree_penalized_recall_at_10": round(recon_recall, 6),
            "reconstruction_time_s": round(t_recon, 2),
        }
        results["runs"].append(run)

    results_path = os.path.join(RESULTS_DIR, "analysis_m_sensitivity.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    ms = [r["M"] for r in results["runs"]]
    adj_recalls = [r["adjacency_only_recall_at_10"] for r in results["runs"]]
    recon_recalls = [r["degree_penalized_recall_at_10"] for r in results["runs"]]

    x = np.arange(len(ms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, adj_recalls, width, label="Adjacency-only",
                   color="#2c7bb6", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, recon_recalls, width,
                   label=r"Degree-penalized ($\alpha$=1.0)",
                   color="#d7191c", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("HNSW Parameter M", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Recall@10 vs HNSW M (SIFT10K, L=256, d=32)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in ms])
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "m_sensitivity.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
