"""Alpha sensitivity ablation on SIFT10K (seed=42).

Sweeps hub-penalty coefficient alpha in {0.0, 0.25, 0.5, 1.0, 2.0, 4.0}
with L=256 landmarks and d=32 components. alpha=0.0 is equivalent to
unweighted geodesic. Saves JSON results and a line plot (Recall@10 vs alpha).
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reconstruction.edge_weights import degree_penalized_weight, unit_weight
from reconstruction.geodesic_reconstruct import geodesic_reconstruct

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SEED = 42
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
N_LANDMARKS = 256
N_COMPONENTS = 32
K = 10
N_SPEARMAN_PAIRS = 50_000


def load_data():
    vectors = np.load(os.path.join(OUTPUT_DIR, "vectors.npy"))
    knn_array = np.load(os.path.join(OUTPUT_DIR, "true_knn_k20.npy"))
    true_knn = {}
    for i in range(knn_array.shape[0]):
        true_knn[i] = knn_array[i, :K].tolist()

    with open(os.path.join(OUTPUT_DIR, f"adj_seed{SEED}.pkl"), "rb") as f:
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


def compute_recall(predicted, true, k):
    n = len(true)
    total = 0.0
    for i in range(n):
        true_set = set(true[i][:k])
        pred_set = set(predicted.get(i, [])[:k])
        total += len(true_set & pred_set) / k
    return total / n


def compute_spearman(vectors, embedding, n_pairs, seed):
    n = vectors.shape[0]
    rng = np.random.default_rng(seed)
    pairs_i = rng.integers(0, n, size=n_pairs)
    pairs_j = rng.integers(0, n, size=n_pairs)
    mask = pairs_i != pairs_j
    pairs_i, pairs_j = pairs_i[mask][:n_pairs], pairs_j[mask][:n_pairs]
    true_dists = np.sqrt(np.sum((vectors[pairs_i] - vectors[pairs_j]) ** 2, axis=1))
    recon_dists = np.sqrt(np.sum((embedding[pairs_i] - embedding[pairs_j]) ** 2, axis=1))
    corr, _ = spearmanr(recon_dists, true_dists)
    return float(corr)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    vectors, true_knn, adjacency_list = load_data()
    n = vectors.shape[0]
    print(f"Loaded {n} vectors, dim={vectors.shape[1]}")

    results = {
        "experiment": "alpha_sensitivity",
        "dataset": "SIFT10K",
        "seed": SEED,
        "n_landmarks": N_LANDMARKS,
        "n_components": N_COMPONENTS,
        "k": K,
        "alpha_values": ALPHAS,
        "runs": [],
    }

    for alpha in ALPHAS:
        print(f"\n--- alpha={alpha} ---")
        t0 = time.time()
        if alpha == 0.0:
            wm = unit_weight(adjacency_list)
        else:
            wm = degree_penalized_weight(adjacency_list, alpha=alpha)
        t_weight = time.time() - t0

        t0 = time.time()
        embedding = geodesic_reconstruct(
            adjacency_list,
            n_landmarks=N_LANDMARKS,
            n_components=N_COMPONENTS,
            weight_matrix=wm,
            seed=SEED,
        )
        t_recon = time.time() - t0
        total_time = t_weight + t_recon

        pred_knn = knn_from_embeddings(embedding[:n], K)
        recall = compute_recall(pred_knn, true_knn, K)
        spearman = compute_spearman(vectors, embedding[:n], N_SPEARMAN_PAIRS, SEED)

        run = {
            "alpha": alpha,
            "recall@10": round(recall, 6),
            "spearman": round(spearman, 6),
            "time_s": round(total_time, 2),
            "weight_time_s": round(t_weight, 2),
            "reconstruct_time_s": round(t_recon, 2),
        }
        results["runs"].append(run)
        print(f"  Recall@10={recall:.4f}, Spearman={spearman:.4f}, Time={total_time:.2f}s")

    results_path = os.path.join(RESULTS_DIR, "analysis_alpha_sensitivity.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    alphas = [r["alpha"] for r in results["runs"]]
    recalls = [r["recall@10"] for r in results["runs"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, recalls, "o-", color="#2c7bb6", linewidth=2, markersize=7)
    ax.set_xlabel(r"Hub-Penalty Coefficient $\alpha$", fontsize=12)
    ax.set_ylabel("Recall@10", fontsize=12)
    ax.set_title("Alpha Sensitivity (SIFT10K, L=256, d=32)", fontsize=13)
    ax.grid(True, alpha=0.3)
    for a, r in zip(alphas, recalls):
        ax.annotate(f"{r:.3f}", (a, r), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "alpha_sensitivity.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
