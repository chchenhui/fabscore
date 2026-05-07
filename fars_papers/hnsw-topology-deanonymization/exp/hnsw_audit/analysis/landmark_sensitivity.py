"""Landmark count sensitivity ablation on SIFT10K (seed=42).

Sweeps landmark count L in {64, 128, 256, 512, 1024} with alpha=1.0 and
d=32 components. Records Recall@10, Spearman, and wall-clock time for the
geodesic computation (L Dijkstra runs). Saves JSON results and a dual-axis
line plot (Recall@10 and time vs L).
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

from reconstruction.edge_weights import degree_penalized_weight
from reconstruction.geodesic_reconstruct import geodesic_reconstruct

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SEED = 42
LANDMARKS = [64, 128, 256, 512, 1024]
ALPHA = 1.0
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

    wm = degree_penalized_weight(adjacency_list, alpha=ALPHA)
    print(f"Weight matrix: nnz={wm.nnz}, alpha={ALPHA}")

    results = {
        "experiment": "landmark_sensitivity",
        "dataset": "SIFT10K",
        "seed": SEED,
        "alpha": ALPHA,
        "n_components": N_COMPONENTS,
        "k": K,
        "landmark_values": LANDMARKS,
        "runs": [],
    }

    for L in LANDMARKS:
        print(f"\n--- L={L} ---")
        t0 = time.time()
        embedding = geodesic_reconstruct(
            adjacency_list,
            n_landmarks=L,
            n_components=N_COMPONENTS,
            weight_matrix=wm,
            seed=SEED,
        )
        t_recon = time.time() - t0

        pred_knn = knn_from_embeddings(embedding[:n], K)
        recall = compute_recall(pred_knn, true_knn, K)
        spearman = compute_spearman(vectors, embedding[:n], N_SPEARMAN_PAIRS, SEED)

        run = {
            "n_landmarks": L,
            "recall@10": round(recall, 6),
            "spearman": round(spearman, 6),
            "time_s": round(t_recon, 2),
        }
        results["runs"].append(run)
        print(f"  Recall@10={recall:.4f}, Spearman={spearman:.4f}, Time={t_recon:.2f}s")

    results_path = os.path.join(RESULTS_DIR, "analysis_landmark_sensitivity.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    ls = [r["n_landmarks"] for r in results["runs"]]
    recalls = [r["recall@10"] for r in results["runs"]]
    times = [r["time_s"] for r in results["runs"]]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color1 = "#2c7bb6"
    ax1.plot(ls, recalls, "o-", color=color1, linewidth=2, markersize=7, label="Recall@10")
    ax1.set_xlabel("Number of Landmarks L", fontsize=12)
    ax1.set_ylabel("Recall@10", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ls)
    ax1.set_xticklabels([str(l) for l in ls])

    ax2 = ax1.twinx()
    color2 = "#d7191c"
    ax2.plot(ls, times, "s--", color=color2, linewidth=2, markersize=7, label="Time (s)")
    ax2.set_ylabel("Computation Time (s)", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.set_title(r"Landmark Sensitivity (SIFT10K, $\alpha$=1.0, d=32)", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "landmark_sensitivity.pdf")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
