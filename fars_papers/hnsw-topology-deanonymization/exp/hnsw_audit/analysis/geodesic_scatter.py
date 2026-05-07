"""Scatter plot: graph-geodesic distances vs true Euclidean distances.

For SIFT10K (seed=42), computes all-pairs shortest paths for both unweighted
and degree-penalized graphs, samples 10K node pairs, and produces side-by-side
scatter plots with Spearman rho overlays. Supports --n_pairs flag for debugging.
"""

import argparse
import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reconstruction.edge_weights import degree_penalized_weight, unit_weight

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "sift10k")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SEED = 42
ALPHA = 3.0
DEFAULT_N_PAIRS = 10_000


def load_data():
    vectors = np.load(os.path.join(OUTPUT_DIR, "vectors.npy"))
    with open(os.path.join(OUTPUT_DIR, f"adj_seed{SEED}.pkl"), "rb") as f:
        data = pickle.load(f)
    return vectors, data["adjacency_list"]


def sample_pairs(n, n_pairs, seed):
    rng = np.random.default_rng(seed)
    pairs_i = rng.integers(0, n, size=n_pairs * 2)
    pairs_j = rng.integers(0, n, size=n_pairs * 2)
    mask = pairs_i != pairs_j
    pairs_i = pairs_i[mask][:n_pairs]
    pairs_j = pairs_j[mask][:n_pairs]
    return pairs_i, pairs_j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=DEFAULT_N_PAIRS)
    args = parser.parse_args()
    n_pairs = args.n_pairs

    os.makedirs(FIGURES_DIR, exist_ok=True)
    vectors, adjacency_list = load_data()
    n = vectors.shape[0]
    print(f"Loaded {n} vectors, dim={vectors.shape[1]}")

    print("Building weight matrices...")
    wm_unit = unit_weight(adjacency_list)
    wm_pen = degree_penalized_weight(adjacency_list, alpha=ALPHA)

    print("Computing all-pairs shortest paths (unweighted)...")
    t0 = time.time()
    dist_unit = shortest_path(wm_unit, directed=False)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Computing all-pairs shortest paths (degree-penalized)...")
    t0 = time.time()
    dist_pen = shortest_path(wm_pen, directed=False)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"Sampling {n_pairs} node pairs...")
    pairs_i, pairs_j = sample_pairs(n, n_pairs, SEED)

    euclidean_dists = np.sqrt(np.sum((vectors[pairs_i] - vectors[pairs_j]) ** 2, axis=1))
    geodesic_unit = dist_unit[pairs_i, pairs_j]
    geodesic_pen = dist_pen[pairs_i, pairs_j]

    finite_mask_unit = np.isfinite(geodesic_unit)
    finite_mask_pen = np.isfinite(geodesic_pen)

    rho_unit, _ = spearmanr(geodesic_unit[finite_mask_unit], euclidean_dists[finite_mask_unit])
    rho_pen, _ = spearmanr(geodesic_pen[finite_mask_pen], euclidean_dists[finite_mask_pen])

    print(f"Spearman rho (unweighted): {rho_unit:.4f}")
    print(f"Spearman rho (degree-penalized): {rho_pen:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(euclidean_dists[finite_mask_unit], geodesic_unit[finite_mask_unit],
                    alpha=0.1, s=4, color="#2c7bb6", edgecolors="none")
    axes[0].set_xlabel("True Euclidean Distance", fontsize=11)
    axes[0].set_ylabel("Unweighted Geodesic Distance (hops)", fontsize=11)
    axes[0].set_title("Unweighted Geodesic vs Euclidean", fontsize=12)
    axes[0].text(0.05, 0.95, f"Spearman $\\rho$ = {rho_unit:.4f}",
                 transform=axes[0].transAxes, fontsize=11, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(euclidean_dists[finite_mask_pen], geodesic_pen[finite_mask_pen],
                    alpha=0.1, s=4, color="#d7191c", edgecolors="none")
    axes[1].set_xlabel("True Euclidean Distance", fontsize=11)
    axes[1].set_ylabel("Degree-Penalized Geodesic Distance", fontsize=11)
    axes[1].set_title(f"Degree-Penalized Geodesic ($\\alpha$={ALPHA}) vs Euclidean", fontsize=12)
    axes[1].text(0.05, 0.95, f"Spearman $\\rho$ = {rho_pen:.4f}",
                 transform=axes[1].transAxes, fontsize=11, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("SIFT10K: Graph-Geodesic vs True Euclidean Distances", fontsize=13, y=1.02)
    fig.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "geodesic_vs_euclidean_scatter.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    meta_path = os.path.join(RESULTS_DIR, "geodesic_scatter_meta.json")
    meta = {
        "dataset": "SIFT10K",
        "seed": SEED,
        "alpha": ALPHA,
        "n_pairs": int(len(pairs_i)),
        "spearman_unweighted": round(float(rho_unit), 6),
        "spearman_degree_penalized": round(float(rho_pen), 6),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
