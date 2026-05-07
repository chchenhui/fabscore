# Computes overlap metrics between FCBoost's static CA-derived mask and
# Kitty's dynamic magnitude-based channel selection. Generates visualizations
# (heatmaps, histograms, scatter plots) and saves results to JSON.
# Uses precomputed CA scores and magnitude statistics (no GPU required).

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_ca_data(masks_dir):
    ca_scores = np.load(os.path.join(masks_dir, "qwen3_8b_ca_scores.npy"))
    masks_dict = torch.load(os.path.join(masks_dir, "qwen3_8b_ca_masks.pt"), weights_only=True)

    num_layers = ca_scores.shape[0]
    num_kv_heads = ca_scores.shape[1]
    head_dim = masks_dict[0].shape[1]

    ca_mask_array = np.zeros((num_layers, num_kv_heads, head_dim), dtype=np.bool_)
    for l in range(num_layers):
        ca_mask_array[l] = masks_dict[l].numpy()

    return ca_scores, ca_mask_array


def load_magnitude_data(mag_stats_path):
    data = np.load(mag_stats_path)
    per_channel_mean_mag = data["per_channel_mean_magnitude"]
    per_pair_mean_mag = data["per_pair_mean_magnitude"]
    freq_mask = data["freq_mask"]
    channel_freq = data["channel_freq"]
    total_pages = data["total_pages"]

    assert per_channel_mean_mag.shape[2] == 128, \
        f"Expected 128 channels, got {per_channel_mean_mag.shape[2]}"
    assert per_pair_mean_mag.shape[2] == 64, \
        f"Expected 64 RoPE pairs, got {per_pair_mean_mag.shape[2]}"

    return {
        "per_channel_mean_magnitude": per_channel_mean_mag,
        "per_pair_mean_magnitude": per_pair_mean_mag,
        "freq_mask": freq_mask,
        "channel_freq": channel_freq,
        "total_pages": total_pages,
    }


def compute_jaccard(ca_mask, mag_mask):
    num_layers, num_kv_heads, _ = ca_mask.shape
    jaccard = np.zeros((num_layers, num_kv_heads))

    for l in range(num_layers):
        for h in range(num_kv_heads):
            intersection = np.logical_and(ca_mask[l, h], mag_mask[l, h]).sum()
            union = np.logical_or(ca_mask[l, h], mag_mask[l, h]).sum()
            jaccard[l, h] = intersection / union if union > 0 else 0.0

    return jaccard


def compute_spearman(ca_scores, per_pair_mean_mag):
    num_layers, num_kv_heads, num_pairs = ca_scores.shape
    spearman_rho = np.zeros((num_layers, num_kv_heads))
    spearman_pval = np.zeros((num_layers, num_kv_heads))

    for l in range(num_layers):
        for h in range(num_kv_heads):
            rho, pval = stats.spearmanr(ca_scores[l, h], per_pair_mean_mag[l, h])
            spearman_rho[l, h] = rho
            spearman_pval[l, h] = pval

    return spearman_rho, spearman_pval


def create_visualizations(jaccard, spearman_rho, ca_scores, per_pair_mean_mag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_layers, num_kv_heads = jaccard.shape

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(jaccard, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=[f"KVH{i}" for i in range(num_kv_heads)],
                yticklabels=[f"L{i}" for i in range(num_layers)],
                vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Jaccard Overlap: CA Mask vs Magnitude Mask\n(Mean={jaccard.mean():.3f})")
    ax.set_xlabel("KV Head")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jaccard_heatmap.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(spearman_rho, annot=True, fmt=".2f", cmap="RdBu_r",
                xticklabels=[f"KVH{i}" for i in range(num_kv_heads)],
                yticklabels=[f"L{i}" for i in range(num_layers)],
                vmin=-1, vmax=1, ax=ax, center=0)
    ax.set_title(f"Spearman Rank Correlation: CA Score vs Magnitude\n(Mean={spearman_rho.mean():.3f})")
    ax.set_xlabel("KV Head")
    ax.set_ylabel("Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spearman_heatmap.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(jaccard.flatten(), bins=20, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.axvline(jaccard.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean={jaccard.mean():.3f}")
    ax.set_xlabel("Jaccard Overlap")
    ax.set_ylabel("Count (layer, KV head)")
    ax.set_title("Distribution of Jaccard Overlap Values")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jaccard_histogram.png"), dpi=150)
    plt.close()

    representative = [
        (0, 0, "Layer 0, KV Head 0 (early)"),
        (num_layers // 4, 0, f"Layer {num_layers // 4}, KV Head 0 (early-mid)"),
        (num_layers // 2, 0, f"Layer {num_layers // 2}, KV Head 0 (mid)"),
        (3 * num_layers // 4, 0, f"Layer {3 * num_layers // 4}, KV Head 0 (late-mid)"),
        (num_layers - 1, 0, f"Layer {num_layers - 1}, KV Head 0 (last)"),
    ]

    best_j_idx = np.unravel_index(np.argmax(jaccard), jaccard.shape)
    worst_j_idx = np.unravel_index(np.argmin(jaccard), jaccard.shape)
    representative.append((best_j_idx[0], best_j_idx[1],
                           f"Layer {best_j_idx[0]}, KVH {best_j_idx[1]} (best Jaccard={jaccard[best_j_idx]:.3f})"))
    representative.append((worst_j_idx[0], worst_j_idx[1],
                           f"Layer {worst_j_idx[0]}, KVH {worst_j_idx[1]} (worst Jaccard={jaccard[worst_j_idx]:.3f})"))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for idx, (l, h, title) in enumerate(representative[:8]):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
        ca = ca_scores[l, h]
        mag = per_pair_mean_mag[l, h]
        rho, _ = stats.spearmanr(ca, mag)
        ax.scatter(ca, mag, alpha=0.5, s=15, color="#4C72B0")
        ax.set_xlabel("CA Score (per RoPE pair)")
        ax.set_ylabel("Mean Magnitude (per RoPE pair)")
        ax.set_title(f"{title}\nSpearman={rho:.3f}", fontsize=9)

    for idx in range(len(representative), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("CA Score vs Magnitude per RoPE Pair", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ca_vs_magnitude_scatter.png"), dpi=150)
    plt.close()

    print(f"Saved 4 figures to {output_dir}")


def interpret_results(jaccard, spearman_rho, fcboost_avg=71.11, kitty_avg=66.67):
    mean_jaccard = float(jaccard.mean())
    mean_spearman = float(spearman_rho.mean())
    fcboost_beats_kitty = fcboost_avg >= kitty_avg - 1.0

    if mean_jaccard > 0.5 and fcboost_beats_kitty:
        scenario = 1
        interpretation = (
            f"Scenario 1: High overlap (mean Jaccard={mean_jaccard:.3f} > 0.5) and "
            f"FCBoost matches/beats Kitty ({fcboost_avg:.2f}% vs {kitty_avg:.2f}%). "
            f"CA is a valid static proxy for magnitude-based channel selection. "
            f"The static CA-derived mask identifies largely the same channels as Kitty's "
            f"dynamic magnitude heuristic."
        )
    elif mean_jaccard <= 0.5 and fcboost_beats_kitty:
        scenario = 2
        interpretation = (
            f"Scenario 2: Low overlap (mean Jaccard={mean_jaccard:.3f} <= 0.5) but "
            f"FCBoost still matches/beats Kitty ({fcboost_avg:.2f}% vs {kitty_avg:.2f}%). "
            f"Static channel selection works, but via a different mechanism than "
            f"magnitude-based selection. CA identifies structurally important RoPE "
            f"frequencies that are quantization-sensitive regardless of their magnitude."
        )
    else:
        scenario = 3
        interpretation = (
            f"Scenario 3: Low overlap (mean Jaccard={mean_jaccard:.3f}) and "
            f"FCBoost fails to match Kitty ({fcboost_avg:.2f}% vs {kitty_avg:.2f}%). "
            f"Dynamic per-page channel selection is necessary; static masks are insufficient."
        )

    correlation_note = ""
    if mean_spearman > 0.5:
        correlation_note = (
            f"High mean Spearman correlation ({mean_spearman:.3f}) indicates CA scores and "
            f"magnitude scores rank channels similarly, reinforcing that CA proxies magnitude."
        )
    elif mean_spearman > 0.0:
        correlation_note = (
            f"Moderate positive Spearman correlation ({mean_spearman:.3f}) indicates partial "
            f"alignment between CA and magnitude rankings, suggesting CA captures some but "
            f"not all of what magnitude selection identifies."
        )
    else:
        correlation_note = (
            f"Low/negative Spearman correlation ({mean_spearman:.3f}) indicates CA and "
            f"magnitude rankings are not aligned. The two methods identify different channels."
        )

    return {
        "scenario": scenario,
        "interpretation": interpretation,
        "correlation_note": correlation_note,
        "mean_jaccard": mean_jaccard,
        "mean_spearman": mean_spearman,
    }


def main():
    parser = argparse.ArgumentParser(description="CA-Magnitude Overlap Analysis")
    parser.add_argument("--ca_dir", type=str, default="fcboost/masks_v2")
    parser.add_argument("--mag_stats", type=str, default="fcboost/analysis/magnitude_stats.npz")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="results/figures")
    args = parser.parse_args()

    print("=" * 80)
    print("CA-Magnitude Overlap Analysis")
    print("=" * 80)

    print("\nLoading CA data...")
    ca_scores, ca_mask = load_ca_data(args.ca_dir)
    print(f"  CA scores shape: {ca_scores.shape}")
    print(f"  CA mask shape: {ca_mask.shape}")
    print(f"  Boosted channels per head: {ca_mask[0, 0].sum()}")

    print("\nLoading magnitude data...")
    mag_data = load_magnitude_data(args.mag_stats)
    print(f"  Per-channel magnitude shape: {mag_data['per_channel_mean_magnitude'].shape}")
    print(f"  Per-pair magnitude shape: {mag_data['per_pair_mean_magnitude'].shape}")
    print(f"  Freq mask shape: {mag_data['freq_mask'].shape}")
    print(f"  Total pages (layer 0): {mag_data['total_pages'][0]}")

    print("\nComputing Jaccard overlap...")
    jaccard = compute_jaccard(ca_mask, mag_data["freq_mask"])
    print(f"  Mean Jaccard: {jaccard.mean():.4f}")
    print(f"  Min Jaccard: {jaccard.min():.4f}")
    print(f"  Max Jaccard: {jaccard.max():.4f}")
    print(f"  Std Jaccard: {jaccard.std():.4f}")

    print("\nComputing Spearman rank correlation...")
    spearman_rho, spearman_pval = compute_spearman(ca_scores, mag_data["per_pair_mean_magnitude"])
    print(f"  Mean Spearman rho: {spearman_rho.mean():.4f}")
    print(f"  Min Spearman rho: {spearman_rho.min():.4f}")
    print(f"  Max Spearman rho: {spearman_rho.max():.4f}")
    sig_fraction = (spearman_pval < 0.05).mean()
    print(f"  Fraction significant (p<0.05): {sig_fraction:.2%}")

    print("\nGenerating visualizations...")
    create_visualizations(jaccard, spearman_rho, ca_scores, mag_data["per_pair_mean_magnitude"], args.figures_dir)

    print("\nInterpreting results...")
    interpretation = interpret_results(jaccard, spearman_rho)
    print(f"\n  Scenario: {interpretation['scenario']}")
    print(f"  {interpretation['interpretation']}")
    print(f"  {interpretation['correlation_note']}")

    results = {
        "task": "CA-Magnitude Overlap Analysis",
        "model": "Qwen/Qwen3-8B",
        "ca_profiling": {
            "source": "fcboost/masks_v2",
            "num_sequences": 16,
            "max_seq_len": 8192,
            "top_f_rope_pairs": 8,
            "k_boosted_channels": 16,
        },
        "magnitude_collection": {
            "source": "fcboost/analysis/magnitude_stats.npz",
            "calibration": "WikiText-2 (same as CA profiling)",
            "buffer_length": 128,
            "k_channels": 16,
            "total_pages_layer0": mag_data["total_pages"][0].tolist(),
        },
        "jaccard_overlap": {
            "mean": float(jaccard.mean()),
            "std": float(jaccard.std()),
            "min": float(jaccard.min()),
            "max": float(jaccard.max()),
            "median": float(np.median(jaccard)),
            "per_layer_mean": jaccard.mean(axis=1).tolist(),
            "per_head_mean": jaccard.mean(axis=0).tolist(),
        },
        "spearman_correlation": {
            "mean_rho": float(spearman_rho.mean()),
            "std_rho": float(spearman_rho.std()),
            "min_rho": float(spearman_rho.min()),
            "max_rho": float(spearman_rho.max()),
            "median_rho": float(np.median(spearman_rho)),
            "fraction_significant_p05": float(sig_fraction),
            "per_layer_mean_rho": spearman_rho.mean(axis=1).tolist(),
            "per_head_mean_rho": spearman_rho.mean(axis=0).tolist(),
        },
        "interpretation": interpretation,
        "fcboost_accuracy": {"avg": 71.11, "aime24": 74.44, "aime25": 67.78},
        "kitty_accuracy": {"avg": 66.67, "aime24": 72.22, "aime25": 61.11},
        "figures": [
            "results/figures/jaccard_heatmap.png",
            "results/figures/spearman_heatmap.png",
            "results/figures/jaccard_histogram.png",
            "results/figures/ca_vs_magnitude_scatter.png",
        ],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "overlap_analysis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
