# FGGM mask overlap (Jaccard similarity) analysis across default order and Order 2.
# Loads per-task binary masks, computes global and layer-wise pairwise Jaccard
# similarity, generates heatmaps/bar charts/layer profiles, and saves metrics JSON.
# Task 0 = all-ones (no mask, all params trainable); tasks 1-7 have saved masks.

import os
import re
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_ORDER_TASKS = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten",
]
ORDER2_TASKS = [
    "NumGLUE-cm", "NumGLUE-ds", "FOMC", "20Minuten",
    "C-STANCE", "Py150", "MeetingBank", "ScienceQA",
]

RUNS = {
    "default_seed42": {
        "mask_dir": BASE_DIR / "audit" / "results" / "fggm_default_seed42_v7" / "masks",
        "task_names": DEFAULT_ORDER_TASKS,
    },
    "order2_seed42": {
        "mask_dir": BASE_DIR / "audit" / "results" / "fggm_order2_seed42" / "masks",
        "task_names": ORDER2_TASKS,
    },
    "order2_seed123": {
        "mask_dir": BASE_DIR / "audit" / "results" / "fggm_order2_seed123" / "masks",
        "task_names": ORDER2_TASKS,
    },
    "order2_seed456": {
        "mask_dir": BASE_DIR / "audit" / "results" / "fggm_order2_seed456" / "masks",
        "task_names": ORDER2_TASKS,
    },
}

OUT_DIR = BASE_DIR / "audit" / "results" / "analysis" / "mask_overlap"


def load_masks(mask_dir, reference_mask_path=None):
    """Load task masks 1-7; create all-ones mask for task 0."""
    m1 = torch.load(mask_dir / "task_1.pt", map_location="cpu")
    task0_mask = {}
    for k, v in m1.items():
        task0_mask[k] = torch.ones_like(v)

    masks = [task0_mask]
    for t in range(1, 8):
        masks.append(torch.load(mask_dir / f"task_{t}.pt", map_location="cpu"))
    return masks


def flatten_mask(mask_dict):
    """Flatten a mask dict into a single 1D boolean tensor."""
    parts = []
    for k in sorted(mask_dict.keys()):
        parts.append(mask_dict[k].flatten().float())
    return torch.cat(parts)


def jaccard(a, b):
    """Jaccard similarity between two 1D binary tensors."""
    a_bool = a > 0.5
    b_bool = b > 0.5
    inter = (a_bool & b_bool).sum().item()
    union = (a_bool | b_bool).sum().item()
    if union == 0:
        return 1.0
    return inter / union


def pairwise_jaccard_matrix(masks):
    """Compute 8x8 pairwise Jaccard matrix from list of 8 mask dicts."""
    flat = [flatten_mask(m) for m in masks]
    n = len(flat)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = jaccard(flat[i], flat[j])
    return mat


def consecutive_jaccard(masks):
    """Compute Jaccard for 7 consecutive task pairs (0-1, 1-2, ..., 6-7)."""
    flat = [flatten_mask(m) for m in masks]
    pairs = []
    for i in range(len(flat) - 1):
        pairs.append(jaccard(flat[i], flat[i + 1]))
    return pairs


def group_params_by_layer(mask_dict):
    """Group param names into layer buckets: 'layer_0' .. 'layer_27', 'other'."""
    groups = {}
    for k in sorted(mask_dict.keys()):
        match = re.search(r"model\.layers\.(\d+)\.", k)
        if match:
            layer_key = f"layer_{int(match.group(1))}"
        else:
            layer_key = "other"
        if layer_key not in groups:
            groups[layer_key] = []
        groups[layer_key].append(k)
    return groups


def layerwise_jaccard(masks):
    """Per-layer average consecutive Jaccard across task pairs."""
    groups = group_params_by_layer(masks[0])
    layer_indices = sorted(
        [k for k in groups if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]),
    )

    result = {}
    for layer_key in layer_indices:
        param_names = groups[layer_key]
        jacc_vals = []
        for t in range(len(masks) - 1):
            parts_a = []
            parts_b = []
            for pn in param_names:
                parts_a.append(masks[t][pn].flatten().float())
                parts_b.append(masks[t + 1][pn].flatten().float())
            flat_a = torch.cat(parts_a)
            flat_b = torch.cat(parts_b)
            jacc_vals.append(jaccard(flat_a, flat_b))
        result[layer_key] = {
            "per_pair": jacc_vals,
            "mean": float(np.mean(jacc_vals)),
        }
    return result


def layerwise_consecutive_pair_jaccard(masks):
    """Per-layer Jaccard for each consecutive pair."""
    groups = group_params_by_layer(masks[0])
    layer_indices = sorted(
        [k for k in groups if k.startswith("layer_")],
        key=lambda x: int(x.split("_")[1]),
    )

    all_pairs = {}
    for t in range(len(masks) - 1):
        pair_key = f"pair_{t}_{t+1}"
        pair_data = {}
        for layer_key in layer_indices:
            param_names = groups[layer_key]
            parts_a = [masks[t][pn].flatten().float() for pn in param_names]
            parts_b = [masks[t + 1][pn].flatten().float() for pn in param_names]
            flat_a = torch.cat(parts_a)
            flat_b = torch.cat(parts_b)
            pair_data[layer_key] = jaccard(flat_a, flat_b)
        all_pairs[pair_key] = pair_data
    return all_pairs, layer_indices


def plot_heatmaps(mat_default, mat_order2, task_names_default, task_names_order2, out_path):
    """Side-by-side 8x8 Jaccard heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    im1 = ax1.imshow(mat_default, cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.set_xticklabels(task_names_default, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(task_names_default, fontsize=8)
    ax1.set_title("Default Order (seed=42)", fontsize=12, fontweight="bold")
    for i in range(8):
        for j in range(8):
            ax1.text(j, i, f"{mat_default[i, j]:.3f}", ha="center", va="center",
                     fontsize=7, color="white" if mat_default[i, j] > 0.6 else "black")

    im2 = ax2.imshow(mat_order2, cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_xticks(range(8))
    ax2.set_yticks(range(8))
    ax2.set_xticklabels(task_names_order2, rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels(task_names_order2, fontsize=8)
    ax2.set_title("Order 2 (seed=42)", fontsize=12, fontweight="bold")
    for i in range(8):
        for j in range(8):
            ax2.text(j, i, f"{mat_order2[i, j]:.3f}", ha="center", va="center",
                     fontsize=7, color="white" if mat_order2[i, j] > 0.6 else "black")

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Jaccard Similarity")
    fig.suptitle("FGGM Per-Task Mask Pairwise Jaccard Similarity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {out_path}")


def plot_bar_consecutive(default_consec, order2_consec_seeds, out_path):
    """Bar chart: avg consecutive Jaccard for default vs Order 2 (mean+/-std)."""
    default_mean = np.mean(default_consec)

    order2_means = [np.mean(s) for s in order2_consec_seeds]
    order2_grand_mean = np.mean(order2_means)
    order2_std = np.std(order2_means, ddof=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Default Order\n(seed=42)", "Order 2\n(mean of 3 seeds)"],
        [default_mean, order2_grand_mean],
        yerr=[0, order2_std],
        capsize=8,
        color=["#4C72B0", "#DD8452"],
        edgecolor="black",
        width=0.5,
    )
    ax.set_ylabel("Average Consecutive-Task Jaccard Similarity", fontsize=11)
    ax.set_title("FGGM Mask Overlap: Default vs Order 2", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)

    for bar, val in zip(bars, [default_mean, order2_grand_mean]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart to {out_path}")


def plot_layerwise(lw_default, lw_order2_seeds, out_path):
    """Line plot: per-layer avg consecutive Jaccard for both orders."""
    layer_keys = sorted(lw_default.keys(), key=lambda x: int(x.split("_")[1]))
    layer_nums = [int(k.split("_")[1]) for k in layer_keys]

    default_vals = [lw_default[k]["mean"] for k in layer_keys]

    order2_all = []
    for lw in lw_order2_seeds:
        order2_all.append([lw[k]["mean"] for k in layer_keys])
    order2_arr = np.array(order2_all)
    order2_mean = order2_arr.mean(axis=0)
    order2_std = order2_arr.std(axis=0, ddof=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layer_nums, default_vals, "o-", color="#4C72B0", label="Default Order (seed=42)",
            linewidth=2, markersize=5)
    ax.plot(layer_nums, order2_mean, "s-", color="#DD8452", label="Order 2 (mean of 3 seeds)",
            linewidth=2, markersize=5)
    ax.fill_between(layer_nums, order2_mean - order2_std, order2_mean + order2_std,
                    alpha=0.2, color="#DD8452")
    ax.set_xlabel("Transformer Layer Index", fontsize=11)
    ax.set_ylabel("Avg Consecutive-Task Jaccard Similarity", fontsize=11)
    ax.set_title("Layer-wise Mask Overlap: Default vs Order 2", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(layer_nums)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved layer-wise plot to {out_path}")


def plot_consecutive_per_pair(default_consec, order2_consec_seeds, task_names_default,
                              task_names_order2, out_path):
    """Bar chart showing Jaccard for each consecutive pair."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    pair_labels_d = [f"{task_names_default[i]}→\n{task_names_default[i+1]}" for i in range(7)]
    ax1.bar(range(7), default_consec, color="#4C72B0", edgecolor="black", width=0.6)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(pair_labels_d, fontsize=8)
    ax1.set_ylabel("Jaccard Similarity", fontsize=10)
    ax1.set_title("Default Order: Consecutive Mask Overlap (seed=42)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    for i, v in enumerate(default_consec):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    pair_labels_o2 = [f"{task_names_order2[i]}→\n{task_names_order2[i+1]}" for i in range(7)]
    o2_arr = np.array(order2_consec_seeds)
    o2_mean = o2_arr.mean(axis=0)
    o2_std = o2_arr.std(axis=0, ddof=0)
    ax2.bar(range(7), o2_mean, yerr=o2_std, capsize=5, color="#DD8452", edgecolor="black", width=0.6)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(pair_labels_o2, fontsize=8)
    ax2.set_ylabel("Jaccard Similarity", fontsize=10)
    ax2.set_title("Order 2: Consecutive Mask Overlap (mean ± std of 3 seeds)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.0)
    for i, v in enumerate(o2_mean):
        ax2.text(i, v + o2_std[i] + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved per-pair bar chart to {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading masks...")
    all_masks = {}
    for run_name, cfg in RUNS.items():
        all_masks[run_name] = load_masks(cfg["mask_dir"])
        print(f"  Loaded {run_name}: {len(all_masks[run_name])} task masks")

    print("\nComputing pairwise Jaccard matrices...")
    mat_default = pairwise_jaccard_matrix(all_masks["default_seed42"])
    mat_order2 = pairwise_jaccard_matrix(all_masks["order2_seed42"])

    print("\nComputing consecutive Jaccard...")
    consec_default = consecutive_jaccard(all_masks["default_seed42"])
    consec_order2 = {
        seed: consecutive_jaccard(all_masks[f"order2_{seed}"])
        for seed in ["seed42", "seed123", "seed456"]
    }
    print(f"  Default: {[f'{v:.4f}' for v in consec_default]}, avg={np.mean(consec_default):.4f}")
    for seed, vals in consec_order2.items():
        print(f"  Order2 {seed}: {[f'{v:.4f}' for v in vals]}, avg={np.mean(vals):.4f}")

    print("\nComputing layer-wise Jaccard...")
    lw_default = layerwise_jaccard(all_masks["default_seed42"])
    lw_order2 = {
        seed: layerwise_jaccard(all_masks[f"order2_{seed}"])
        for seed in ["seed42", "seed123", "seed456"]
    }

    print("\nGenerating visualizations...")
    plot_heatmaps(
        mat_default, mat_order2,
        DEFAULT_ORDER_TASKS, ORDER2_TASKS,
        OUT_DIR / "heatmap_jaccard.png",
    )

    plot_bar_consecutive(
        consec_default,
        [consec_order2[s] for s in ["seed42", "seed123", "seed456"]],
        OUT_DIR / "bar_consecutive_jaccard.png",
    )

    plot_layerwise(
        lw_default,
        [lw_order2[s] for s in ["seed42", "seed123", "seed456"]],
        OUT_DIR / "layerwise_jaccard.png",
    )

    plot_consecutive_per_pair(
        consec_default,
        [consec_order2[s] for s in ["seed42", "seed123", "seed456"]],
        DEFAULT_ORDER_TASKS, ORDER2_TASKS,
        OUT_DIR / "consecutive_per_pair.png",
    )

    print("\nSaving metrics...")
    order2_avgs = [np.mean(consec_order2[s]) for s in ["seed42", "seed123", "seed456"]]
    metrics = {
        "pairwise_jaccard_default_seed42": mat_default.tolist(),
        "pairwise_jaccard_order2_seed42": mat_order2.tolist(),
        "consecutive_jaccard": {
            "default_seed42": consec_default,
            "order2_seed42": consec_order2["seed42"],
            "order2_seed123": consec_order2["seed123"],
            "order2_seed456": consec_order2["seed456"],
        },
        "avg_consecutive_jaccard": {
            "default_seed42": float(np.mean(consec_default)),
            "order2_seed42": float(np.mean(consec_order2["seed42"])),
            "order2_seed123": float(np.mean(consec_order2["seed123"])),
            "order2_seed456": float(np.mean(consec_order2["seed456"])),
            "order2_mean": float(np.mean(order2_avgs)),
            "order2_std": float(np.std(order2_avgs, ddof=0)),
        },
        "layerwise_jaccard": {
            "default_seed42": {k: v["mean"] for k, v in lw_default.items()},
            "order2_seed42": {k: v["mean"] for k, v in lw_order2["seed42"].items()},
            "order2_seed123": {k: v["mean"] for k, v in lw_order2["seed123"].items()},
            "order2_seed456": {k: v["mean"] for k, v in lw_order2["seed456"].items()},
        },
        "task_names": {
            "default": DEFAULT_ORDER_TASKS,
            "order2": ORDER2_TASKS,
        },
        "first_pair_jaccard": {
            "default_seed42_pair01": consec_default[0],
            "order2_seed42_pair01": consec_order2["seed42"][0],
            "order2_seed123_pair01": consec_order2["seed123"][0],
            "order2_seed456_pair01": consec_order2["seed456"][0],
        },
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {OUT_DIR / 'metrics.json'}")

    print("\n=== Summary ===")
    print(f"Default order avg consecutive Jaccard: {np.mean(consec_default):.4f}")
    print(f"Order 2 avg consecutive Jaccard: {np.mean(order2_avgs):.4f} ± {np.std(order2_avgs, ddof=0):.4f}")
    print(f"First pair (task0→task1) Jaccard:")
    print(f"  Default (C-STANCE→FOMC): {consec_default[0]:.4f}")
    for seed in ["seed42", "seed123", "seed456"]:
        print(f"  Order2 {seed} (NumGLUE-cm→NumGLUE-ds): {consec_order2[seed][0]:.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
