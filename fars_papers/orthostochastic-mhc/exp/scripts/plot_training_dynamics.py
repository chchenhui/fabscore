"""
Generate training dynamics visualization figures for mHC-Sinkhorn vs mHC-Orthostochastic.

Produces 3 PDF figures:
  1. results/validation_loss_curves.pdf   - val loss mean+/-std over training
  2. results/gradient_norm_trajectories.pdf - smoothed grad norm trajectories
  3. results/h_res_heatmaps.pdf           - learned H_res matrices at selected layers

Data sources:
  - Val loss: parsed from .train_service_logs stdout
  - Grad norms: diagnostics/gradient_spikes.json per run
  - H_res: ckpt.pt model state dicts + projection functions
"""

import json
import os
import re
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "orthostochastic_mhc_experiments" / "logs"
TRAIN_LOGS = BASE / ".train_service_logs"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE / "mHC-manifold-constrained-hyper-connections"))
from hyper_connections.hyper_connections import sinkhorn_log, orthostochastic_project

RUN_GROUPS = {
    "A": {
        "sinkhorn": [f"setting_a_sinkhorn_seed{s}" for s in range(1, 6)],
        "orthostochastic": [f"setting_a_orthostochastic_optimized_seed{s}" for s in range(1, 6)],
        "hc_unconstrained": ["setting_a_hc_unconstrained_seed1"],
    },
    "B": {
        "sinkhorn": [f"setting_b_sinkhorn_seed{s}" for s in range(1, 4)],
        "orthostochastic": [
            "setting_b_orthostochastic_optimized_seed11",
            "setting_b_orthostochastic_optimized_seed2",
            "setting_b_orthostochastic_optimized_seed3",
        ],
    },
}

SINKHORN_PARAMS = {"num_iters": 10, "tau": 0.05}
ORTHO_PARAMS_A = {"ns_steps": 15, "ns_eps": 1e-7, "ns_coeffs": (3.0, -3.2, 1.2)}
ORTHO_PARAMS_B = {"ns_steps": 20, "ns_eps": 1e-7, "ns_coeffs": (3.0, -3.2, 1.2)}

HEATMAP_LAYERS_A = [0, 11, 23, 35, 47]
HEATMAP_LAYERS_B = [0, 2, 5]

COLORS = {"sinkhorn": "C0", "orthostochastic": "C1", "hc_unconstrained": "gray"}
LABELS = {"sinkhorn": "mHC-Sinkhorn", "orthostochastic": "mHC-Orthostochastic", "hc_unconstrained": "HC Unconstrained"}


_LOG_MAPPING_CACHE = None

def _build_log_mapping():
    global _LOG_MAPPING_CACHE
    if _LOG_MAPPING_CACHE is not None:
        return _LOG_MAPPING_CACHE
    _LOG_MAPPING_CACHE = {}
    for log_path in sorted(TRAIN_LOGS.glob("*/output.log")):
        try:
            with open(log_path, "r", errors="replace") as f:
                content = f.read(80000)
            m = re.search(r"out_dir=(\S+?)(?:/diagnostics|\s|$)", content)
            if m:
                out_dir = m.group(1)
                run_name = Path(out_dir).name
                _LOG_MAPPING_CACHE[run_name] = log_path
        except Exception:
            continue
    return _LOG_MAPPING_CACHE

def find_train_service_log(run_dir_name):
    mapping = _build_log_mapping()
    return mapping.get(run_dir_name)


def parse_val_loss_from_log(log_path):
    pattern = re.compile(r"iter (\d+):.*val loss ([\d.]+)")
    iters, losses = [], []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(iters), np.array(losses)


def load_grad_norms(run_dir_name):
    path = LOGS / run_dir_name / "diagnostics" / "gradient_spikes.json"
    with open(path) as f:
        data = json.load(f)
    entries = data["grad_norms"]
    iters = np.array([e["iter"] for e in entries])
    norms = np.array([e["grad_norm"] for e in entries])
    return iters, norms


def moving_average(arr, window=200):
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    return smoothed


def aggregate_curves(all_iters_list, all_vals_list, min_seeds=2):
    all_iters_set = set()
    for iters in all_iters_list:
        all_iters_set.update(iters.tolist())
    common_iters = sorted(all_iters_set)

    iter_to_vals = {it: [] for it in common_iters}
    for iters, vals in zip(all_iters_list, all_vals_list):
        mapping = dict(zip(iters.tolist(), vals.tolist()))
        for it in common_iters:
            if it in mapping:
                iter_to_vals[it].append(mapping[it])

    final_iters, means, stds = [], [], []
    for it in common_iters:
        vs = iter_to_vals[it]
        if len(vs) >= min_seeds:
            final_iters.append(it)
            means.append(np.mean(vs))
            stds.append(np.std(vs))
    return np.array(final_iters), np.array(means), np.array(stds)


def aggregate_grad_norms(all_iters_list, all_smoothed_list):
    min_len = min(len(s) for s in all_smoothed_list)
    stacked = np.stack([s[:min_len] for s in all_smoothed_list], axis=0)
    iters = all_iters_list[0][:min_len]
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    return iters, mean, std


def project_h_res(logits_tensor, method, setting, alpha_logit=None):
    with torch.no_grad():
        if method == "sinkhorn":
            S = sinkhorn_log(logits_tensor, **SINKHORN_PARAMS)
        elif method == "orthostochastic":
            params = ORTHO_PARAMS_A if setting == "A" else ORTHO_PARAMS_B
            S = orthostochastic_project(logits_tensor, **params)
        else:
            return None

        if alpha_logit is not None:
            alpha = torch.sigmoid(alpha_logit)
            n = S.shape[0]
            I = torch.eye(n, dtype=S.dtype, device=S.device)
            H_res = (1 - alpha) * I + alpha * S
        else:
            H_res = S
    return H_res.numpy()


# ============================================================
# Phase 1: Validation Loss Curves
# ============================================================
def plot_val_loss_curves(test_only=False):
    print("\n=== Phase 1: Validation Loss Curves ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, setting in enumerate(["A", "B"]):
        ax = axes[ax_idx]
        groups = RUN_GROUPS[setting]

        for method in ["sinkhorn", "orthostochastic"]:
            runs = groups[method]
            all_iters, all_vals = [], []
            for run in runs:
                log_path = find_train_service_log(run)
                if log_path is None:
                    print(f"  WARNING: No TrainService log found for {run}")
                    continue
                iters, losses = parse_val_loss_from_log(log_path)
                print(f"  {run}: {len(iters)} val loss points from {log_path.parent.name}")
                all_iters.append(iters)
                all_vals.append(losses)
                if test_only:
                    break

            if not all_iters:
                continue

            for i, (it, vl) in enumerate(zip(all_iters, all_vals)):
                ax.plot(it, vl, color=COLORS[method], alpha=0.15, linewidth=0.7)

            if len(all_iters) == 1:
                ax.plot(all_iters[0], all_vals[0], color=COLORS[method], label=LABELS[method])
            else:
                iters, mean, std = aggregate_curves(all_iters, all_vals)
                ax.plot(iters, mean, color=COLORS[method], label=LABELS[method], linewidth=2)
                ax.fill_between(iters, mean - std, mean + std, alpha=0.2, color=COLORS[method])

        if "hc_unconstrained" in groups:
            run = groups["hc_unconstrained"][0]
            log_path = find_train_service_log(run)
            if log_path:
                iters, losses = parse_val_loss_from_log(log_path)
                print(f"  {run}: {len(iters)} val loss points from {log_path.parent.name}")
                ax.plot(iters, losses, color=COLORS["hc_unconstrained"],
                        linestyle="--", label=LABELS["hc_unconstrained"])

        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Validation Loss")
        ax.set_title(f"Setting {'A (48-layer, n=4)' if setting == 'A' else 'B (6-layer, n=8)'}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = RESULTS / "validation_loss_curves.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ============================================================
# Phase 2: Gradient Norm Trajectories
# ============================================================
def plot_grad_norm_trajectories(test_only=False):
    print("\n=== Phase 2: Gradient Norm Trajectories ===")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, setting in enumerate(["A", "B"]):
        ax = axes[ax_idx]
        groups = RUN_GROUPS[setting]

        for method in ["sinkhorn", "orthostochastic"]:
            runs = groups[method]
            all_iters, all_smoothed = [], []
            for run in runs:
                iters, norms = load_grad_norms(run)
                print(f"  {run}: {len(norms)} grad norm points")
                smoothed = moving_average(norms, window=200)
                trimmed_iters = iters[199:]
                all_iters.append(trimmed_iters)
                all_smoothed.append(smoothed)
                if test_only:
                    break

            if not all_iters:
                continue
            if len(all_iters) == 1:
                ax.plot(all_iters[0], all_smoothed[0], color=COLORS[method], label=LABELS[method], linewidth=0.8)
            else:
                iters, mean, std = aggregate_grad_norms(all_iters, all_smoothed)
                ax.plot(iters, mean, color=COLORS[method], label=LABELS[method], linewidth=0.8)
                ax.fill_between(iters, np.maximum(mean - std, 1e-6), mean + std, alpha=0.2, color=COLORS[method])

        if "hc_unconstrained" in groups:
            run = groups["hc_unconstrained"][0]
            iters, norms = load_grad_norms(run)
            print(f"  {run}: {len(norms)} grad norm points")
            smoothed = moving_average(norms, window=200)
            trimmed_iters = iters[199:]
            ax.plot(trimmed_iters, smoothed, color=COLORS["hc_unconstrained"],
                    linestyle="--", label=LABELS["hc_unconstrained"], linewidth=0.8)

        ax.set_yscale("log")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Gradient Norm (log scale)")
        ax.set_title(f"Setting {'A (48-layer, n=4)' if setting == 'A' else 'B (6-layer, n=8)'}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    out_path = RESULTS / "gradient_norm_trajectories.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ============================================================
# Phase 3: H_res Heatmaps
# ============================================================
def load_h_res_for_layers(run_dir_name, method, setting, layers):
    ckpt_path = LOGS / run_dir_name / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    h_res_matrices = {}
    for layer_idx in layers:
        key = f"transformer.h.{layer_idx}.hc_attn.H_res_logits"
        alpha_key = f"transformer.h.{layer_idx}.hc_attn.H_res_alpha_logit"
        logits = sd[key]
        alpha_logit = sd.get(alpha_key, None)
        print(f"    Layer {layer_idx}: H_res_logits shape={logits.shape}" +
              (f", alpha_logit={alpha_logit.item():.4f}" if alpha_logit is not None else ""))
        h_res = project_h_res(logits, method, setting, alpha_logit)
        h_res_matrices[layer_idx] = h_res
    return h_res_matrices


def plot_h_res_heatmaps():
    print("\n=== Phase 3: H_res Heatmaps ===")

    configs = [
        {
            "setting": "A",
            "layers": HEATMAP_LAYERS_A,
            "sinkhorn_run": "setting_a_sinkhorn_seed1",
            "ortho_run": "setting_a_orthostochastic_optimized_seed1",
        },
        {
            "setting": "B",
            "layers": HEATMAP_LAYERS_B,
            "sinkhorn_run": "setting_b_sinkhorn_seed1",
            "ortho_run": "setting_b_orthostochastic_optimized_seed2",
        },
    ]

    total_rows = sum(len(c["layers"]) for c in configs)
    fig, axes = plt.subplots(total_rows, 2, figsize=(7, 2.2 * total_rows))
    if total_rows == 1:
        axes = axes.reshape(1, 2)

    row = 0
    for cfg in configs:
        setting = cfg["setting"]
        layers = cfg["layers"]
        n_layers_total = 48 if setting == "A" else 6

        print(f"  Setting {setting} - Sinkhorn: {cfg['sinkhorn_run']}")
        sink_matrices = load_h_res_for_layers(cfg["sinkhorn_run"], "sinkhorn", setting, layers)
        print(f"  Setting {setting} - Orthostochastic: {cfg['ortho_run']}")
        ortho_matrices = load_h_res_for_layers(cfg["ortho_run"], "orthostochastic", setting, layers)

        for layer_idx in layers:
            for col, (method_label, mat) in enumerate([
                ("Sinkhorn", sink_matrices[layer_idx]),
                ("Orthostochastic", ortho_matrices[layer_idx]),
            ]):
                ax = axes[row, col]
                im = ax.imshow(mat, cmap="viridis", aspect="equal",
                               vmin=0, vmax=max(mat.max(), 0.5))
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                layer_1idx = layer_idx + 1
                ax.set_title(f"Set {setting} L{layer_1idx}/{n_layers_total} - {method_label}", fontsize=9)
                ax.set_xlabel("Target stream")
                ax.set_ylabel("Source stream")
                n = mat.shape[0]
                ax.set_xticks(range(n))
                ax.set_yticks(range(n))
            row += 1

    fig.tight_layout()
    out_path = RESULTS / "h_res_heatmaps.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    test_only = "--test" in sys.argv

    if test_only:
        print("=== TEST MODE: single run per method ===")

    plot_val_loss_curves(test_only=test_only)
    plot_grad_norm_trajectories(test_only=test_only)
    if not test_only:
        plot_h_res_heatmaps()
    else:
        print("\n=== Skipping heatmaps in test mode ===")

    print("\nDone!")
