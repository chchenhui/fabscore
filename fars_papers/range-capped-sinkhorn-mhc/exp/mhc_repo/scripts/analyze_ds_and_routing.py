"""
Analysis script for DS error, H_res routing patterns, and product stability.
Produces 4 figures comparing mHC default, Cap-init, and RRCS (r_cap=2.0) conditions.

96 HC modules per model (48 transformer layers x 2: hc_attn even, hc_mlp odd).
Uses DiagnosticLogger's sinkhorn normalization (log_marginal=0, no *n scaling)
for consistency with CSV values.
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / "examples" / "nanogpt" / "results" / "logs"
FIGS = BASE / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

NUM_HC = 96
NUM_TRANSFORMER_LAYERS = 48

CONDITIONS = {
    "mHC default": {
        "dir": "mhc_default_seed42",
        "tau": 0.05,
        "r_cap": None,
        "color": "#1f77b4",
    },
    "Cap-init": {
        "dir": "capinit_seed42",
        "tau": 0.2667,
        "r_cap": None,
        "color": "#ff7f0e",
    },
    "RRCS (r=2)": {
        "dir": "rrcs_opt_rcap2p0_seed42",
        "tau": 0.05,
        "r_cap": 2.0,
        "color": "#2ca02c",
    },
}


def sinkhorn_log(logits, num_iters=10, tau=0.05, r_cap=None):
    n = logits.shape[-1]
    Z = logits / tau
    if r_cap is not None:
        r = Z.max() - Z.min()
        s = torch.clamp(r_cap / (r + 1e-8), max=1.0)
        Z = s * Z
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)
    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


def load_csv(cond_dir):
    path = LOGS / cond_dir / "diagnostics.csv"
    return pd.read_csv(path)


def load_h_res_logits(cond_dir):
    path = LOGS / cond_dir / "ckpt.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    logits_list = []
    for layer_idx in range(NUM_TRANSFORMER_LAYERS):
        for sub in ("hc_attn", "hc_mlp"):
            key = f"transformer.h.{layer_idx}.{sub}.H_res_logits"
            logits_list.append(state[key].float())
    return logits_list


def compute_all_hres(logits_list, tau, r_cap):
    hres_list = []
    for logits in logits_list:
        hres_list.append(sinkhorn_log(logits, num_iters=10, tau=tau, r_cap=r_cap))
    return hres_list


def row_entropy(H):
    n = H.shape[-1]
    return -(H * torch.log(H + 1e-10)).sum() / n


# ---------------------------------------------------------------------------
# Step 1: DS Error Comparison
# ---------------------------------------------------------------------------

def plot_ds_error_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Box plots at step 5000
    ax = axes[0]
    box_data = []
    labels = []
    positions = []
    colors_list = []
    pos = 0
    for cname, cinfo in CONDITIONS.items():
        df = load_csv(cinfo["dir"])
        row5000 = df[df["iter"] == 5000].iloc[0]
        row_errs = [row5000[f"ds_row_err_L{i}"] for i in range(NUM_HC)]
        col_errs = [row5000[f"ds_col_err_L{i}"] for i in range(NUM_HC)]
        box_data.append(row_errs)
        labels.append(f"{cname}\nrow")
        positions.append(pos)
        colors_list.append(cinfo["color"])
        pos += 1
        box_data.append(col_errs)
        labels.append(f"{cname}\ncol")
        positions.append(pos)
        colors_list.append(cinfo["color"])
        pos += 1.5

    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, flierprops=dict(markersize=2))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("DS Error (max abs deviation)")
    ax.set_title("(a) Per-Module DS Error at Step 5000")
    ax.set_yscale("symlog", linthresh=1e-10)

    # (b) Time-series of mean DS error
    ax = axes[1]
    for cname, cinfo in CONDITIONS.items():
        df = load_csv(cinfo["dir"])
        mean_ds = (df["ds_row_error_mean"] + df["ds_col_error_mean"]) / 2
        ax.plot(df["iter"], mean_ds, label=cname, color=cinfo["color"], linewidth=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean DS Error")
    ax.set_title("(b) Mean DS Error Over Training")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGS / "ds_error_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / 'ds_error_comparison.png'}")


# ---------------------------------------------------------------------------
# Step 2a: H_res Routing Heatmaps
# ---------------------------------------------------------------------------

def plot_hres_heatmaps():
    selected_layers = [0, 11, 23, 35, 47]
    layer_labels = ["Layer 1", "Layer 12", "Layer 24", "Layer 36", "Layer 48"]
    cond_names = list(CONDITIONS.keys())

    fig, axes = plt.subplots(
        len(cond_names) * 2, len(selected_layers),
        figsize=(3.0 * len(selected_layers), 2.5 * len(cond_names) * 2),
    )

    for ci, cname in enumerate(cond_names):
        cinfo = CONDITIONS[cname]
        logits_list = load_h_res_logits(cinfo["dir"])
        hres_list = compute_all_hres(logits_list, cinfo["tau"], cinfo["r_cap"])

        for li, tl in enumerate(selected_layers):
            attn_idx = tl * 2
            mlp_idx = tl * 2 + 1
            for sub_idx, (idx, sub_label) in enumerate([(attn_idx, "attn"), (mlp_idx, "mlp")]):
                row = ci * 2 + sub_idx
                ax = axes[row, li]
                H = hres_list[idx].detach().numpy()
                sns.heatmap(H, ax=ax, vmin=0, vmax=max(0.5, H.max()),
                            cmap="YlOrRd", annot=True, fmt=".2f",
                            cbar=False, square=True,
                            annot_kws={"fontsize": 6})
                if li == 0:
                    ax.set_ylabel(f"{cname}\n{sub_label}", fontsize=8)
                else:
                    ax.set_ylabel("")
                if ci == 0 and sub_idx == 0:
                    ax.set_title(layer_labels[li], fontsize=9)
                else:
                    ax.set_title("")
                ax.tick_params(labelsize=6)

    fig.suptitle("H_res Routing Matrices (seed=42, step 5000)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGS / "hres_routing_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / 'hres_routing_heatmaps.png'}")


# ---------------------------------------------------------------------------
# Step 2b: H_res Entropy by Layer
# ---------------------------------------------------------------------------

def plot_hres_entropy():
    fig, ax = plt.subplots(figsize=(14, 5))
    for cname, cinfo in CONDITIONS.items():
        logits_list = load_h_res_logits(cinfo["dir"])
        hres_list = compute_all_hres(logits_list, cinfo["tau"], cinfo["r_cap"])
        entropies = [row_entropy(H).item() for H in hres_list]
        ax.plot(range(NUM_HC), entropies, label=cname, color=cinfo["color"],
                linewidth=1, alpha=0.8)

    for tl in range(NUM_TRANSFORMER_LAYERS):
        ax.axvline(x=tl * 2, color="gray", linewidth=0.2, alpha=0.3)

    ax.set_xlabel("HC Module Index (even=attn, odd=mlp)")
    ax.set_ylabel("Mean Row Entropy")
    ax.set_title("H_res Row Entropy by HC Module (seed=42, step 5000)")
    ax.legend(fontsize=9)

    sec_ax = ax.secondary_xaxis("top")
    tick_positions = [tl * 2 + 0.5 for tl in range(0, NUM_TRANSFORMER_LAYERS, 6)]
    tick_labels = [f"TL{tl}" for tl in range(0, NUM_TRANSFORMER_LAYERS, 6)]
    sec_ax.set_xticks(tick_positions)
    sec_ax.set_xticklabels(tick_labels, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGS / "hres_entropy_by_layer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / 'hres_entropy_by_layer.png'}")


# ---------------------------------------------------------------------------
# Step 3: Product Stability
# ---------------------------------------------------------------------------

def plot_product_stability():
    fig, ax = plt.subplots(figsize=(12, 5))
    for cname, cinfo in CONDITIONS.items():
        logits_list = load_h_res_logits(cinfo["dir"])
        hres_list = compute_all_hres(logits_list, cinfo["tau"], cinfo["r_cap"])

        P = torch.eye(4, dtype=torch.float64)
        col_devs = []
        for idx in range(NUM_HC):
            P = P @ hres_list[idx].double()
            col_sums = P.sum(dim=0)
            dev = (col_sums - 1.0).abs().max().item()
            col_devs.append(dev)

        ax.plot(range(1, NUM_HC + 1), col_devs, label=cname,
                color=cinfo["color"], linewidth=1)

    for tl in range(NUM_TRANSFORMER_LAYERS):
        ax.axvline(x=tl * 2 + 1, color="gray", linewidth=0.2, alpha=0.3)

    ax.set_xlabel("Depth (cumulative HC modules)")
    ax.set_ylabel("Max Column-Sum Deviation from 1")
    ax.set_title("H_res Product Stability: Accumulated DS Error vs Depth")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGS / "hres_product_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGS / 'hres_product_stability.png'}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check():
    print("=== Sanity Check ===")
    cinfo = CONDITIONS["RRCS (r=2)"]
    df = load_csv(cinfo["dir"])
    print(f"CSV rows: {len(df)}, max iter: {df['iter'].max()}")
    row_cols = [c for c in df.columns if c.startswith("ds_row_err_L")]
    col_cols = [c for c in df.columns if c.startswith("ds_col_err_L")]
    print(f"Per-layer DS row cols: {len(row_cols)}, col cols: {len(col_cols)}")

    logits_list = load_h_res_logits(cinfo["dir"])
    print(f"H_res_logits loaded: {len(logits_list)}, shape: {logits_list[0].shape}")

    hres_list = compute_all_hres(logits_list, cinfo["tau"], cinfo["r_cap"])

    row5000 = df[df["iter"] == 5000].iloc[0]
    print("\nDS error comparison (recomputed vs CSV) for first 5 modules:")
    all_ok = True
    for i in range(5):
        H = hres_list[i]
        row_err = (H.sum(dim=-1) - 1.0).abs().max().item()
        col_err = (H.sum(dim=-2) - 1.0).abs().max().item()
        csv_row = row5000[f"ds_row_err_L{i}"]
        csv_col = row5000[f"ds_col_err_L{i}"]
        match_row = abs(row_err - csv_row) < max(1e-5, csv_row * 0.1) if csv_row > 0 else row_err < 1e-5
        match_col = abs(col_err - csv_col) < max(1e-5, csv_col * 0.1) if csv_col > 0 else col_err < 1e-5
        status = "OK" if (match_row and match_col) else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"  L{i}: recomp row={row_err:.2e} col={col_err:.2e} | csv row={csv_row:.2e} col={csv_col:.2e} [{status}]")

    if all_ok:
        print("\nSanity check PASSED")
    else:
        print("\nSanity check had MISMATCHES (may be due to float precision or checkpoint vs final step)")
    return all_ok


def collect_results():
    results = {}
    for cname, cinfo in CONDITIONS.items():
        df = load_csv(cinfo["dir"])
        row5000 = df[df["iter"] == 5000].iloc[0]
        row_errs = [row5000[f"ds_row_err_L{i}"] for i in range(NUM_HC)]
        col_errs = [row5000[f"ds_col_err_L{i}"] for i in range(NUM_HC)]

        logits_list = load_h_res_logits(cinfo["dir"])
        hres_list = compute_all_hres(logits_list, cinfo["tau"], cinfo["r_cap"])
        entropies = [row_entropy(H).item() for H in hres_list]

        P = torch.eye(4, dtype=torch.float64)
        final_col_dev = 0.0
        for idx in range(NUM_HC):
            P = P @ hres_list[idx].double()
        col_sums = P.sum(dim=0)
        final_col_dev = (col_sums - 1.0).abs().max().item()

        results[cname] = {
            "ds_row_error_mean_step5000": float(np.mean(row_errs)),
            "ds_row_error_max_step5000": float(np.max(row_errs)),
            "ds_col_error_mean_step5000": float(np.mean(col_errs)),
            "ds_col_error_max_step5000": float(np.max(col_errs)),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_min": float(np.min(entropies)),
            "entropy_max": float(np.max(entropies)),
            "product_col_dev_depth96": final_col_dev,
        }
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "sanity":
        sanity_check()
    elif mode == "all":
        sanity_check()
        print("\n=== Generating Figures ===")
        plot_ds_error_comparison()
        plot_hres_heatmaps()
        plot_hres_entropy()
        plot_product_stability()
        results = collect_results()
        print("\n=== Summary Results ===")
        print(json.dumps(results, indent=2))
    else:
        print(f"Unknown mode: {mode}. Use 'sanity' or 'all'.")
