"""
Diagnostic visualization for RRCS gradient vanishing prevention mechanism.
Produces 6 figures comparing mhc_default, capinit, and RRCS (r_cap=2.0) conditions
from diagnostic CSV logs across 3 seeds (42, 123, 456).

Figures:
  1. hres_grad_norm_timeseries.png - H_res gradient norm over training
  2. sinkhorn_log_range_distribution.png - Violin plots of log-range per condition
  3. sinkhorn_log_range_timeseries.png - Log-range over training steps
  4. hres_param_drift.png - Cumulative parameter drift approximation
  5. global_grad_norm.png - Global gradient norm with smoothing and spike annotation
  6. range_cap_mechanism.png - Theoretical illustration of range capping
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LOGS_DIR = os.path.join(REPO_ROOT, "examples", "nanogpt", "results", "logs")
FIG_DIR = os.path.join(REPO_ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
CONDITIONS = {
    "mhc_default": {"prefix": "mhc_default_seed", "label": "mHC Default (τ=0.05)", "color": "#d62728"},
    "capinit": {"prefix": "capinit_seed", "label": "Fixed τ Cap-Init", "color": "#ff7f0e"},
    "rrcs": {"prefix": "rrcs_opt_rcap2p0_seed", "label": "RRCS (r_cap=2.0)", "color": "#2ca02c"},
}
N_LAYERS = 96
PRIMARY_SEED = 42

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def load_csv(cond_key, seed):
    cfg = CONDITIONS[cond_key]
    path = os.path.join(LOGS_DIR, f"{cfg['prefix']}{seed}", "diagnostics.csv")
    return pd.read_csv(path)


def load_all():
    data = {}
    for cond in CONDITIONS:
        data[cond] = {}
        for seed in SEEDS:
            data[cond][seed] = load_csv(cond, seed)
    return data


def plot_hres_grad_norm_timeseries(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    for cond, cfg in CONDITIONS.items():
        dfs = [data[cond][s] for s in SEEDS]
        iters = dfs[0]['iter'].values

        grad_cols = [f'h_res_grad_L{i}' for i in range(N_LAYERS)]
        medians = []
        for df in dfs:
            layer_grads = df[grad_cols].values
            median_per_step = np.median(layer_grads, axis=1)
            medians.append(median_per_step)

        medians = np.array(medians)
        floor = 1e-25
        medians_clipped = np.maximum(medians, floor)

        primary_idx = SEEDS.index(PRIMARY_SEED)
        primary = medians_clipped[primary_idx]
        lo = medians_clipped.min(axis=0)
        hi = medians_clipped.max(axis=0)

        ax.semilogy(iters, primary, color=cfg['color'], label=cfg['label'], linewidth=1.5)
        ax.fill_between(iters, lo, hi, color=cfg['color'], alpha=0.15)

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Median ||∇H_res_logits||₂ across layers")
    ax.set_title("H_res Gradient Norm Time Series")
    ax.legend(loc='best')
    ax.set_xlim(0, 5000)
    ax.grid(True, alpha=0.3, which='both')

    path = os.path.join(FIG_DIR, "hres_grad_norm_timeseries.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_sinkhorn_log_range_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    range_cols = [f'sinkhorn_range_L{i}' for i in range(N_LAYERS)]
    plot_data = []
    labels = []

    for cond in ["mhc_default", "capinit", "rrcs"]:
        cfg = CONDITIONS[cond]
        all_vals = []
        for seed in SEEDS:
            df = data[cond][seed]
            vals = df[range_cols].values.flatten()
            all_vals.append(vals)
        all_vals = np.concatenate(all_vals)
        plot_data.append(all_vals)
        labels.append(cfg['label'])

    rrcs_capped = []
    rrcs_s_cols = [f'rrcs_s_L{i}' for i in range(N_LAYERS)]
    for seed in SEEDS:
        df = data["rrcs"][seed]
        r_vals = df[range_cols].values
        s_vals = df[rrcs_s_cols].values
        capped = r_vals * s_vals
        rrcs_capped.append(capped.flatten())
    rrcs_capped = np.concatenate(rrcs_capped)
    plot_data.append(rrcs_capped)
    labels.append("RRCS (post-cap)")

    positions = [1, 2, 3, 4]
    colors = [CONDITIONS["mhc_default"]["color"], CONDITIONS["capinit"]["color"],
              CONDITIONS["rrcs"]["color"], "#1f77b4"]

    parts = ax.violinplot(plot_data, positions=positions, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')

    ax.axhline(y=30, color='gray', linestyle='--', linewidth=1.5, label='r = 30 (cap threshold)')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel("Log-Range r = max(Z) - min(Z)")
    ax.set_title("Sinkhorn Input Log-Range Distribution")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(FIG_DIR, "sinkhorn_log_range_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_sinkhorn_log_range_timeseries(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    range_cols = [f'sinkhorn_range_L{i}' for i in range(N_LAYERS)]

    for cond, cfg in CONDITIONS.items():
        dfs = [data[cond][s] for s in SEEDS]
        iters = dfs[0]['iter'].values

        means_per_seed = []
        for df in dfs:
            mean_across_layers = df[range_cols].mean(axis=1).values
            means_per_seed.append(mean_across_layers)
        means_per_seed = np.array(means_per_seed)

        primary_idx = SEEDS.index(PRIMARY_SEED)
        primary = means_per_seed[primary_idx]
        lo = means_per_seed.min(axis=0)
        hi = means_per_seed.max(axis=0)

        ax.plot(iters, primary, color=cfg['color'], label=cfg['label'], linewidth=1.5)
        ax.fill_between(iters, lo, hi, color=cfg['color'], alpha=0.15)

    rrcs_s_cols = [f'rrcs_s_L{i}' for i in range(N_LAYERS)]
    rrcs_capped_per_seed = []
    for seed in SEEDS:
        df = data["rrcs"][seed]
        r_vals = df[range_cols].values
        s_vals = df[rrcs_s_cols].values
        capped = (r_vals * s_vals).mean(axis=1)
        rrcs_capped_per_seed.append(capped)
    rrcs_capped_per_seed = np.array(rrcs_capped_per_seed)
    iters = data["rrcs"][PRIMARY_SEED]['iter'].values
    primary_capped = rrcs_capped_per_seed[SEEDS.index(PRIMARY_SEED)]
    ax.plot(iters, primary_capped, color='#1f77b4', label='RRCS (post-cap effective)', linewidth=1.5, linestyle='--')
    ax.fill_between(iters, rrcs_capped_per_seed.min(axis=0), rrcs_capped_per_seed.max(axis=0),
                    color='#1f77b4', alpha=0.10)

    ax.axhline(y=30, color='gray', linestyle=':', linewidth=1.5, label='r = 30')
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Mean Log-Range r across layers")
    ax.set_title("Sinkhorn Input Log-Range Over Training")
    ax.legend(loc='right')
    ax.set_xlim(0, 5000)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "sinkhorn_log_range_timeseries.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_hres_param_drift(data):
    fig, ax = plt.subplots(figsize=(10, 5))

    grad_cols = [f'h_res_grad_L{i}' for i in range(N_LAYERS)]

    for cond, cfg in CONDITIONS.items():
        dfs = [data[cond][s] for s in SEEDS]
        iters = dfs[0]['iter'].values
        step_size = 10

        drift_per_seed = []
        for df in dfs:
            layer_grads = df[grad_cols].values
            per_step_norm = np.sqrt((layer_grads ** 2).sum(axis=1))
            cumulative = np.cumsum(per_step_norm) * step_size
            drift_per_seed.append(cumulative)
        drift_per_seed = np.array(drift_per_seed)

        primary_idx = SEEDS.index(PRIMARY_SEED)
        primary = drift_per_seed[primary_idx]
        lo = drift_per_seed.min(axis=0)
        hi = drift_per_seed.max(axis=0)

        ax.plot(iters, primary, color=cfg['color'], label=cfg['label'], linewidth=1.5)
        ax.fill_between(iters, lo, hi, color=cfg['color'], alpha=0.15)

    final_drifts = {}
    for cond in CONDITIONS:
        drifts = []
        for seed in SEEDS:
            seed_dir = os.path.join(LOGS_DIR, f"{CONDITIONS[cond]['prefix']}{seed}")
            init_path = os.path.join(seed_dir, "h_res_logits_init.pt")
            ckpt_path = os.path.join(seed_dir, "ckpt.pt")
            if os.path.exists(init_path) and os.path.exists(ckpt_path):
                init_params = torch.load(init_path, map_location='cpu', weights_only=True)
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model_state = ckpt['model']
                total_drift_sq = 0.0
                for key, init_val in init_params.items():
                    if key in model_state:
                        diff = model_state[key].float() - init_val.float()
                        total_drift_sq += (diff ** 2).sum().item()
                drifts.append(total_drift_sq ** 0.5)
        if drifts:
            final_drifts[cond] = np.mean(drifts)

    y_max = ax.get_ylim()[1]
    for cond, cfg in CONDITIONS.items():
        if cond in final_drifts and final_drifts[cond] > 0:
            ax.annotate(f"Final Fro drift: {final_drifts[cond]:.2f}",
                        xy=(4800, y_max * 0.9),
                        fontsize=9, color=cfg['color'],
                        ha='right')
            y_max *= 0.82

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Cumulative ||∇H_res|| · Δt (parameter drift proxy)")
    ax.set_title("H_res Parameter Drift Over Training")
    ax.legend(loc='upper left')
    ax.set_xlim(0, 5000)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "hres_param_drift.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_global_grad_norm(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    window = 20

    annotations = []
    for cond, cfg in CONDITIONS.items():
        dfs = [data[cond][s] for s in SEEDS]
        iters = dfs[0]['iter'].values

        gnorms_per_seed = []
        for df in dfs:
            gnorms_per_seed.append(df['grad_norm_global'].values)
        gnorms_per_seed = np.array(gnorms_per_seed)

        primary_idx = SEEDS.index(PRIMARY_SEED)
        raw = gnorms_per_seed[primary_idx]

        smoothed = pd.Series(raw).rolling(window=window, min_periods=1, center=True).mean().values

        ax.plot(iters, smoothed, color=cfg['color'], label=cfg['label'], linewidth=1.5)
        ax.fill_between(iters,
                        pd.Series(gnorms_per_seed.min(axis=0)).rolling(window=window, min_periods=1, center=True).mean().values,
                        pd.Series(gnorms_per_seed.max(axis=0)).rolling(window=window, min_periods=1, center=True).mean().values,
                        color=cfg['color'], alpha=0.10)

        max_val = raw.max()
        median_val = np.median(raw)
        spike_ratio = max_val / median_val if median_val > 0 else float('inf')
        max_iter = iters[np.argmax(raw)]
        annotations.append((cond, cfg, spike_ratio, max_iter, max_val))

    text_y = 0.95
    for cond, cfg, spike_ratio, max_iter, max_val in annotations:
        ax.annotate(f"{cfg['label']}: r_max={spike_ratio:.2f}",
                    xy=(0.02, text_y), xycoords='axes fraction',
                    fontsize=9, color=cfg['color'], va='top')
        text_y -= 0.06

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Global Gradient Norm")
    ax.set_title(f"Global Gradient Norm (smoothed, window={window} steps)")
    ax.legend(loc='upper right')
    ax.set_xlim(0, 5000)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIG_DIR, "global_grad_norm.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_range_cap_mechanism():
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    Z = np.linspace(-200, 0, 2000)
    exp_Z = np.exp(Z)
    ax1.semilogy(Z, exp_Z, 'b-', linewidth=2)
    ax1.axhline(y=np.exp(-30), color='red', linestyle='--', linewidth=1, label=f'exp(-30) ≈ {np.exp(-30):.1e}')
    ax1.axhline(y=1e-64, color='orange', linestyle=':', linewidth=1, label='float64 underflow zone')
    ax1.fill_between(Z, 1e-100, exp_Z, where=(exp_Z < 1e-30), color='red', alpha=0.1)
    ax1.set_xlabel("Z (Sinkhorn input)")
    ax1.set_ylabel("exp(Z)")
    ax1.set_title("(a) Uncapped: exp(Z) for Z ∈ [-200, 0]\n(τ=0.05, range ≈ 160)")
    ax1.legend(fontsize=8)
    ax1.set_ylim(1e-100, 10)
    ax1.grid(True, alpha=0.3, which='both')

    ax2 = fig.add_subplot(2, 2, 2)
    r_original = 200
    r_cap = 30
    s = min(r_cap / r_original, 1.0)
    Z_capped = Z * s
    exp_Z_capped = np.exp(Z_capped)
    ax2.semilogy(Z, exp_Z_capped, 'g-', linewidth=2, label=f'exp(Z·s), s={s:.3f}')
    ax2.semilogy(Z, exp_Z, 'b-', linewidth=1, alpha=0.3, label='Original exp(Z)')
    ax2.axhline(y=np.exp(-r_cap), color='red', linestyle='--', linewidth=1,
                label=f'exp(-{r_cap}) ≈ {np.exp(-r_cap):.1e}')
    ax2.set_xlabel("Z (original Sinkhorn input)")
    ax2.set_ylabel("exp(Z')")
    ax2.set_title(f"(b) Range Capped (r_cap={r_cap})\nZ' = Z·s, s = r_cap/r = {s:.3f}")
    ax2.legend(fontsize=8)
    ax2.set_ylim(1e-100, 10)
    ax2.grid(True, alpha=0.3, which='both')

    ax3 = fig.add_subplot(2, 2, 3)
    r_cap2 = 2.0
    s2 = min(r_cap2 / r_original, 1.0)
    Z_capped2 = Z * s2
    exp_Z_capped2 = np.exp(Z_capped2)
    ax3.semilogy(Z, exp_Z_capped2, color='#2ca02c', linewidth=2, label=f'exp(Z·s), s={s2:.4f}')
    ax3.semilogy(Z, exp_Z, 'b-', linewidth=1, alpha=0.3, label='Original exp(Z)')
    ax3.axhline(y=np.exp(-r_cap2), color='red', linestyle='--', linewidth=1,
                label=f'exp(-{r_cap2}) ≈ {np.exp(-r_cap2):.3f}')
    ax3.set_xlabel("Z (original Sinkhorn input)")
    ax3.set_ylabel("exp(Z')")
    ax3.set_title(f"(c) RRCS (r_cap={r_cap2})\nZ' = Z·s, s = {s2:.4f}")
    ax3.legend(fontsize=8)
    ax3.set_ylim(1e-5, 10)
    ax3.grid(True, alpha=0.3, which='both')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    chain_text = (
        "Gradient Chain Through Sinkhorn:\n\n"
        "  ∂L/∂(H_res_logits)\n"
        "    = ∂L/∂H_res · ∂H_res/∂P · ∂P/∂Z · ∂Z/∂(logits) · (1/τ)\n\n"
        "Where:\n"
        "  Z = logits / τ     (τ = 0.05 → Z amplified 20×)\n"
        "  P = Sinkhorn(exp(Z))  (doubly-stochastic)\n"
        "  H_res = P (learned routing matrix)\n\n"
        "Problem (uncapped):\n"
        "  range(Z) ≈ 160 → exp(Z_min) ≈ exp(-160) ≈ 0\n"
        "  → P becomes a hard permutation\n"
        "  → ∂P/∂Z ≈ 0 everywhere\n"
        "  → Gradients vanish completely\n\n"
        "Solution (RRCS, r_cap=2.0):\n"
        "  Z' = Z · s,  s = min(r_cap/range(Z), 1)\n"
        "  range(Z') ≤ 2.0 → exp(Z'_min) ≥ exp(-2) ≈ 0.135\n"
        "  → P stays soft (entropy ≈ 0.93)\n"
        "  -> dP/dZ' >> 0\n"
        "  → Gradients flow into H_res_logits"
    )
    ax4.text(0.05, 0.95, chain_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    ax4.set_title("(d) Gradient Chain Mechanism")

    fig.suptitle("Theoretical Mechanism: Range Capping Prevents Gradient Vanishing", fontsize=14, y=1.01)
    fig.tight_layout()

    path = os.path.join(FIG_DIR, "range_cap_mechanism.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print("Loading diagnostic CSVs...")
    data = load_all()
    print("Data loaded successfully.\n")

    print("--- Figure 1: H_res Gradient Norm Time Series ---")
    plot_hres_grad_norm_timeseries(data)

    print("--- Figure 2a: Sinkhorn Log-Range Distribution ---")
    plot_sinkhorn_log_range_distribution(data)

    print("--- Figure 2b: Sinkhorn Log-Range Time Series ---")
    plot_sinkhorn_log_range_timeseries(data)

    print("--- Figure 3: H_res Parameter Drift ---")
    plot_hres_param_drift(data)

    print("--- Figure 4: Global Gradient Norm ---")
    plot_global_grad_norm(data)

    print("--- Figure 5: Theoretical Range Cap Mechanism ---")
    plot_range_cap_mechanism()

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
