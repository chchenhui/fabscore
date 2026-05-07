# Ablation analysis: compare overlap-resampled L-BFGS with o=0.25 vs o=0.5.
# Generates comparison table, training loss curves, and curvature quality bar chart.
# Usage: python -m overlap_lbfgs_pinn.scripts.analyze_ablation_o025

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_O05 = os.path.join(BASE_DIR, "outputs", "overlap_lbfgs_ice_shelf")
DIR_O025 = os.path.join(BASE_DIR, "outputs", "overlap_lbfgs_ice_shelf_o025")
PROJECT_DIR = os.path.dirname(BASE_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "EXPERIMENT_RESULTS", "ablation_o025")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]


def load_metrics(dir_path, seeds):
    results = []
    for s in seeds:
        with open(os.path.join(dir_path, f"seed_{s}_metrics.json")) as f:
            results.append(json.load(f))
    return results


def load_loss_histories(dir_path, seeds):
    histories = []
    for s in seeds:
        with open(os.path.join(dir_path, f"seed_{s}_loss_history.json")) as f:
            histories.append(json.load(f))
    return histories


def print_comparison_table(metrics_o05, metrics_o025):
    print("\n" + "=" * 130)
    print("COMPARISON TABLE: o=0.5 vs o=0.25")
    print("=" * 130)
    header = (f"{'Overlap':>8} | {'B_err (mean+-std)':>24} | {'u_err (mean+-std)':>24} | "
              f"{'h_err (mean+-std)':>24} | {'L-BFGS Iters':>13} | {'Cautious Skips':>17} | {'Early Stop?':>14}")
    print(header)
    print("-" * 130)

    for label, metrics in [("o=0.50", metrics_o05), ("o=0.25", metrics_o025)]:
        B = [m["final_errors"]["B_err"] for m in metrics]
        u = [m["final_errors"]["u_err"] for m in metrics]
        h = [m["final_errors"]["h_err"] for m in metrics]
        iters = [m["lbfgs_diagnostics"]["lbfgs_outer_steps"] for m in metrics]
        skips = [m["lbfgs_diagnostics"]["cautious_skips"] for m in metrics]
        terms = [m["lbfgs_diagnostics"]["termination_reason"] for m in metrics]

        early = any(t != "budget_exhausted" for t in terms)
        early_str = "Yes (" + ",".join(t for t in terms if t != "budget_exhausted") + ")" if early else "No"
        skip_rates = [sk / it * 100 if it > 0 else 0 for sk, it in zip(skips, iters)]

        print(f"{label:>8} | {np.mean(B):.4e}+-{np.std(B):.4e} | "
              f"{np.mean(u):.4e}+-{np.std(u):.4e} | "
              f"{np.mean(h):.4e}+-{np.std(h):.4e} | "
              f"{np.mean(iters):>10.0f}+-{np.std(iters):>3.0f} | "
              f"{np.mean(skips):>7.0f} ({np.mean(skip_rates):>4.1f}%) | "
              f"{early_str:>14}")

    print("\n--- Per-seed details ---")
    for label, metrics in [("o=0.50", metrics_o05), ("o=0.25", metrics_o025)]:
        print(f"\n  {label}:")
        for m in metrics:
            d = m["lbfgs_diagnostics"]
            skip_rate = d["cautious_skips"] / d["lbfgs_outer_steps"] * 100 if d["lbfgs_outer_steps"] > 0 else 0
            print(f"    Seed {m['seed']}: B_err={m['final_errors']['B_err']:.4e}, "
                  f"u_err={m['final_errors']['u_err']:.4e}, "
                  f"h_err={m['final_errors']['h_err']:.4e}, "
                  f"best_step={m['best_step']}, "
                  f"lbfgs_steps={d['lbfgs_outer_steps']}, "
                  f"cautious_skips={d['cautious_skips']} ({skip_rate:.1f}%), "
                  f"ls_failures={d['line_search_failures']}, "
                  f"total_evals={m['total_evals']}, "
                  f"termination={d['termination_reason']}")


def plot_training_loss(hist_o05, hist_o025, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {"o=0.50": "tab:blue", "o=0.25": "tab:orange"}

    for label, histories, color in [("o=0.50", hist_o05, colors["o=0.50"]),
                                     ("o=0.25", hist_o025, colors["o=0.25"])]:
        all_steps = []
        all_losses = []
        for si, hist in enumerate(histories):
            steps = [e["step"] for e in hist]
            losses = [e["total_loss"] for e in hist]
            ax.plot(steps, losses, color=color, alpha=0.2, linewidth=0.5)
            all_steps.append(np.array(steps))
            all_losses.append(np.array(losses))

        min_len = min(len(s) for s in all_steps)
        mean_steps = all_steps[0][:min_len]
        mean_losses = np.mean([l[:min_len] for l in all_losses], axis=0)
        ax.plot(mean_steps, mean_losses, color=color, linewidth=2, label=f"{label} (mean)")

    ax.set_xlabel("Budget (gradient evaluations)", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("Training Loss: Overlap Ratio Ablation (o=0.25 vs o=0.50)", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "training_loss_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lbfgs_phase_loss(hist_o05, hist_o025, out_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {"o=0.50": "tab:blue", "o=0.25": "tab:orange"}

    for label, histories, color in [("o=0.50", hist_o05, colors["o=0.50"]),
                                     ("o=0.25", hist_o025, colors["o=0.25"])]:
        for si, hist in enumerate(histories):
            lbfgs = [e for e in hist if e.get("phase") == "overlap_lbfgs"]
            if not lbfgs:
                continue
            steps = list(range(1, len(lbfgs) + 1))
            losses = [e["total_loss"] for e in lbfgs]
            ax.plot(steps, losses, color=color, alpha=0.3, linewidth=0.7)

        all_losses = []
        for hist in histories:
            lbfgs = [e for e in hist if e.get("phase") == "overlap_lbfgs"]
            if lbfgs:
                all_losses.append([e["total_loss"] for e in lbfgs])

        if all_losses:
            min_len = min(len(l) for l in all_losses)
            mean_loss = np.mean([l[:min_len] for l in all_losses], axis=0)
            ax.plot(range(1, min_len + 1), mean_loss, color=color, linewidth=2, label=f"{label} (mean)")

    ax.set_xlabel("L-BFGS Iteration", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("L-BFGS Phase Loss: o=0.25 vs o=0.50", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "lbfgs_phase_loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cautious_skip_bar(metrics_o05, metrics_o025, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    seeds = [0, 1, 2]
    x = np.arange(len(seeds))
    width = 0.35

    skips_o05 = [m["lbfgs_diagnostics"]["cautious_skips"] for m in metrics_o05]
    skips_o025 = [m["lbfgs_diagnostics"]["cautious_skips"] for m in metrics_o025]
    ax1.bar(x - width / 2, skips_o05, width, label="o=0.50", color="tab:blue")
    ax1.bar(x + width / 2, skips_o025, width, label="o=0.25", color="tab:orange")
    ax1.set_xlabel("Seed", fontsize=12)
    ax1.set_ylabel("Cautious Skips (count)", fontsize=12)
    ax1.set_title("Cautious Update Skips by Seed", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    rates_o05 = [m["lbfgs_diagnostics"]["cautious_skips"] / m["lbfgs_diagnostics"]["lbfgs_outer_steps"] * 100
                 if m["lbfgs_diagnostics"]["lbfgs_outer_steps"] > 0 else 0 for m in metrics_o05]
    rates_o025 = [m["lbfgs_diagnostics"]["cautious_skips"] / m["lbfgs_diagnostics"]["lbfgs_outer_steps"] * 100
                  if m["lbfgs_diagnostics"]["lbfgs_outer_steps"] > 0 else 0 for m in metrics_o025]
    ax2.bar(x - width / 2, rates_o05, width, label="o=0.50", color="tab:blue")
    ax2.bar(x + width / 2, rates_o025, width, label="o=0.25", color="tab:orange")
    ax2.set_xlabel("Seed", fontsize=12)
    ax2.set_ylabel("Cautious Skip Rate (%)", fontsize=12)
    ax2.set_title("Cautious Skip Rate by Seed", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "cautious_skip_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    metrics_o05 = load_metrics(DIR_O05, SEEDS)
    metrics_o025 = load_metrics(DIR_O025, SEEDS)
    print_comparison_table(metrics_o05, metrics_o025)

    hist_o05 = load_loss_histories(DIR_O05, SEEDS)
    hist_o025 = load_loss_histories(DIR_O025, SEEDS)
    plot_training_loss(hist_o05, hist_o025, OUT_DIR)
    plot_lbfgs_phase_loss(hist_o05, hist_o025, OUT_DIR)
    plot_cautious_skip_bar(metrics_o05, metrics_o025, OUT_DIR)

    print("\nAnalysis complete. Figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
