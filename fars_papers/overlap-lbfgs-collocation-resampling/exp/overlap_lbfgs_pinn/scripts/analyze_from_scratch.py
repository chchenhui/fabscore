"""Analysis: Compare from-scratch overlap-LBFGS vs warmstart variants on 2D Poisson.
Generates comparison table and training loss curve plots.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(BASE), "EXPERIMENT_RESULTS", "from_scratch_overlap_lbfgs_poisson2d")

FROM_SCRATCH_DIR = os.path.join(BASE, "outputs", "from_scratch_overlap_lbfgs_poisson2d")
WARMSTART_DIR = os.path.join(BASE, "outputs", "overlap_lbfgs_poisson2d")


def load_eval_history(output_dir, seeds=(0, 1, 2)):
    histories = {}
    for seed in seeds:
        path = os.path.join(output_dir, f"seed_{seed}_eval_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[seed] = json.load(f)
    return histories


def load_loss_history(output_dir, seeds=(0, 1, 2)):
    histories = {}
    for seed in seeds:
        path = os.path.join(output_dir, f"seed_{seed}_loss_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[seed] = json.load(f)
    return histories


def build_comparison_table():
    with open(os.path.join(FROM_SCRATCH_DIR, "summary.json")) as f:
        scratch = json.load(f)

    warmstart_results = os.path.join(os.path.dirname(BASE), "EXPERIMENT_RESULTS", "overlap_lbfgs_poisson2d", "RESULTS.json")
    with open(warmstart_results) as f:
        warmstart = json.load(f)

    fixed_lbfgs_results = os.path.join(os.path.dirname(BASE), "EXPERIMENT_RESULTS", "adam_lbfgs_poisson2d", "RESULTS.json")
    with open(fixed_lbfgs_results) as f:
        fixed_lbfgs = json.load(f)

    scratch_seeds = scratch["per_seed_results"]
    scratch_lbfgs_iters = [s["lbfgs_diagnostics"]["lbfgs_outer_steps"] for s in scratch_seeds]
    scratch_cautious = [s["lbfgs_diagnostics"]["cautious_skips"] for s in scratch_seeds]
    scratch_termination = [s["lbfgs_diagnostics"]["termination_reason"] for s in scratch_seeds]
    scratch_ls_failures = [s["lbfgs_diagnostics"]["line_search_failures"] for s in scratch_seeds]

    warmstart_seeds = warmstart["per_seed_results"]
    warmstart_lbfgs_iters = [s["lbfgs_outer_steps"] for s in warmstart_seeds]
    warmstart_cautious = [s["cautious_skips"] for s in warmstart_seeds]
    warmstart_termination = [s["termination"] for s in warmstart_seeds]

    fixed_seeds = fixed_lbfgs["per_seed_results"]
    fixed_lbfgs_iters = [s["lbfgs_outer_steps"] for s in fixed_seeds]
    fixed_termination = [s["termination"] for s in fixed_seeds]

    table = {
        "from_scratch_overlap_lbfgs": {
            "rel_l2_mean": scratch["summary"]["rel_l2_mean"],
            "rel_l2_std": scratch["summary"]["rel_l2_std"],
            "per_seed_rel_l2": [s["final_rel_l2"] for s in scratch_seeds],
            "lbfgs_iters_mean": float(np.mean(scratch_lbfgs_iters)),
            "lbfgs_iters_std": float(np.std(scratch_lbfgs_iters)),
            "lbfgs_iters_per_seed": scratch_lbfgs_iters,
            "cautious_skips_mean": float(np.mean(scratch_cautious)),
            "cautious_skips_std": float(np.std(scratch_cautious)),
            "cautious_skips_per_seed": scratch_cautious,
            "ls_failures_per_seed": scratch_ls_failures,
            "termination_per_seed": scratch_termination,
        },
        "warmstart_overlap_lbfgs": {
            "rel_l2_mean": warmstart["summary"]["rel_l2_mean"],
            "rel_l2_std": warmstart["summary"]["rel_l2_std"],
            "per_seed_rel_l2": [s["best_rel_l2"] for s in warmstart_seeds],
            "lbfgs_iters_mean": float(np.mean(warmstart_lbfgs_iters)),
            "lbfgs_iters_std": float(np.std(warmstart_lbfgs_iters)),
            "lbfgs_iters_per_seed": warmstart_lbfgs_iters,
            "cautious_skips_mean": float(np.mean(warmstart_cautious)),
            "cautious_skips_std": float(np.std(warmstart_cautious)),
            "cautious_skips_per_seed": warmstart_cautious,
            "termination_per_seed": warmstart_termination,
        },
        "adam_then_fixed_lbfgs": {
            "rel_l2_mean": fixed_lbfgs["summary"]["rel_l2_mean"],
            "rel_l2_std": fixed_lbfgs["summary"]["rel_l2_std"],
            "per_seed_rel_l2": [s["best_rel_l2"] for s in fixed_seeds],
            "lbfgs_iters_mean": float(np.mean(fixed_lbfgs_iters)),
            "lbfgs_iters_std": float(np.std(fixed_lbfgs_iters)),
            "lbfgs_iters_per_seed": fixed_lbfgs_iters,
            "cautious_skips_mean": 0.0,
            "cautious_skips_std": 0.0,
            "cautious_skips_per_seed": [0, 0, 0],
            "termination_per_seed": fixed_termination,
        },
    }
    return table


def plot_training_curves(output_path):
    scratch_evals = load_eval_history(FROM_SCRATCH_DIR)
    warmstart_evals = load_eval_history(WARMSTART_DIR)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    colors_scratch = ["#d62728", "#ff7f0e", "#e377c2"]
    colors_warmstart = ["#1f77b4", "#2ca02c", "#9467bd"]

    for i, seed in enumerate([0, 1, 2]):
        if seed in scratch_evals:
            steps = [e["step"] for e in scratch_evals[seed]]
            vals = [e["rel_l2"] for e in scratch_evals[seed]]
            vals_clipped = [min(v, 1.0) for v in vals]
            ax1.plot(steps, vals_clipped, color=colors_scratch[i], alpha=0.7,
                     label=f"From-scratch s{seed}" if i == 0 else f"_s{seed}",
                     linestyle="--")

        if seed in warmstart_evals:
            steps = [e["step"] for e in warmstart_evals[seed]]
            vals = [e["rel_l2"] for e in warmstart_evals[seed]]
            ax1.plot(steps, vals, color=colors_warmstart[i], alpha=0.7,
                     label=f"Warmstart s{seed}" if i == 0 else f"_s{seed}",
                     linestyle="-")

    ax1.set_yscale("log")
    ax1.set_xlabel("Gradient Evaluations (Budget)")
    ax1.set_ylabel("Relative L2 Error")
    ax1.set_title("From-Scratch vs Warmstart Overlap-LBFGS (2D Poisson)")
    ax1.legend(["From-scratch (3 seeds)", "Warmstart (3 seeds)"], loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-4, 1.0)

    scratch_losses = load_loss_history(FROM_SCRATCH_DIR)
    warmstart_losses = load_loss_history(WARMSTART_DIR)

    ax2 = axes[1]
    for i, seed in enumerate([0, 1, 2]):
        if seed in scratch_losses:
            data = scratch_losses[seed]
            steps = [e["step"] for e in data[::10]]
            vals = [e["total_loss"] for e in data[::10]]
            vals_clipped = [min(v, 1e3) for v in vals]
            ax2.plot(steps, vals_clipped, color=colors_scratch[i], alpha=0.5,
                     linewidth=0.5, linestyle="--")

        if seed in warmstart_losses:
            data = warmstart_losses[seed]
            steps = [e["step"] for e in data[::10]]
            vals = [e["total_loss"] for e in data[::10]]
            ax2.plot(steps, vals, color=colors_warmstart[i], alpha=0.5,
                     linewidth=0.5, linestyle="-")

    ax2.set_yscale("log")
    ax2.set_xlabel("Gradient Evaluations (Budget)")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss: From-Scratch vs Warmstart")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    table = build_comparison_table()

    with open(os.path.join(RESULTS_DIR, "comparison_table.json"), "w") as f:
        json.dump(table, f, indent=2)

    print("\n" + "=" * 100)
    print("COMPARISON TABLE: From-Scratch vs Warmstart Overlap-LBFGS on 2D Poisson")
    print("=" * 100)
    print(f"{'Configuration':<45} {'Rel L2 (mean +/- std)':<25} {'LBFGS Iters':<20} {'Cautious Skips':<20} {'Termination'}")
    print("-" * 130)

    for key, label in [
        ("from_scratch_overlap_lbfgs", "Overlap-LBFGS from scratch"),
        ("warmstart_overlap_lbfgs", "Overlap-LBFGS + warmstart"),
        ("adam_then_fixed_lbfgs", "Adam -> fixed-LBFGS"),
    ]:
        r = table[key]
        rel_l2_str = f"{r['rel_l2_mean']:.2e} +/- {r['rel_l2_std']:.2e}"
        iters_str = f"{r['lbfgs_iters_mean']:.0f} +/- {r['lbfgs_iters_std']:.0f}"
        cautious_str = f"{r['cautious_skips_mean']:.0f} +/- {r['cautious_skips_std']:.0f}"
        terms = set(r["termination_per_seed"])
        term_str = ", ".join(terms)
        print(f"{label:<45} {rel_l2_str:<25} {iters_str:<20} {cautious_str:<20} {term_str}")

    print()
    print("Per-seed details (from-scratch):")
    scratch = table["from_scratch_overlap_lbfgs"]
    for i in range(3):
        print(f"  Seed {i}: rel_l2={scratch['per_seed_rel_l2'][i]:.4e}, "
              f"iters={scratch['lbfgs_iters_per_seed'][i]}, "
              f"cautious={scratch['cautious_skips_per_seed'][i]}, "
              f"ls_failures={scratch['ls_failures_per_seed'][i]}, "
              f"termination={scratch['termination_per_seed'][i]}")

    plot_path = os.path.join(RESULTS_DIR, "from_scratch_vs_warmstart.png")
    plot_training_curves(plot_path)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
