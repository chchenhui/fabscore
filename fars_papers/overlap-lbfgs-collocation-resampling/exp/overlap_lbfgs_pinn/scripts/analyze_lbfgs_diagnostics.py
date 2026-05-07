# Visualization of L-BFGS per-step diagnostics across 4 collocation configs.
# Produces 4 figures: bar chart of total iterations, grad_norm trajectory,
# step size trajectory, and curvature pair quality trajectory.
# Also generates RESULTS.json summary.

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = {
    "Naive (o=0)": os.path.join(BASE, "outputs", "diagnostics_naive_o0"),
    "Fixed": os.path.join(BASE, "outputs", "diagnostics_fixed"),
    "Overlap o=0.25": os.path.join(BASE, "outputs", "diagnostics_overlap_o025"),
    "Overlap o=0.5": os.path.join(BASE, "outputs", "diagnostics_overlap_o05"),
}
SEEDS = [0, 1, 2]
COLORS = {"Naive (o=0)": "#d62728", "Fixed": "#1f77b4", "Overlap o=0.25": "#ff7f0e", "Overlap o=0.5": "#2ca02c"}

OUT_DIR = os.path.join(os.path.dirname(BASE), "EXPERIMENT_RESULTS", "lbfgs_diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)


def load_step_histories(config_dir):
    histories = {}
    for s in SEEDS:
        path = os.path.join(config_dir, f"seed_{s}_lbfgs_step_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[s] = json.load(f)
    return histories


def load_metrics(config_dir):
    metrics = {}
    for s in SEEDS:
        path = os.path.join(config_dir, f"seed_{s}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                metrics[s] = json.load(f)
    return metrics


def plot_bar_chart(all_data, all_metrics):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(CONFIGS.keys())
    means, stds = [], []
    for name in names:
        steps = [all_metrics[name][s]["lbfgs_diagnostics"]["lbfgs_outer_steps"]
                 for s in SEEDS if s in all_metrics[name]]
        means.append(np.mean(steps))
        stds.append(np.std(steps))

    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=[COLORS[n] for n in names], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Total L-BFGS Outer Iterations", fontsize=12)
    ax.set_title("L-BFGS Iterations Completed per Configuration\n(mean +/- std, 3 seeds)", fontsize=13)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 50, f"{m:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bar_total_iterations.png"), dpi=150)
    plt.close(fig)
    print("Saved bar_total_iterations.png")


def plot_grad_norm(all_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, histories in all_data.items():
        if 0 not in histories:
            continue
        h = histories[0]
        iters = [d["iter"] for d in h if np.isfinite(d["grad_norm"]) and d["grad_norm"] > 0]
        gn = [d["grad_norm"] for d in h if np.isfinite(d["grad_norm"]) and d["grad_norm"] > 0]
        ax.plot(iters, gn, label=name, color=COLORS[name], linewidth=0.8, alpha=0.9)

    ax.set_yscale("log")
    ax.set_xlabel("L-BFGS Iteration", fontsize=12)
    ax.set_ylabel("Gradient Norm ||g_k||", fontsize=12)
    ax.set_title("Gradient Norm vs L-BFGS Iteration (seed 0)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grad_norm_trajectory.png"), dpi=150)
    plt.close(fig)
    print("Saved grad_norm_trajectory.png")


def plot_step_size(all_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, histories in all_data.items():
        if 0 not in histories:
            continue
        h = histories[0]
        valid = [(d["iter"], d["alpha"]) for d in h
                 if d["alpha"] is not None and np.isfinite(d["alpha"]) and d["alpha"] > 0]
        if not valid:
            continue
        iters, alphas = zip(*valid)
        ax.plot(iters, alphas, label=name, color=COLORS[name], linewidth=0.8, alpha=0.9)

    ax.set_yscale("log")
    ax.set_xlabel("L-BFGS Iteration", fontsize=12)
    ax.set_ylabel("Step Size alpha_k", fontsize=12)
    ax.set_title("Step Size vs L-BFGS Iteration (seed 0)\n"
                 "(Fixed-LBFGS omitted: torch.optim.LBFGS does not expose per-step alpha)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "step_size_trajectory.png"), dpi=150)
    plt.close(fig)
    print("Saved step_size_trajectory.png")


def plot_curvature_quality(all_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, histories in all_data.items():
        if name == "Fixed":
            continue
        if 0 not in histories:
            continue
        h = histories[0]
        iters, quality = [], []
        for d in h:
            ys = d.get("ys_value")
            sn = d.get("s_norm")
            gn = d.get("grad_norm")
            if (ys is not None and sn is not None and gn is not None
                    and sn > 0 and gn > 0 and np.isfinite(ys) and np.isfinite(sn) and np.isfinite(gn)):
                q = ys / (sn ** 2 * gn)
                if np.isfinite(q):
                    iters.append(d["iter"])
                    quality.append(q)
        if iters:
            ax.plot(iters, quality, label=name, color=COLORS[name], linewidth=0.8, alpha=0.9)

    ax.set_xlabel("L-BFGS Iteration", fontsize=12)
    ax.set_ylabel(r"$y_k^T s_k \;/\; (||s_k||^2 \cdot ||g_k||)$", fontsize=12)
    ax.set_title("Curvature Pair Quality vs L-BFGS Iteration (seed 0)\n"
                 "(Fixed-LBFGS omitted: torch.optim.LBFGS does not expose curvature pairs)",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "curvature_quality_trajectory.png"), dpi=150)
    plt.close(fig)
    print("Saved curvature_quality_trajectory.png")


def generate_results_json(all_metrics):
    summary = {}
    for name in CONFIGS:
        seeds_data = []
        for s in SEEDS:
            if s not in all_metrics[name]:
                continue
            m = all_metrics[name][s]
            d = m["lbfgs_diagnostics"]
            seeds_data.append({
                "seed": s,
                "lbfgs_outer_steps": d["lbfgs_outer_steps"],
                "line_search_failures": d.get("line_search_failures", 0),
                "cautious_skips": d.get("cautious_skips", 0),
                "termination_reason": d["termination_reason"],
                "total_evals": m["total_evals"],
                "best_B_err": m["best_B_err"],
            })
        steps = [s["lbfgs_outer_steps"] for s in seeds_data]
        ls_fail = [s["line_search_failures"] for s in seeds_data]
        c_skip = [s["cautious_skips"] for s in seeds_data]
        berr = [s["best_B_err"] for s in seeds_data]
        summary[name] = {
            "per_seed": seeds_data,
            "lbfgs_steps_mean": float(np.mean(steps)),
            "lbfgs_steps_std": float(np.std(steps)),
            "ls_failures_mean": float(np.mean(ls_fail)),
            "ls_failures_total": int(np.sum(ls_fail)),
            "cautious_skips_mean": float(np.mean(c_skip)),
            "B_err_mean": float(np.mean(berr)),
            "B_err_std": float(np.std(berr)),
        }

    with open(os.path.join(OUT_DIR, "RESULTS.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved RESULTS.json")
    return summary


def main():
    all_data = {}
    all_metrics = {}
    for name, d in CONFIGS.items():
        all_data[name] = load_step_histories(d)
        all_metrics[name] = load_metrics(d)
        print(f"{name}: {len(all_data[name])} seeds loaded "
              f"({', '.join(f's{s}:{len(h)} steps' for s, h in all_data[name].items())})")

    plot_bar_chart(all_data, all_metrics)
    plot_grad_norm(all_data)
    plot_step_size(all_data)
    plot_curvature_quality(all_data)
    summary = generate_results_json(all_metrics)

    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    for name, s in summary.items():
        print(f"\n{name}:")
        print(f"  L-BFGS steps: {s['lbfgs_steps_mean']:.0f} +/- {s['lbfgs_steps_std']:.0f}")
        print(f"  LS failures total: {s['ls_failures_total']}")
        print(f"  Cautious skips avg: {s['cautious_skips_mean']:.0f}")
        print(f"  B_err: {s['B_err_mean']:.6e} +/- {s['B_err_std']:.6e}")
        for sd in s["per_seed"]:
            print(f"    Seed {sd['seed']}: steps={sd['lbfgs_outer_steps']}, "
                  f"ls_fail={sd['line_search_failures']}, skips={sd['cautious_skips']}, "
                  f"term={sd['termination_reason']}")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
