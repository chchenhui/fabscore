# Compute PDM-H curves from saved per-step logits and generate publication figure.
# Loads npz files from outputs/pdm_analysis/{method}/{benchmark}/, computes H^2
# at each step, averages across items with SEM bands, and plots 2-panel figure.
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mi_decoding.evaluation.pdm_h import hellinger_squared


def load_logit_pairs(output_dir):
    """Load and merge all shard npz files from output_dir.

    Returns:
        dict: {item_id: {step: (lc, lu)}}
    """
    all_data = {}
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".npz"):
            continue
        fpath = os.path.join(output_dir, fname)
        data = np.load(fpath, allow_pickle=True)

        item_ids = data.get("_item_ids", np.array([]))

        step_keys = [k for k in data.files if k.endswith("_lc") and k != "_item_ids"]
        seen_items = set()
        for key in step_keys:
            parts = key.rsplit("_step", 1)
            if len(parts) == 2:
                seen_items.add(parts[0])

        for iid in seen_items:
            if iid not in all_data:
                all_data[iid] = {}
            for key in data.files:
                if key.startswith(f"{iid}_step") and key.endswith("_lc"):
                    step_str = key.split("_step")[1].replace("_lc", "")
                    step = int(step_str)
                    lc = data[key]
                    lu_key = f"{iid}_step{step}_lu"
                    if lu_key in data.files:
                        lu = data[lu_key]
                        all_data[iid][step] = (lc, lu)

    return all_data


def compute_pdm_h_curves(logit_data):
    """Compute per-item PDM-H at each step.

    Returns:
        steps: sorted list of step numbers
        per_item_curves: list of dicts {step: h2}
    """
    all_steps = set()
    per_item_curves = []

    for item_id, step_data in logit_data.items():
        curve = {}
        for step, (lc, lu) in step_data.items():
            h2 = hellinger_squared(lc, lu)
            curve[step] = h2
            all_steps.add(step)
        per_item_curves.append(curve)

    steps = sorted(all_steps)
    return steps, per_item_curves


def aggregate_curves(steps, per_item_curves, min_items=5):
    """Compute mean and SEM across items at each step.

    Only includes steps where at least min_items have data.

    Returns:
        valid_steps, means, sems, counts (numpy arrays)
    """
    valid_steps = []
    means = []
    sems = []
    counts = []

    for s in steps:
        vals = [c[s] for c in per_item_curves if s in c]
        if len(vals) >= min_items:
            valid_steps.append(s)
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            counts.append(len(vals))

    return np.array(valid_steps), np.array(means), np.array(sems), np.array(counts)


def compute_auc(steps, means):
    """Area under PDM-H curve using trapezoidal rule."""
    if len(steps) < 2:
        return 0.0
    return float(np.trapz(means, steps))


def main():
    base_dir = os.path.join(PROJECT_ROOT, "mi_decoding", "outputs", "pdm_analysis")
    results_dir = os.path.join(PROJECT_ROOT, "mi_decoding", "results")
    os.makedirs(results_dir, exist_ok=True)

    methods = ["vanilla", "adaptive_mi"]
    benchmarks = ["mmstar", "hallusionbench"]
    method_labels = {"vanilla": "Vanilla", "adaptive_mi": "Adaptive MI"}
    method_colors = {"vanilla": "#1f77b4", "adaptive_mi": "#d62728"}
    bench_titles = {"mmstar": "MMStar (50-item subset)", "hallusionbench": "HallusionBench (50-item subset)"}

    all_results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for bi, bench in enumerate(benchmarks):
        ax = axes[bi]
        all_results[bench] = {}

        for method in methods:
            data_dir = os.path.join(base_dir, method, bench)
            if not os.path.exists(data_dir):
                print(f"WARNING: {data_dir} not found, skipping")
                continue

            print(f"Loading {method}/{bench}...")
            logit_data = load_logit_pairs(data_dir)
            print(f"  {len(logit_data)} items loaded")

            steps, per_item_curves = compute_pdm_h_curves(logit_data)
            valid_steps, means, sems, counts = aggregate_curves(steps, per_item_curves, min_items=5)

            auc = compute_auc(valid_steps, means)
            print(f"  Valid steps: {len(valid_steps)} (range {valid_steps[0]}-{valid_steps[-1]})")
            print(f"  Mean PDM-H range: [{means.min():.4f}, {means.max():.4f}]")
            print(f"  Items at first/last step: {counts[0]}/{counts[-1]}")
            print(f"  AUC-PDM-H: {auc:.2f}")

            all_results[bench][method] = {
                "steps": valid_steps.tolist(),
                "mean_pdm_h": means.tolist(),
                "sem_pdm_h": sems.tolist(),
                "item_counts": counts.tolist(),
                "auc_pdm_h": auc,
                "num_items": len(logit_data),
            }

            ax.plot(valid_steps, means,
                    label=method_labels[method],
                    color=method_colors[method],
                    linewidth=2)
            ax.fill_between(valid_steps, means - sems, means + sems,
                            color=method_colors[method], alpha=0.15)

        ax.set_title(bench_titles[bench], fontsize=13)
        ax.set_xlabel("Generation Step", fontsize=12)
        if bi == 0:
            ax.set_ylabel("PDM-H (Squared Hellinger Distance)", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    auc_summary = {}
    for bench in benchmarks:
        if bench in all_results and "vanilla" in all_results[bench] and "adaptive_mi" in all_results[bench]:
            v = all_results[bench]["vanilla"]
            m = all_results[bench]["adaptive_mi"]
            common_steps = sorted(set(v["steps"]) & set(m["steps"]))
            v_step_map = dict(zip(v["steps"], v["mean_pdm_h"]))
            m_step_map = dict(zip(m["steps"], m["mean_pdm_h"]))
            v_common = np.array([v_step_map[s] for s in common_steps])
            m_common = np.array([m_step_map[s] for s in common_steps])
            common_arr = np.array(common_steps)
            vanilla_auc = float(np.trapz(v_common, common_arr))
            mi_auc = float(np.trapz(m_common, common_arr))
            ratio = mi_auc / vanilla_auc if vanilla_auc > 0 else float("inf")
            auc_summary[bench] = {
                "common_step_range": [int(common_arr[0]), int(common_arr[-1])],
                "num_common_steps": len(common_steps),
                "vanilla_auc": vanilla_auc,
                "adaptive_mi_auc": mi_auc,
                "ratio_mi_over_vanilla": ratio,
            }

    plt.tight_layout()
    fig_path = os.path.join(results_dir, "pdm_h_vs_step.pdf")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")

    png_path = os.path.join(results_dir, "pdm_h_vs_step.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"PNG saved to {png_path}")
    plt.close()

    output = {
        "per_benchmark": all_results,
        "auc_summary": auc_summary,
    }
    json_path = os.path.join(results_dir, "pdm_h_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Numerical data saved to {json_path}")

    print("\n=== AUC-PDM-H Summary ===")
    for bench, summ in auc_summary.items():
        print(f"{bench}: Vanilla AUC={summ['vanilla_auc']:.2f}, "
              f"Adaptive MI AUC={summ['adaptive_mi_auc']:.2f}, "
              f"Ratio={summ['ratio_mi_over_vanilla']:.3f}")


if __name__ == "__main__":
    main()
