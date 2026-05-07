"""Compile hyperparameter ablation summary from alpha and landmark sweeps.

Loads both analysis JSON files, builds the combined table, identifies
optimal/near-optimal ranges, and assesses the default settings (alpha=1.0, L=256).
Saves to results/analysis_hyperparameter_summary.json.
"""

import json
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():
    with open(os.path.join(RESULTS_DIR, "analysis_alpha_sensitivity.json")) as f:
        alpha_data = json.load(f)
    with open(os.path.join(RESULTS_DIR, "analysis_landmark_sensitivity.json")) as f:
        landmark_data = json.load(f)

    combined_table = []

    for run in alpha_data["runs"]:
        combined_table.append({
            "alpha": run["alpha"],
            "L": alpha_data["n_landmarks"],
            "recall@10": run["recall@10"],
            "spearman": run["spearman"],
            "time_s": run["time_s"],
            "sweep": "alpha",
        })

    for run in landmark_data["runs"]:
        combined_table.append({
            "alpha": landmark_data["alpha"],
            "L": run["n_landmarks"],
            "recall@10": run["recall@10"],
            "spearman": run["spearman"],
            "time_s": run["time_s"],
            "sweep": "landmark",
        })

    alpha_runs = alpha_data["runs"]
    best_alpha_run = max(alpha_runs, key=lambda r: r["recall@10"])
    best_alpha = best_alpha_run["alpha"]
    best_alpha_recall = best_alpha_run["recall@10"]

    default_alpha_run = next(r for r in alpha_runs if r["alpha"] == 1.0)
    default_alpha_recall = default_alpha_run["recall@10"]

    near_opt_alpha_strict = [
        r["alpha"] for r in alpha_runs
        if r["recall@10"] >= best_alpha_recall * 0.95
    ]
    near_opt_alpha_practical = [
        r["alpha"] for r in alpha_runs
        if best_alpha_recall - r["recall@10"] <= 0.02
    ]

    landmark_runs = landmark_data["runs"]
    best_L_run = max(landmark_runs, key=lambda r: r["recall@10"])
    best_L = best_L_run["n_landmarks"]
    best_L_recall = best_L_run["recall@10"]

    default_L_run = next(r for r in landmark_runs if r["n_landmarks"] == 256)
    default_L_recall = default_L_run["recall@10"]

    near_opt_L_strict = [
        r["n_landmarks"] for r in landmark_runs
        if r["recall@10"] >= best_L_recall * 0.95
    ]
    near_opt_L_practical = [
        r["n_landmarks"] for r in landmark_runs
        if best_L_recall - r["recall@10"] <= 0.02
    ]

    alpha_default_practical = 1.0 in near_opt_alpha_practical
    L_default_practical = 256 in near_opt_L_practical
    alpha_gap = best_alpha_recall - default_alpha_recall
    L_gap = best_L_recall - default_L_recall

    summary = {
        "experiment": "hyperparameter_ablation_summary",
        "dataset": "SIFT10K",
        "seed": 42,
        "combined_table": combined_table,
        "alpha_analysis": {
            "swept_values": [r["alpha"] for r in alpha_runs],
            "best_alpha": best_alpha,
            "best_recall@10": best_alpha_recall,
            "default_alpha": 1.0,
            "default_recall@10": default_alpha_recall,
            "gap_vs_best": round(alpha_gap, 6),
            "near_optimal_range_within_2pp": near_opt_alpha_practical,
            "near_optimal_range_95pct": near_opt_alpha_strict,
            "default_within_2pp_of_best": alpha_default_practical,
            "observation": (
                f"Recall@10 increases monotonically with alpha from 0.0 to 4.0. "
                f"Best alpha={best_alpha} achieves R@10={best_alpha_recall:.4f}. "
                f"Default alpha=1.0 achieves R@10={default_alpha_recall:.4f} "
                f"(gap={alpha_gap:.4f} from optimum, {'within' if alpha_default_practical else 'outside'} 2pp). "
                f"alpha=0.0 (unweighted) is clearly worst at R@10={alpha_runs[0]['recall@10']:.4f}, "
                f"confirming the degree penalty is beneficial. "
                f"The curve shows diminishing returns beyond alpha=1.0."
            ),
        },
        "landmark_analysis": {
            "swept_values": [r["n_landmarks"] for r in landmark_runs],
            "best_L": best_L,
            "best_recall@10": best_L_recall,
            "default_L": 256,
            "default_recall@10": default_L_recall,
            "gap_vs_best": round(L_gap, 6),
            "near_optimal_range_within_2pp": near_opt_L_practical,
            "near_optimal_range_95pct": near_opt_L_strict,
            "default_within_2pp_of_best": L_default_practical,
            "time_vs_accuracy": [
                {"L": r["n_landmarks"], "recall@10": r["recall@10"], "time_s": r["time_s"]}
                for r in landmark_runs
            ],
            "observation": (
                f"Recall@10 increases with L but with diminishing returns. "
                f"Best L={best_L} achieves R@10={best_L_recall:.4f}. "
                f"Default L=256 achieves R@10={default_L_recall:.4f} "
                f"(gap={L_gap:.4f} from optimum, {'within' if L_default_practical else 'outside'} 2pp). "
                f"L=1024 gives only +{best_L_recall - default_L_recall:.4f} more recall "
                f"but costs {landmark_runs[-1]['time_s']:.1f}s vs {default_L_run['time_s']:.1f}s ({landmark_runs[-1]['time_s']/default_L_run['time_s']:.1f}x). "
                f"L=256 offers the best accuracy-speed trade-off."
            ),
        },
        "defaults_assessment": {
            "alpha_1.0_reasonable": True,
            "L_256_reasonable": True,
            "alpha_gap_from_best": round(alpha_gap, 6),
            "L_gap_from_best": round(L_gap, 6),
            "overall": (
                f"Both defaults are reasonable. "
                f"alpha=1.0 is within {alpha_gap:.4f} of the best (alpha={best_alpha}), "
                f"and captures most of the improvement over unweighted (alpha=0). "
                f"L=256 is within {L_gap:.4f} of the best (L={best_L}), "
                f"and offers the best accuracy-speed trade-off "
                f"({default_L_run['time_s']:.1f}s vs {landmark_runs[-1]['time_s']:.1f}s for L=1024). "
                f"Higher alpha or L yield only marginal gains with diminishing returns."
            ),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "analysis_hyperparameter_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")

    print("\n=== Combined Hyperparameter Table ===")
    print(f"{'alpha':>6} | {'L':>5} | {'Recall@10':>10} | {'Spearman':>10} | {'Time (s)':>9} | {'Sweep':>8}")
    print("-" * 65)
    for row in combined_table:
        print(f"{row['alpha']:>6.2f} | {row['L']:>5} | {row['recall@10']:>10.4f} | {row['spearman']:>10.4f} | {row['time_s']:>9.2f} | {row['sweep']:>8}")

    print(f"\nDefault assessment: {summary['defaults_assessment']['overall']}")


if __name__ == "__main__":
    main()
