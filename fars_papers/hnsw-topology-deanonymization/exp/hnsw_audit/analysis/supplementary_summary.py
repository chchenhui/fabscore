"""Compile supplementary analysis summary from k-sensitivity and scatter plot results.

Loads analysis_k_sensitivity.json and geodesic_scatter_meta.json, combines into
results/analysis_supplementary_summary.json with observations.
"""

import json
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():
    k_path = os.path.join(RESULTS_DIR, "analysis_k_sensitivity.json")
    scatter_path = os.path.join(RESULTS_DIR, "geodesic_scatter_meta.json")

    with open(k_path) as f:
        k_data = json.load(f)

    with open(scatter_path) as f:
        scatter_data = json.load(f)

    table = k_data["table"]

    def find_row(method, dataset):
        for row in table:
            if row["method"] == method and row["dataset"] == dataset:
                return row
        return None

    observations = []

    for ds in ["SIFT10K", "MSMARCO-10K"]:
        adj = find_row("adjacency_only", ds)
        uw = find_row("unweighted_geodesic", ds)
        dp = find_row("degree_penalized_geodesic", ds)

        dp_beats_uw_all_k = all(
            dp[f"recall@{k}"] > uw[f"recall@{k}"] for k in [5, 10, 20]
        )
        dp_beats_adj_all_k = all(
            dp[f"recall@{k}"] > adj[f"recall@{k}"] for k in [5, 10, 20]
        )
        observations.append({
            "dataset": ds,
            "degree_penalized_beats_unweighted_all_k": dp_beats_uw_all_k,
            "degree_penalized_beats_adjacency_all_k": dp_beats_adj_all_k,
        })

    scatter_improvement = (
        scatter_data["spearman_degree_penalized"] > scatter_data["spearman_unweighted"]
    )

    summary = {
        "experiment": "supplementary_analyses",
        "k_sensitivity": {
            "description": "Recall@k for k in {5, 10, 20} across three methods and two datasets (seed=42)",
            "table": table,
            "observations": observations,
        },
        "geodesic_scatter": {
            "description": "Scatter plot of graph-geodesic vs Euclidean distances for SIFT10K (seed=42)",
            "spearman_unweighted": scatter_data["spearman_unweighted"],
            "spearman_degree_penalized": scatter_data["spearman_degree_penalized"],
            "n_pairs": scatter_data["n_pairs"],
            "degree_penalty_improves_correlation": scatter_improvement,
            "figure_path": "results/figures/geodesic_vs_euclidean_scatter.pdf",
        },
        "conclusions": [],
    }

    all_dp_wins = all(o["degree_penalized_beats_unweighted_all_k"] for o in observations)
    if all_dp_wins:
        summary["conclusions"].append(
            "Degree-penalized geodesic consistently outperforms unweighted geodesic "
            "across all k values (5, 10, 20) on both SIFT10K and MSMARCO-10K."
        )

    any_dp_beats_adj = any(o["degree_penalized_beats_adjacency_all_k"] for o in observations)
    if any_dp_beats_adj:
        datasets_beating = [o["dataset"] for o in observations if o["degree_penalized_beats_adjacency_all_k"]]
        summary["conclusions"].append(
            f"Degree-penalized geodesic surpasses the adjacency-only baseline "
            f"at all k values on: {', '.join(datasets_beating)}."
        )

    if scatter_improvement:
        summary["conclusions"].append(
            f"Scatter plots confirm the degree penalty improves distance correlation: "
            f"Spearman rho increases from {scatter_data['spearman_unweighted']:.4f} "
            f"(unweighted) to {scatter_data['spearman_degree_penalized']:.4f} "
            f"(degree-penalized)."
        )

    out_path = os.path.join(RESULTS_DIR, "analysis_supplementary_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
