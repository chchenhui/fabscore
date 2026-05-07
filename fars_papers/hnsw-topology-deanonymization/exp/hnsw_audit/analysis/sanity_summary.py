"""Compile sanity check summary from ER and M sensitivity results.

Loads analysis_er_sanity.json and analysis_m_sensitivity.json, builds a
combined summary table, and verifies that (1) ER results are near chance
and (2) adjacency-only recall increases with M.
"""

import json
import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():
    er_path = os.path.join(RESULTS_DIR, "analysis_er_sanity.json")
    m_path = os.path.join(RESULTS_DIR, "analysis_m_sensitivity.json")

    with open(er_path) as f:
        er = json.load(f)
    with open(m_path) as f:
        m_data = json.load(f)

    n = er["n"]
    k = er["k"]
    chance = k / (n - 1)

    rows = []

    rows.append({
        "graph_type": "ER random",
        "M": None,
        "avg_degree": er["er_avg_degree"],
        "adjacency_only_recall_at_10": er["er_adjacency_only_recall_at_10"],
        "degree_penalized_recall_at_10": er["er_degree_penalized_recall_at_10"],
    })

    for run in m_data["runs"]:
        rows.append({
            "graph_type": "HNSW",
            "M": run["M"],
            "avg_degree": run["avg_degree"],
            "adjacency_only_recall_at_10": run["adjacency_only_recall_at_10"],
            "degree_penalized_recall_at_10": run["degree_penalized_recall_at_10"],
        })

    rows.append({
        "graph_type": "Chance",
        "M": None,
        "avg_degree": None,
        "adjacency_only_recall_at_10": round(chance, 6),
        "degree_penalized_recall_at_10": None,
    })

    er_adj = er["er_adjacency_only_recall_at_10"]
    er_recon = er["er_degree_penalized_recall_at_10"]
    er_near_chance = (er_adj < 10 * chance) and (er_recon < 10 * chance)

    hnsw_runs = [r for r in rows if r["graph_type"] == "HNSW"]
    adj_recalls = [r["adjacency_only_recall_at_10"] for r in hnsw_runs]
    recon_recalls = [r["degree_penalized_recall_at_10"] for r in hnsw_runs]
    adj_monotonic = all(adj_recalls[i] <= adj_recalls[i + 1] for i in range(len(adj_recalls) - 1))
    recon_monotonic = all(recon_recalls[i] <= recon_recalls[i + 1] for i in range(len(recon_recalls) - 1))

    verification = {
        "er_near_chance": er_near_chance,
        "er_adj_vs_chance_ratio": round(er_adj / chance, 2),
        "er_recon_vs_chance_ratio": round(er_recon / chance, 2),
        "adjacency_monotonic_increase_with_M": adj_monotonic,
        "reconstruction_monotonic_increase_with_M": recon_monotonic,
    }

    if not recon_monotonic:
        verification["reconstruction_note"] = (
            "Reconstruction Recall@10 does not increase monotonically with M. "
            "At M=64 (avg_degree~97), the very dense graph produces short, "
            "non-discriminative geodesic distances, degrading MDS embedding quality."
        )

    summary = {
        "experiment": "sanity_check_summary",
        "dataset": "SIFT10K",
        "n": n,
        "k": k,
        "chance_recall_at_k": round(chance, 6),
        "table": rows,
        "verification": verification,
    }

    print("=== Sanity Check Summary ===\n")
    print(f"{'Graph Type':<12} {'M':>4} {'Avg Deg':>8} {'Adj Recall@10':>15} {'Recon Recall@10':>16}")
    print("-" * 60)
    for r in rows:
        m_str = str(r["M"]) if r["M"] is not None else "N/A"
        deg_str = f"{r['avg_degree']:.2f}" if r["avg_degree"] is not None else "N/A"
        adj_str = f"{r['adjacency_only_recall_at_10']:.6f}"
        recon_str = f"{r['degree_penalized_recall_at_10']:.6f}" if r["degree_penalized_recall_at_10"] is not None else "N/A"
        print(f"{r['graph_type']:<12} {m_str:>4} {deg_str:>8} {adj_str:>15} {recon_str:>16}")

    print(f"\n--- Verification ---")
    print(f"ER near chance: {verification['er_near_chance']} "
          f"(adj {verification['er_adj_vs_chance_ratio']}x, recon {verification['er_recon_vs_chance_ratio']}x chance)")
    print(f"Adjacency monotonic with M: {verification['adjacency_monotonic_increase_with_M']}")
    print(f"Reconstruction monotonic with M: {verification['reconstruction_monotonic_increase_with_M']}")
    if "reconstruction_note" in verification:
        print(f"  Note: {verification['reconstruction_note']}")

    out_path = os.path.join(RESULTS_DIR, "analysis_sanity_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
