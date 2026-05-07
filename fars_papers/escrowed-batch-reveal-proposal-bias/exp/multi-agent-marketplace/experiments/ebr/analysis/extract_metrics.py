"""Extract proposal-bias metrics from experiment SQLite databases.

Given a results directory of SQLite DBs, determines which proposal rank
(by arrival time) was paid, computes completion rate, and aggregates
across scenarios and repetitions.

Reuses SQL logic from experiments/position/generate_proposal_data.py.
"""

import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np


def analyze_single_db(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    result = {
        "db_path": db_path,
        "payment_made": False,
        "chosen_proposal_rank": None,
        "total_proposals_received": 0,
        "earliest_arrival_chosen": False,
        "chosen_price": None,
    }

    cursor.execute("""
    SELECT
        json_extract(data, '$.agent_id') as from_business,
        json_extract(data, '$.request.parameters.to_agent_id') as to_customer,
        json_extract(data, '$.request.parameters.message.id') as proposal_id,
        json_extract(data, '$.request.parameters.message.proposal.total_price') as price,
        created_at
    FROM actions
    WHERE json_extract(data, '$.request.name') = 'SendMessage'
        AND json_extract(data, '$.request.parameters.message.type') = 'order_proposal'
    ORDER BY created_at
    """)
    proposals = cursor.fetchall()
    result["total_proposals_received"] = len(proposals)

    cursor.execute("""
    SELECT
        json_extract(data, '$.agent_id') as from_customer,
        json_extract(data, '$.request.parameters.message.proposal_message_id') as paid_proposal_id
    FROM actions
    WHERE json_extract(data, '$.request.name') = 'SendMessage'
        AND json_extract(data, '$.request.parameters.message.type') = 'payment'
    LIMIT 1
    """)
    payment = cursor.fetchone()

    if payment and proposals:
        paid_proposal_id = payment[1]
        result["payment_made"] = True

        for rank, (from_biz, to_cust, proposal_id, price, created_at) in enumerate(proposals, start=1):
            if proposal_id == paid_proposal_id:
                result["chosen_proposal_rank"] = rank
                result["earliest_arrival_chosen"] = (rank == 1)
                result["chosen_price"] = price
                break

    conn.close()
    return result


def extract_metrics(results_dir: str, output_csv: str | None = None) -> dict:
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return {}

    all_results = []
    scenario_results = {}

    for db_file in sorted(results_path.glob("*.db")):
        if db_file.name.endswith(("-shm", "-wal")):
            continue

        r = analyze_single_db(str(db_file))
        r["filename"] = db_file.name

        name = db_file.stem
        for folder in ["contractors_first", "contractors_second", "contractors_third"]:
            if folder in name:
                r["scenario"] = folder
                break
        else:
            r["scenario"] = "unknown"

        all_results.append(r)
        scenario_results.setdefault(r["scenario"], []).append(r)

    if not all_results:
        print("No results found!")
        return {}

    n_total = len(all_results)
    n_completed = sum(1 for r in all_results if r["payment_made"])
    n_earliest = sum(1 for r in all_results if r["earliest_arrival_chosen"])

    completion_rate = n_completed / n_total if n_total > 0 else 0
    earliest_rate = n_earliest / n_total if n_total > 0 else 0

    rank_counts = {}
    for r in all_results:
        rank = r["chosen_proposal_rank"]
        if rank is not None:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

    per_scenario = {}
    for scenario, runs in sorted(scenario_results.items()):
        n = len(runs)
        nc = sum(1 for r in runs if r["payment_made"])
        ne = sum(1 for r in runs if r["earliest_arrival_chosen"])
        sc_rank_counts = {}
        for r in runs:
            rank = r["chosen_proposal_rank"]
            if rank is not None:
                sc_rank_counts[rank] = sc_rank_counts.get(rank, 0) + 1

        per_scenario[scenario] = {
            "n_runs": n,
            "n_completed": nc,
            "completion_rate": nc / n if n > 0 else 0,
            "n_earliest_chosen": ne,
            "earliest_arrival_rate": ne / n if n > 0 else 0,
            "rank_counts": sc_rank_counts,
        }

    completion_rates_per_run = [1.0 if r["payment_made"] else 0.0 for r in all_results]
    earliest_rates_per_run = [1.0 if r["earliest_arrival_chosen"] else 0.0 for r in all_results]

    summary = {
        "total_runs": n_total,
        "completion_rate_mean": float(np.mean(completion_rates_per_run)),
        "completion_rate_std": float(np.std(completion_rates_per_run)),
        "earliest_arrival_rate_mean": float(np.mean(earliest_rates_per_run)),
        "earliest_arrival_rate_std": float(np.std(earliest_rates_per_run)),
        "rank_histogram": {str(k): v for k, v in sorted(rank_counts.items())},
        "rank_histogram_fractions": {
            str(k): v / n_completed for k, v in sorted(rank_counts.items())
        } if n_completed > 0 else {},
        "per_scenario": per_scenario,
    }

    print("\n=== METRICS SUMMARY ===")
    print(f"Total runs: {n_total}")
    print(f"Completion rate: {summary['completion_rate_mean']:.3f} +/- {summary['completion_rate_std']:.3f}")
    print(f"Earliest-arrival chosen rate: {summary['earliest_arrival_rate_mean']:.3f} +/- {summary['earliest_arrival_rate_std']:.3f}")
    print(f"\nRank histogram (counts): {rank_counts}")
    if n_completed > 0:
        print(f"Rank histogram (fractions):")
        for rank, count in sorted(rank_counts.items()):
            print(f"  Rank {rank}: {count}/{n_completed} = {count/n_completed:.3f}")

    print("\nPer-scenario breakdown:")
    for scenario, stats in sorted(per_scenario.items()):
        print(f"  {scenario}: completion={stats['completion_rate']:.3f}, earliest={stats['earliest_arrival_rate']:.3f}, ranks={stats['rank_counts']}")

    if summary["earliest_arrival_rate_mean"] <= 0.50:
        print(f"\nGATE CHECK: earliest-arrival rate ({summary['earliest_arrival_rate_mean']:.3f}) <= 0.50 -- insufficient bias to study. Note for later.")

    if output_csv:
        fieldnames = ["filename", "scenario", "payment_made", "chosen_proposal_rank",
                       "total_proposals_received", "earliest_arrival_chosen", "chosen_price"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow({k: r[k] for k in fieldnames})
        print(f"\nCSV written to: {output_csv}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory containing experiment SQLite DBs")
    parser.add_argument("--csv", default=None, help="Output CSV path")
    parser.add_argument("--json", default=None, help="Output JSON summary path")
    args = parser.parse_args()

    summary = extract_metrics(args.results_dir, output_csv=args.csv)

    if args.json and summary:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary written to: {args.json}")
