# Create ranking JSON from base NLL scores CSV.
# Uses -NLL as composite_score (higher = better fit = higher rank).

import csv
import json
import sys
from pathlib import Path


def main():
    csv_path = sys.argv[1]
    output_path = sys.argv[2]

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "dataset": row["dataset"],
                "mean_nll": float(row["mean_nll"]),
                "total_tokens": int(row["total_tokens"]),
                "n_samples": int(row["n_samples"]),
            })

    rows.sort(key=lambda r: r["mean_nll"])

    ranking = []
    for i, row in enumerate(rows):
        ranking.append({
            "rank": i + 1,
            "dataset": row["dataset"],
            "composite_score": -row["mean_nll"],
            "mean_nll": row["mean_nll"],
            "total_tokens": row["total_tokens"],
            "n_samples": row["n_samples"],
        })

    result = {
        "ranking": ranking,
        "metadata": {
            "base_model": "Qwen/Qwen2.5-1.5B",
            "method": "base_model_nll",
            "score_formula": "composite_score = -mean_nll (higher = better fit)",
            "cutoff_len": 4096,
            "loss_on": "assistant_tokens_only",
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=== Base-Model NLL Ranking (lower NLL = better fit) ===")
    for entry in ranking:
        print(f"  Rank {entry['rank']:2d}: {entry['dataset']:45s} NLL={entry['mean_nll']:.4f}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
