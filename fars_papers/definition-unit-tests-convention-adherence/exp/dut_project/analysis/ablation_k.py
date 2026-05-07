"""Ablation study: k=1 vs k=3 discriminative checks.

Compares DUT performance with different numbers of checks.
Loads pre-generated outputs for k=1 and k=3, scores them, and computes
bootstrap CIs for the difference.

Usage: python -m dut_project.analysis.ablation_k \
    --k1-output outputs/cond_c_k1.jsonl \
    --k3-output outputs/cond_c_k3.jsonl \
    --bench data/erdos_conventions_bench.jsonl
"""

import argparse
import json
from typing import Any

from dut_project.evaluation.score import score_item, aggregate_scores
from dut_project.evaluation.statistics import paired_bootstrap_ci


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def run_ablation(
    k1_outputs: list[dict],
    k3_outputs: list[dict],
    bench_items: list[dict],
) -> dict[str, Any]:
    bench_by_id = {item["item_id"]: item for item in bench_items}

    k1_scored = []
    for out in k1_outputs:
        item = bench_by_id[out["item_id"]]
        k1_scored.append(score_item(out["parsed"], item, out["condition"]))

    k3_scored = []
    for out in k3_outputs:
        item = bench_by_id[out["item_id"]]
        k3_scored.append(score_item(out["parsed"], item, out["condition"]))

    k1_agg = aggregate_scores(k1_scored)
    k3_agg = aggregate_scores(k3_scored)

    k1_correct = [s["main_correct"] for s in k1_scored]
    k3_correct = [s["main_correct"] for s in k3_scored]

    if len(k1_correct) == len(k3_correct):
        ci = paired_bootstrap_ci(k3_correct, k1_correct)
    else:
        ci = {"error": "Mismatched item counts between k=1 and k=3"}

    return {
        "k1_accuracy": k1_agg["main_accuracy"],
        "k3_accuracy": k3_agg["main_accuracy"],
        "bootstrap_ci_k3_minus_k1": ci,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1-output", required=True)
    parser.add_argument("--k3-output", required=True)
    parser.add_argument("--bench", required=True)
    args = parser.parse_args()

    k1 = load_jsonl(args.k1_output)
    k3 = load_jsonl(args.k3_output)
    bench = load_jsonl(args.bench)

    results = run_ablation(k1, k3, bench)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
