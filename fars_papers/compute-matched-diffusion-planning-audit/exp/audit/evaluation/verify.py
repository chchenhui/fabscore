"""Unified evaluation script for Reasoning Gym tasks.
Loads model output JSONL, scores each instance against the Reasoning Gym verifier,
and writes a summary CSV.

Usage:
  python audit/evaluation/verify.py \
    --results_file audit/results/raw/qwen_greedy_countdown.jsonl \
    --task countdown --seed 2024 --size 500 \
    --method qwen_greedy \
    --output_csv audit/results/tables/qwen_greedy_results.csv

  Can also run oracle/negative sanity checks with --sanity_check flag.
"""

import argparse
import csv
import json
import os
import reasoning_gym


def load_results(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def evaluate(results, dataset):
    scores = []
    for rec in results:
        idx = rec["id"]
        entry = dataset[idx]
        parsed = rec.get("parsed_answer", "")
        raw_score = dataset.score_answer(answer=parsed, entry=entry)
        binary = 1.0 if raw_score == 1.0 else 0.0
        scores.append({"id": idx, "raw_score": raw_score, "binary": binary})
    accuracy = sum(s["binary"] for s in scores) / len(scores) if scores else 0.0
    num_correct = int(sum(s["binary"] for s in scores))
    return accuracy, num_correct, len(scores), scores


def sanity_check(task, seed, size):
    ds = reasoning_gym.create_dataset(task, seed=seed, size=size)
    oracle_pass = 0
    negative_pass = 0
    for i, entry in enumerate(ds):
        oracle_score = ds.score_answer(answer=entry["answer"], entry=entry)
        if oracle_score == 1.0:
            oracle_pass += 1
        else:
            print(f"  WARNING: Oracle failed for instance {i}, score={oracle_score}")

        neg_score = ds.score_answer(answer="random_string_xyz_999", entry=entry)
        if neg_score < 1.0:
            negative_pass += 1
        else:
            print(f"  WARNING: Negative check failed for instance {i}, score={neg_score}")

    print(f"  Oracle check: {oracle_pass}/{size} passed (expect {size}/{size})")
    print(f"  Negative check: {negative_pass}/{size} passed (expect {size}/{size})")
    assert oracle_pass == size, f"Oracle check failed: {oracle_pass}/{size}"
    assert negative_pass == size, f"Negative check failed: {negative_pass}/{size}"
    print("  Sanity checks PASSED.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, help="Path to results JSONL")
    parser.add_argument("--task", type=str, choices=["countdown", "mini_sudoku"])
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--method", type=str, default="unknown")
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV")
    args = parser.parse_args()

    if args.sanity_check:
        print(f"Running sanity checks for {args.task} (seed={args.seed}, size={args.size})...")
        sanity_check(args.task, args.seed, args.size)
        return

    ds = reasoning_gym.create_dataset(args.task, seed=args.seed, size=args.size)
    results = load_results(args.results_file)
    accuracy, num_correct, num_total, per_instance = evaluate(results, ds)

    print(f"Task: {args.task}")
    print(f"Method: {args.method}")
    print(f"Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        write_header = not (args.append and os.path.exists(args.output_csv))
        mode = "a" if args.append else "w"
        with open(args.output_csv, mode, newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["task", "method", "accuracy", "num_correct", "num_total"])
            writer.writerow([args.task, args.method, f"{accuracy:.4f}", num_correct, num_total])
        print(f"Results written to {args.output_csv}")

    detail_path = args.results_file.replace(".jsonl", "_scored.jsonl")
    with open(detail_path, "w") as f:
        for s in per_instance:
            f.write(json.dumps(s) + "\n")
    print(f"Per-instance scores written to {detail_path}")


if __name__ == "__main__":
    main()
