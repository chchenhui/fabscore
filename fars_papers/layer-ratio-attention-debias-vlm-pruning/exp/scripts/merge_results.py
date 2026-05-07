# Merge sharded evaluation results from multi-GPU data-parallel runs.
# Each rank produces per-dataset JSON files (e.g., refcoco_val_rank0.json).
# This script aggregates them into a single summary.json with totals.

import argparse
import json
import os
import numpy as np
from collections import defaultdict

DS_NAMES = [
    "refcoco_val", "refcoco_testA", "refcoco_testB",
    "refcoco+_val", "refcoco+_testA", "refcoco+_testB",
    "refcocog_val", "refcocog_test",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    args = parser.parse_args()

    all_results = {}
    for ds_name in DS_NAMES:
        total_correct = 0
        total_count = 0
        for rank in range(args.num_gpus):
            fname = os.path.join(args.out_dir, f"{ds_name}_rank{rank}.json")
            if not os.path.exists(fname):
                print(f"Warning: missing {fname}")
                continue
            with open(fname) as f:
                data = json.load(f)
            total_correct += data["correct"]
            total_count += data["total"]

        if total_count > 0:
            acc = total_correct / total_count
            all_results[ds_name] = {
                "correct": total_correct,
                "total": total_count,
                "accuracy": acc,
            }
            print(f"{ds_name}: {total_correct}/{total_count} = {acc:.4f}")

    if all_results:
        avg_acc = np.mean([r["accuracy"] for r in all_results.values()])
        all_results["average"] = {"accuracy": avg_acc}
        print(f"\nAverage accuracy: {avg_acc:.4f}")

    summary_file = os.path.join(args.out_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved merged summary to {summary_file}")


if __name__ == "__main__":
    main()
