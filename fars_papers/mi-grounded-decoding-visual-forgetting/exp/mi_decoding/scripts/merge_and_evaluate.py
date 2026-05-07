# Merge shard JSONL files and compute evaluation metrics.
# Usage: python merge_and_evaluate.py --benchmark mmstar --output_dir <path> --results_file <path>
import argparse
import json
import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mi_decoding.evaluation.metrics import compute_mmstar_accuracy, compute_hallusionbench_aacc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["mmstar", "hallusionbench"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--results_file", default=None)
    args = parser.parse_args()

    shard_files = sorted(glob.glob(os.path.join(args.output_dir, "shard_*.jsonl")))
    if not shard_files:
        print(f"No shard files found in {args.output_dir}")
        return

    all_predictions = []
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                all_predictions.append(json.loads(line.strip()))
    print(f"Loaded {len(all_predictions)} predictions from {len(shard_files)} shards")

    merged_file = os.path.join(args.output_dir, "all_predictions.jsonl")
    with open(merged_file, "w") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    print(f"Merged predictions saved to {merged_file}")

    if args.benchmark == "mmstar":
        acc = compute_mmstar_accuracy(all_predictions)
        print(f"\nMMStar Results:")
        print(f"  Total: {len(all_predictions)}")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        results = {
            "benchmark": "mmstar",
            "total": len(all_predictions),
            "accuracy": round(acc * 100, 2),
        }
    else:
        metrics = compute_hallusionbench_aacc(all_predictions)
        print(f"\nHallusionBench Results:")
        print(f"  Total: {metrics['total']}")
        print(f"  Correct: {metrics['correct']}")
        print(f"  aAcc: {metrics['aAcc']:.4f} ({metrics['aAcc']*100:.2f}%)")
        print(f"  VD acc: {metrics['vd_acc']:.4f} ({metrics['vd_acc']*100:.2f}%)")
        print(f"  VS acc: {metrics['vs_acc']:.4f} ({metrics['vs_acc']*100:.2f}%)")
        results = {
            "benchmark": "hallusionbench",
            "total": metrics["total"],
            "correct": metrics["correct"],
            "aAcc": round(metrics["aAcc"] * 100, 2),
            "vd_acc": round(metrics["vd_acc"] * 100, 2),
            "vs_acc": round(metrics["vs_acc"] * 100, 2),
        }

    if args.results_file:
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.results_file}")


if __name__ == "__main__":
    main()
