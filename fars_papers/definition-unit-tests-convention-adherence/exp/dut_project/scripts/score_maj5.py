"""Majority-vote scoring for multi-sample inference outputs (A@5, B@5).

Groups 5 samples per item by item_id, parses answers, applies majority vote
(ties broken randomly with seed=42), computes accuracy and per-family breakdown.
Logs summary metrics to WandB.

Usage:
  python dut_project/scripts/score_maj5.py \
    --outputs dut_project/outputs/qwen25_math_7b/condition_a_maj5.jsonl \
    --bench dut_project/data/erdos_conventions_bench.jsonl \
    --results dut_project/results/qwen25_math_7b/condition_a_maj5_results.json \
    --model-type qwen \
    --condition A \
    --run-name qwen_condition_a_maj5
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from dut_project.inference.parse_outputs import _normalize_answer, extract_answer_robust, extract_boxed_answer


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def extract_answer_qwen(raw_output, family=None):
    text = raw_output.strip()
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    first_line = text.split("\n")[0].strip().rstrip("]").rstrip("[").strip()
    if first_line:
        return _normalize_answer(first_line)
    return None


def majority_vote(answers, seed=42):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    candidates = [a for a, c in counts.items() if c == max_count]
    if len(candidates) == 1:
        return candidates[0]
    rng = random.Random(seed)
    return rng.choice(sorted(candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model-type", required=True, choices=["qwen", "llama"])
    parser.add_argument("--condition", required=True, choices=["A", "B"])
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    outputs = load_jsonl(args.outputs)
    bench_items = load_jsonl(args.bench)
    bench_map = {item["item_id"]: item for item in bench_items}

    grouped = {}
    for o in outputs:
        iid = o["item_id"]
        if iid not in grouped:
            grouped[iid] = []
        grouped[iid].append(o)

    scored = []
    per_fam = {}
    alt_count = 0
    total = 0

    for iid in sorted(grouped.keys()):
        samples = grouped[iid]
        item = bench_map[iid]
        gt = _normalize_answer(item["ground_truth_glossary"])
        gt_alt = _normalize_answer(item["ground_truth_alternate"])
        family = item["family"]

        answers = []
        for s in samples:
            ans = extract_answer_robust(s["raw_output"], family=family)
            answers.append(ans)

        voted = majority_vote(answers, seed=42)
        correct = (voted == gt) if voted else False

        if voted and voted == gt_alt:
            alt_count += 1
        total += 1

        if family not in per_fam:
            per_fam[family] = {"correct": 0, "total": 0}
        per_fam[family]["total"] += 1
        if correct:
            per_fam[family]["correct"] += 1

        scored.append({
            "item_id": iid,
            "family": family,
            "ground_truth": gt,
            "ground_truth_alt": gt_alt,
            "sample_answers": answers,
            "voted_answer": voted,
            "correct": correct,
        })

    overall_correct = sum(1 for s in scored if s["correct"])
    overall_accuracy = overall_correct / total if total > 0 else 0.0

    results = {
        "model_type": args.model_type,
        "condition": args.condition,
        "method": f"{args.condition}@5 (majority vote)",
        "num_samples": 5,
        "overall_accuracy": overall_accuracy,
        "overall_correct": overall_correct,
        "total": total,
        "per_family": {
            fam: {
                "accuracy": pf["correct"] / pf["total"] if pf["total"] > 0 else 0.0,
                "correct": pf["correct"],
                "total": pf["total"],
            }
            for fam, pf in per_fam.items()
        },
        "alt_rate": alt_count / total if total > 0 else 0.0,
        "alt_count": alt_count,
        "per_item": scored,
    }

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.results}")
    print(f"Overall accuracy: {overall_accuracy:.1%} ({overall_correct}/{total})")
    for fam in sorted(per_fam.keys()):
        pf = per_fam[fam]
        acc = pf["correct"] / pf["total"] if pf["total"] > 0 else 0.0
        print(f"  {fam}: {acc:.0%} ({pf['correct']}/{pf['total']})")
    print(f"Alt rate: {alt_count/total:.1%}")

    try:
        import wandb
        wandb_project = os.environ.get("WANDB_PROJECT", "definition-unit-tests-convention-adherence")
        wandb.init(
            project=wandb_project,
            name=args.run_name,
            config={
                "model_type": args.model_type,
                "condition": args.condition,
                "method": f"{args.condition}@5",
                "num_samples": 5,
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 512,
            },
        )
        wandb.log({
            "overall_accuracy": overall_accuracy,
            "overall_correct": overall_correct,
            "total_items": total,
            "alt_rate": alt_count / total if total > 0 else 0.0,
        })
        for fam, pf in per_fam.items():
            acc = pf["correct"] / pf["total"] if pf["total"] > 0 else 0.0
            wandb.log({f"{fam}_accuracy": acc})
        wandb.finish()
        print("WandB logging complete.")
    except Exception as e:
        print(f"WandB logging failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
