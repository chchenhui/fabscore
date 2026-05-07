"""Score Condition C outputs: main accuracy, per-family breakdown, check accuracy, alt match rate.

Identical pipeline to Condition B scoring but uses condition="C" which maps to
discriminative_checks ground truths in score_item().
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dut_project.evaluation.score import score_item, aggregate_scores, save_results
from dut_project.inference.parse_outputs import _normalize_answer, parse_output


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def extract_final_answer_robust(raw_output: str, parsed_final: str | None) -> str | None:
    text = raw_output.strip()
    if not text:
        return None

    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        ans = boxed[-1].strip()
        ans = re.sub(r"^\\text\{", "", ans)
        ans = ans.rstrip("}")
        return _normalize_answer(ans)

    if parsed_final is not None:
        return _normalize_answer(parsed_final)

    return None


def compute_alternate_match_rate(scored_items, bench_map):
    total = 0
    alt_matches = 0
    for s in scored_items:
        item = bench_map[s["item_id"]]
        predicted = s["predicted_answer"]
        if predicted is not None:
            gt_alt = item["ground_truth_alternate"]
            if _normalize_answer(predicted) == _normalize_answer(gt_alt):
                alt_matches += 1
        total += 1
    return {"alternate_match_rate": alt_matches / total if total > 0 else 0.0,
            "alternate_matches": alt_matches, "total": total}


def compute_per_check_accuracy(scored_items):
    per_check = {}
    for s in scored_items:
        for cr in s["check_results"]:
            idx = cr["check_idx"]
            if idx not in per_check:
                per_check[idx] = {"correct": 0, "total": 0}
            per_check[idx]["total"] += 1
            per_check[idx]["correct"] += int(cr["correct"])
    result = {}
    for idx in sorted(per_check):
        c = per_check[idx]
        result[f"check_{idx+1}"] = {
            "accuracy": c["correct"] / c["total"] if c["total"] > 0 else 0.0,
            **c,
        }
    return result


def main():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(proj, "dut_project/outputs/qwen25_math_7b/condition_c.jsonl")
    bench_path = os.path.join(proj, "dut_project/data/erdos_conventions_bench.jsonl")
    results_path = os.path.join(proj, "dut_project/results/qwen25_math_7b/condition_c_results.json")

    outputs = load_jsonl(output_path)
    bench_items = load_jsonl(bench_path)
    bench_map = {item["item_id"]: item for item in bench_items}

    scored = []
    for out in outputs:
        item = bench_map[out["item_id"]]
        raw = out["raw_output"]
        reparsed = parse_output(raw, k=3)
        final_answer = extract_final_answer_robust(raw, reparsed["final_answer"])
        parsed = {"checks": reparsed["checks"], "final_answer": final_answer}
        scored.append(score_item(parsed, item, condition="C"))

    agg = aggregate_scores(scored)
    agg["per_check_accuracy"] = compute_per_check_accuracy(scored)
    alt = compute_alternate_match_rate(scored, bench_map)
    agg["alternate_convention"] = alt

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    save_results(agg, results_path)

    print(json.dumps(agg, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
