"""Score Condition A outputs: main accuracy, per-family breakdown, and alternate-convention match rate.

For Condition A the prompt ends with 'FINAL_ANSWER:' so the model's raw_output
IS the answer continuation. We extract the answer using: (1) \boxed{} if present,
(2) first line if it is a short direct answer, (3) last \boxed{} as fallback.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dut_project.evaluation.score import score_item, aggregate_scores, save_results
from dut_project.inference.parse_outputs import _normalize_answer


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def extract_answer_condition_a(raw_output: str) -> str | None:
    text = raw_output.strip()
    if not text:
        return None

    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        ans = boxed[-1].strip()
        ans = re.sub(r"^\\text\{", "", ans)
        ans = ans.rstrip("}")
        return _normalize_answer(ans)

    first_line = text.split("\n")[0].strip().rstrip("]").rstrip("[").strip()
    if re.match(r"^-?\d+(\.\d+)?$", first_line):
        return _normalize_answer(first_line)
    if first_line.lower().rstrip(".,!") in ("yes", "no"):
        return _normalize_answer(first_line.rstrip(".,!"))

    return _normalize_answer(first_line)


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


def main():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(proj, "dut_project/outputs/qwen25_math_7b/condition_a.jsonl")
    bench_path = os.path.join(proj, "dut_project/data/erdos_conventions_bench.jsonl")
    results_path = os.path.join(proj, "dut_project/results/qwen25_math_7b/condition_a_results.json")

    outputs = load_jsonl(output_path)
    bench_items = load_jsonl(bench_path)
    bench_map = {item["item_id"]: item for item in bench_items}

    scored = []
    for out in outputs:
        item = bench_map[out["item_id"]]
        final_answer = extract_answer_condition_a(out["raw_output"])
        parsed = {"checks": [], "final_answer": final_answer}
        scored.append(score_item(parsed, item, condition="A"))

    agg = aggregate_scores(scored)
    alt = compute_alternate_match_rate(scored, bench_map)
    agg["alternate_convention"] = alt

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    save_results(agg, results_path)

    print(json.dumps(agg, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
