"""Score all conditions (A, B, C) for Llama-3.1-8B-Instruct using robust answer extraction.
Uses extract_answer_robust() which has fallback heuristics for items without \boxed{} answers:
  - 'the answer is X' pattern extraction
  - Last Yes/No for asymptotics/completeness families
  - Last numeric value for convolution family
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dut_project.inference.parse_outputs import _normalize_answer, extract_answer_robust
from dut_project.evaluation.statistics import paired_bootstrap_ci


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def score_condition(outputs, bench_map):
    per_fam = {}
    correct_map = {}
    alt_count = 0
    total = 0

    for o in outputs:
        item = bench_map[o["item_id"]]
        gt = _normalize_answer(item["ground_truth_glossary"])
        gt_alt = _normalize_answer(item["ground_truth_alternate"])
        ans = extract_answer_robust(o["raw_output"], family=o["family"])
        ok = (ans == gt) if ans else False
        correct_map[o["item_id"]] = ok
        if ans and ans == gt_alt:
            alt_count += 1
        total += 1

        fam = o["family"]
        if fam not in per_fam:
            per_fam[fam] = {"correct": 0, "total": 0}
        per_fam[fam]["total"] += 1
        if ok:
            per_fam[fam]["correct"] += 1

    overall = sum(pf["correct"] for pf in per_fam.values())
    return {
        "overall_accuracy": overall / total,
        "overall_correct": overall,
        "total": total,
        "per_family": {
            fam: {
                "accuracy": pf["correct"] / pf["total"],
                "correct": pf["correct"],
                "total": pf["total"],
            }
            for fam, pf in per_fam.items()
        },
        "alt_rate": alt_count / total,
        "alt_count": alt_count,
        "correct_map": correct_map,
    }


def main():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bench_path = os.path.join(proj, "dut_project/data/erdos_conventions_bench.jsonl")
    bench_items = load_jsonl(bench_path)
    bench_map = {item["item_id"]: item for item in bench_items}

    output_dir = os.path.join(proj, "dut_project/outputs/llama31_8b")
    results = {}

    for cond in ["A", "B", "C"]:
        path = os.path.join(output_dir, f"condition_{cond.lower()}.jsonl")
        outputs = load_jsonl(path)
        results[cond] = score_condition(outputs, bench_map)
        r = results[cond]
        print(f"Condition {cond}: {r['overall_accuracy']:.1%} ({r['overall_correct']}/{r['total']}), alt_rate={r['alt_rate']:.1%}")
        for fam in ["asymptotics", "completeness", "convolution"]:
            pf = r["per_family"][fam]
            print(f"  {fam}: {pf['accuracy']:.0%} ({pf['correct']}/{pf['total']})")

    all_ids = sorted(
        set(results["A"]["correct_map"].keys())
        & set(results["B"]["correct_map"].keys())
        & set(results["C"]["correct_map"].keys())
    )

    cb = paired_bootstrap_ci(
        [results["C"]["correct_map"][i] for i in all_ids],
        [results["B"]["correct_map"][i] for i in all_ids],
        n_resamples=10000, ci_level=0.95, seed=42,
    )
    ca = paired_bootstrap_ci(
        [results["C"]["correct_map"][i] for i in all_ids],
        [results["A"]["correct_map"][i] for i in all_ids],
        n_resamples=10000, ci_level=0.95, seed=42,
    )

    print(f"\nC-B: {cb['observed_diff']:+.1%}, CI [{cb['ci_lower']:+.1%}, {cb['ci_upper']:+.1%}], excludes_zero={cb['excludes_zero']}")
    print(f"C-A: {ca['observed_diff']:+.1%}, CI [{ca['ci_lower']:+.1%}, {ca['ci_upper']:+.1%}], excludes_zero={ca['excludes_zero']}")

    out_json = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "benchmark": "ErdosConventionsBench (300 items)",
        "decoding": {"temperature": 0.0, "max_new_tokens": 2048},
        "extraction": "robust (boxed + heuristic fallbacks)",
        "conditions": {},
        "comparison": {
            "overall_accuracy": {},
            "per_family": {},
        },
        "bootstrap_ci_c_minus_b": {
            "observed_diff": cb["observed_diff"],
            "ci_lower": cb["ci_lower"],
            "ci_upper": cb["ci_upper"],
            "excludes_zero": cb["excludes_zero"],
            "n_resamples": cb["n_resamples"],
        },
        "bootstrap_ci_c_minus_a": {
            "observed_diff": ca["observed_diff"],
            "ci_lower": ca["ci_lower"],
            "ci_upper": ca["ci_upper"],
            "excludes_zero": ca["excludes_zero"],
            "n_resamples": ca["n_resamples"],
        },
    }

    for cond in ["A", "B", "C"]:
        r = results[cond]
        out_json["conditions"][cond] = {
            "overall": r["overall_accuracy"],
            "main_correct": r["overall_correct"],
            "total": r["total"],
            "per_family": r["per_family"],
            "alt_rate": r["alt_rate"],
        }
        out_json["comparison"]["overall_accuracy"][cond] = r["overall_accuracy"]
        for fam in ["asymptotics", "completeness", "convolution"]:
            if fam not in out_json["comparison"]["per_family"]:
                out_json["comparison"]["per_family"][fam] = {}
            out_json["comparison"]["per_family"][fam][cond] = r["per_family"][fam]["accuracy"]

    results_dir = os.path.join(proj, "dut_project/results/llama31_8b")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "abc_comparison_robust.json")
    with open(results_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
