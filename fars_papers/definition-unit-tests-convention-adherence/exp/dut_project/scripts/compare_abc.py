"""Compare Conditions A, B, C for Qwen2.5-Math-7B-Instruct.

Builds a comparison table with overall/per-family main accuracy, check accuracy,
alternate match rates, and paired bootstrap 95% CI for C minus B.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dut_project.evaluation.statistics import paired_bootstrap_ci
from dut_project.inference.parse_outputs import _normalize_answer


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def extract_final_answer_robust(raw_output: str) -> str | None:
    text = raw_output.strip()
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        ans = boxed[-1].strip()
        ans = re.sub(r"^\\text\{", "", ans)
        ans = ans.rstrip("}")
        return _normalize_answer(ans)
    first_line = text.split("\n")[0].strip()
    return _normalize_answer(first_line)


def get_main_correct(outputs, bench_map, condition):
    results = []
    for out in outputs:
        item = bench_map[out["item_id"]]
        gt = _normalize_answer(item["ground_truth_glossary"])
        pred = extract_final_answer_robust(out["raw_output"])
        results.append({
            "item_id": out["item_id"],
            "family": item["family"],
            "correct": pred == gt if pred is not None else False,
        })
    return results


def main():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bench_path = os.path.join(proj, "dut_project/data/erdos_conventions_bench.jsonl")
    out_a_path = os.path.join(proj, "dut_project/outputs/qwen25_math_7b/condition_a.jsonl")
    out_b_path = os.path.join(proj, "dut_project/outputs/qwen25_math_7b/condition_b.jsonl")
    out_c_path = os.path.join(proj, "dut_project/outputs/qwen25_math_7b/condition_c.jsonl")
    res_a_path = os.path.join(proj, "dut_project/results/qwen25_math_7b/condition_a_results.json")
    res_b_path = os.path.join(proj, "dut_project/results/qwen25_math_7b/condition_b_results.json")
    res_c_path = os.path.join(proj, "dut_project/results/qwen25_math_7b/condition_c_results.json")
    compare_path = os.path.join(proj, "dut_project/results/qwen25_math_7b/abc_comparison.json")

    bench_items = load_jsonl(bench_path)
    bench_map = {item["item_id"]: item for item in bench_items}

    out_a = load_jsonl(out_a_path)
    out_b = load_jsonl(out_b_path)
    out_c = load_jsonl(out_c_path)

    res_a = json.load(open(res_a_path))
    res_b = json.load(open(res_b_path))
    res_c = json.load(open(res_c_path))

    corr_b = get_main_correct(out_b, bench_map, "B")
    corr_c = get_main_correct(out_c, bench_map, "C")

    id_order_b = [x["item_id"] for x in corr_b]
    id_order_c = [x["item_id"] for x in corr_c]

    c_map = {x["item_id"]: x["correct"] for x in corr_c}
    b_map = {x["item_id"]: x["correct"] for x in corr_b}

    all_ids = sorted(set(id_order_b) & set(id_order_c))
    correct_c_aligned = [c_map[i] for i in all_ids]
    correct_b_aligned = [b_map[i] for i in all_ids]

    bootstrap = paired_bootstrap_ci(
        correct_a=correct_c_aligned,
        correct_b=correct_b_aligned,
        n_resamples=10000,
        ci_level=0.95,
        seed=42,
    )

    families = ["asymptotics", "completeness", "convolution"]
    per_family_comparison = {}
    for fam in families:
        per_family_comparison[fam] = {
            "A": res_a["per_family"].get(fam, {}).get("accuracy"),
            "B": res_b["per_family"].get(fam, {}).get("accuracy"),
            "C": res_c["per_family"].get(fam, {}).get("accuracy"),
        }

    comparison = {
        "model": "Qwen/Qwen2.5-Math-7B-Instruct",
        "overall_main_accuracy": {
            "A": res_a["main_accuracy"],
            "B": res_b["main_accuracy"],
            "C": res_c["main_accuracy"],
        },
        "per_family_main_accuracy": per_family_comparison,
        "check_accuracy": {
            "B": res_b.get("check_accuracy"),
            "C": res_c.get("check_accuracy"),
        },
        "per_check_accuracy": {
            "B": res_b.get("per_check_accuracy"),
            "C": res_c.get("per_check_accuracy"),
        },
        "alternate_convention_match_rate": {
            "A": res_a.get("alternate_convention", {}).get("alternate_match_rate"),
            "B": res_b.get("alternate_convention", {}).get("alternate_match_rate"),
            "C": res_c.get("alternate_convention", {}).get("alternate_match_rate"),
        },
        "bootstrap_ci_c_minus_b": {
            "observed_diff": bootstrap["observed_diff"],
            "ci_lower": bootstrap["ci_lower"],
            "ci_upper": bootstrap["ci_upper"],
            "ci_level": bootstrap["ci_level"],
            "n_resamples": bootstrap["n_resamples"],
            "excludes_zero": bootstrap["excludes_zero"],
            "n_items": bootstrap["n_items"],
            "mean_c": bootstrap["mean_a"],
            "mean_b": bootstrap["mean_b"],
        },
    }

    os.makedirs(os.path.dirname(compare_path), exist_ok=True)
    with open(compare_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps(comparison, indent=2))
    print(f"\nComparison saved to {compare_path}")


if __name__ == "__main__":
    main()
