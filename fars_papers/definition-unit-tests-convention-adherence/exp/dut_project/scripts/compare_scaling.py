"""Compare majority-vote baselines (A@5, B@5) against single-sample conditions (A, B, C).

Computes paired bootstrap 95% CIs for (C minus A@5) and (C minus B@5) to determine
whether discriminative checks outperform simply sampling more under simpler conditions.

Usage:
  python dut_project/scripts/compare_scaling.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dut_project.evaluation.statistics import paired_bootstrap_ci
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


def get_correct_map_single(outputs, bench_map):
    correct_map = {}
    for o in outputs:
        item = bench_map[o["item_id"]]
        gt = _normalize_answer(item["ground_truth_glossary"])
        ans = extract_answer_robust(o["raw_output"], family=o["family"])
        correct_map[o["item_id"]] = (ans == gt) if ans else False
    return correct_map


def get_correct_map_maj5(results_path):
    with open(results_path) as f:
        results = json.load(f)
    correct_map = {}
    for item in results["per_item"]:
        correct_map[item["item_id"]] = item["correct"]
    return correct_map


def build_comparison(model_name, model_type, model_key, single_sample_output_key, proj):
    bench_path = os.path.join(proj, "dut_project/data/erdos_conventions_bench.jsonl")
    bench_items = load_jsonl(bench_path)
    bench_map = {item["item_id"]: item for item in bench_items}

    single_output_dir = os.path.join(proj, f"dut_project/outputs/{single_sample_output_key}")
    results_dir = os.path.join(proj, f"dut_project/results/{model_key}")

    out_a = load_jsonl(os.path.join(single_output_dir, "condition_a.jsonl"))
    out_b = load_jsonl(os.path.join(single_output_dir, "condition_b.jsonl"))
    out_c = load_jsonl(os.path.join(single_output_dir, "condition_c.jsonl"))

    correct_a = get_correct_map_single(out_a, bench_map)
    correct_b = get_correct_map_single(out_b, bench_map)
    correct_c = get_correct_map_single(out_c, bench_map)

    res_a5_path = os.path.join(results_dir, "condition_a_maj5_results.json")
    res_b5_path = os.path.join(results_dir, "condition_b_maj5_results.json")
    correct_a5 = get_correct_map_maj5(res_a5_path)
    correct_b5 = get_correct_map_maj5(res_b5_path)

    with open(res_a5_path) as f:
        res_a5 = json.load(f)
    with open(res_b5_path) as f:
        res_b5 = json.load(f)

    all_ids = sorted(
        set(correct_a.keys()) & set(correct_b.keys()) & set(correct_c.keys())
        & set(correct_a5.keys()) & set(correct_b5.keys())
    )

    c_arr = [correct_c[i] for i in all_ids]
    a_arr = [correct_a[i] for i in all_ids]
    b_arr = [correct_b[i] for i in all_ids]
    a5_arr = [correct_a5[i] for i in all_ids]
    b5_arr = [correct_b5[i] for i in all_ids]

    ci_c_minus_a5 = paired_bootstrap_ci(c_arr, a5_arr, n_resamples=10000, ci_level=0.95, seed=42)
    ci_c_minus_b5 = paired_bootstrap_ci(c_arr, b5_arr, n_resamples=10000, ci_level=0.95, seed=42)

    def compute_acc(correct_map, ids, family=None):
        if family:
            fam_ids = [i for i in ids if bench_map[i]["family"] == family]
            return sum(correct_map[i] for i in fam_ids) / len(fam_ids) if fam_ids else 0.0
        return sum(correct_map[i] for i in ids) / len(ids) if ids else 0.0

    families = ["asymptotics", "completeness", "convolution"]

    comparison = {
        "model": model_name,
        "single_sample_output_dir": single_sample_output_key,
        "overall_accuracy": {
            "A": compute_acc(correct_a, all_ids),
            "B": compute_acc(correct_b, all_ids),
            "C": compute_acc(correct_c, all_ids),
            "A@5": res_a5["overall_accuracy"],
            "B@5": res_b5["overall_accuracy"],
        },
        "per_family_accuracy": {
            fam: {
                "A": compute_acc(correct_a, all_ids, fam),
                "B": compute_acc(correct_b, all_ids, fam),
                "C": compute_acc(correct_c, all_ids, fam),
                "A@5": res_a5["per_family"].get(fam, {}).get("accuracy", 0.0),
                "B@5": res_b5["per_family"].get(fam, {}).get("accuracy", 0.0),
            }
            for fam in families
        },
        "bootstrap_ci_c_minus_a5": {
            "observed_diff": ci_c_minus_a5["observed_diff"],
            "ci_lower": ci_c_minus_a5["ci_lower"],
            "ci_upper": ci_c_minus_a5["ci_upper"],
            "excludes_zero": ci_c_minus_a5["excludes_zero"],
            "n_resamples": ci_c_minus_a5["n_resamples"],
            "n_items": ci_c_minus_a5["n_items"],
        },
        "bootstrap_ci_c_minus_b5": {
            "observed_diff": ci_c_minus_b5["observed_diff"],
            "ci_lower": ci_c_minus_b5["ci_lower"],
            "ci_upper": ci_c_minus_b5["ci_upper"],
            "excludes_zero": ci_c_minus_b5["excludes_zero"],
            "n_resamples": ci_c_minus_b5["n_resamples"],
            "n_items": ci_c_minus_b5["n_items"],
        },
    }

    out_path = os.path.join(results_dir, "scaling_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n=== {model_name} ===")
    print(json.dumps(comparison, indent=2))
    print(f"Saved to {out_path}")
    return comparison


def main():
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    build_comparison(
        model_name="Qwen/Qwen2.5-Math-7B-Instruct",
        model_type="qwen",
        model_key="qwen25_math_7b",
        single_sample_output_key="qwen25_math_7b_v3",
        proj=proj,
    )
    build_comparison(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        model_type="llama",
        model_key="llama31_8b",
        single_sample_output_key="llama31_8b",
        proj=proj,
    )


if __name__ == "__main__":
    main()
