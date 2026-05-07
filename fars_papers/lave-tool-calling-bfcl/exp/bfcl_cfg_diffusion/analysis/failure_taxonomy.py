"""Failure taxonomy classifier for BFCL tool-calling experiments.

Classifies each model output into one of four mutually exclusive categories:
  Success       -- BFCL AST evaluator returns True
  Parse Failure -- ast.parse() raises an exception on the cleaned output
  Wrong Function-- parses OK but function name set doesn't match ground truth
  Wrong Arguments-- function names match but BFCL eval still fails (bad args)

Usage:
  python -m bfcl_cfg_diffusion.analysis.failure_taxonomy \
      --results-dirs unconstrained best_of_2 lave_cfg_v3 \
      --seeds 42 123 456 \
      --output bfcl_cfg_diffusion/scores/failure_taxonomy_results.json
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
GORILLA_ROOT = EXP_ROOT / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

from bfcl_eval.model_handler.utils import default_decode_ast_prompting
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.utils import is_java, is_js, is_relevance_or_irrelevance, load_file

DATA_DIR = GORILLA_ROOT / "bfcl_eval" / "data"
POSSIBLE_ANSWER_DIR = DATA_DIR / "possible_answer"
RESULTS_ROOT = EXP_ROOT / "bfcl_cfg_diffusion" / "results"
SAMPLED_DATA = EXP_ROOT / "bfcl_cfg_diffusion" / "data" / "bfcl_nonlive_300.json"

NON_LIVE_CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple", "irrelevance",
]
VERSION_PREFIX = "BFCL_v4"

CATEGORY_DISPLAY = {
    "Simple": ["simple_python", "simple_java", "simple_javascript"],
    "Multiple": ["multiple"],
    "Parallel": ["parallel"],
    "Parallel Multiple": ["parallel_multiple"],
    "Irrelevance": ["irrelevance"],
}

CONDITION_LABELS = {
    "unconstrained": "(A) Unconstrained",
    "best_of_2": "(B) Best-of-2",
    "lave_cfg_v3": "(C) LAVE CFG",
}


def load_prompt_entries(category: str) -> dict:
    path = DATA_DIR / f"{VERSION_PREFIX}_{category}.json"
    entries = load_file(str(path), sort_by_id=False, use_lock=False)
    return {e["id"]: e for e in entries}


def load_ground_truth_entries(category: str) -> dict:
    if is_relevance_or_irrelevance(category):
        return {}
    path = POSSIBLE_ANSWER_DIR / f"{VERSION_PREFIX}_{category}.json"
    entries = load_file(str(path), sort_by_id=False, use_lock=False)
    return {e["id"]: e for e in entries}


def get_language_and_format(category: str):
    if is_java(category):
        return Language.JAVA, ReturnFormat.JAVA
    elif is_js(category):
        return Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
    return Language.PYTHON, ReturnFormat.PYTHON


def load_sampled_data() -> dict:
    with open(SAMPLED_DATA) as f:
        data = json.load(f)
    return {d["id"]: d for d in data}


def clean_output(result_str: str) -> str:
    cleaned = result_str.strip("`\n ")
    if not cleaned.startswith("["):
        cleaned = "[" + cleaned
    if not cleaned.endswith("]"):
        cleaned = cleaned + "]"
    return cleaned


def try_ast_parse(result_str: str):
    cleaned = clean_output(result_str)
    try:
        tree = ast.parse(cleaned)
        return tree, cleaned
    except (SyntaxError, ValueError):
        return None, cleaned


def extract_func_name(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        prefix = extract_func_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    elif isinstance(node, ast.Subscript):
        return extract_func_name(node.value)
    return ""


def extract_call_names(tree) -> list:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = extract_func_name(node.func)
            if name:
                names.append(name)
    return names


def get_gt_func_names(ground_truth) -> list:
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    names = []
    for item in ground_truth:
        if isinstance(item, dict):
            names.extend(item.keys())
    return names


def check_bfcl_success(category, example_id, model_output, prompt_entry, gt_entry):
    if is_relevance_or_irrelevance(category):
        return _check_relevance_success(category, model_output)
    language, return_format = get_language_and_format(category)
    try:
        decoded = default_decode_ast_prompting(model_output, return_format, has_tool_call_tag=False)
    except Exception:
        return False
    from bfcl_eval.model_handler.utils import is_function_calling_format_output
    if not is_function_calling_format_output(decoded):
        return False
    try:
        result = ast_checker(
            prompt_entry["function"], decoded, gt_entry["ground_truth"],
            language, category, "llada_unconstrained",
        )
        return result["valid"]
    except Exception:
        return False


def _check_relevance_success(category, model_output):
    contain_func_call = False
    try:
        decoded = default_decode_ast_prompting(model_output, ReturnFormat.PYTHON, has_tool_call_tag=False)
        contain_func_call = True
        from bfcl_eval.model_handler.utils import is_empty_output
        if is_empty_output(decoded):
            contain_func_call = False
    except Exception:
        contain_func_call = False
    if "irrelevance" in category:
        return not contain_func_call
    return contain_func_call


def classify_entry(category, example_id, model_output, prompt_entry, gt_entry, sampled_entry):
    if check_bfcl_success(category, example_id, model_output, prompt_entry, gt_entry):
        return "success"

    tree, _ = try_ast_parse(model_output)
    if tree is None:
        return "parse_failure"

    if is_relevance_or_irrelevance(category):
        gt_raw = sampled_entry.get("ground_truth", "[]")
        if isinstance(gt_raw, str):
            gt_raw = json.loads(gt_raw)
        if gt_raw == []:
            pred_names = extract_call_names(tree)
            if len(pred_names) > 0:
                return "wrong_function"
            return "wrong_arguments"
        return "wrong_function"

    gt_names = get_gt_func_names(sampled_entry.get("ground_truth", "[]"))
    pred_names = extract_call_names(tree)

    gt_name_sorted = sorted(gt_names)
    pred_name_sorted = sorted(pred_names)

    if gt_name_sorted != pred_name_sorted:
        return "wrong_function"

    return "wrong_arguments"


def classify_seed(result_path: str, sampled_data: dict):
    results = []
    with open(result_path) as f:
        for line in f:
            results.append(json.loads(line.strip()))

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    classifications = []

    for cat in NON_LIVE_CATEGORIES:
        cat_results = by_category.get(cat, [])
        if not cat_results:
            continue

        prompts = load_prompt_entries(cat)
        gt_entries = load_ground_truth_entries(cat)

        for r in cat_results:
            eid = r["id"]
            output = r["result"]
            prompt_entry = prompts.get(eid, {})
            gt_entry = gt_entries.get(eid, {})
            sampled = sampled_data.get(eid, {})

            label = classify_entry(cat, eid, output, prompt_entry, gt_entry, sampled)
            classifications.append({
                "id": eid,
                "category": cat,
                "label": label,
            })

    return classifications


def aggregate_classifications(classifications: list) -> dict:
    labels = ["success", "parse_failure", "wrong_function", "wrong_arguments"]
    total = len(classifications)
    counts = defaultdict(int)
    by_category = defaultdict(lambda: defaultdict(int))
    for c in classifications:
        counts[c["label"]] += 1
        by_category[c["category"]][c["label"]] += 1

    result = {"total": total, "overall": {}, "per_category": {}}
    for label in labels:
        result["overall"][label] = {"count": counts[label], "pct": counts[label] / total * 100 if total else 0}

    for cat in NON_LIVE_CATEGORIES:
        cat_total = sum(by_category[cat].values())
        result["per_category"][cat] = {"total": cat_total}
        for label in labels:
            c = by_category[cat][label]
            result["per_category"][cat][label] = {
                "count": c, "pct": c / cat_total * 100 if cat_total else 0
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirs", nargs="+", default=["unconstrained", "best_of_2", "lave_cfg_v3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sampled_data = load_sampled_data()
    all_results = {}

    for cond_dir in args.results_dirs:
        cond_label = CONDITION_LABELS.get(cond_dir, cond_dir)
        seed_aggs = {}
        per_seed_detailed = {}

        for seed in args.seeds:
            result_path = RESULTS_ROOT / cond_dir / f"seed_{seed}.jsonl"
            if not result_path.exists():
                print(f"WARNING: {result_path} not found, skipping")
                continue
            classifications = classify_seed(str(result_path), sampled_data)
            agg = aggregate_classifications(classifications)
            seed_aggs[seed] = agg
            per_seed_detailed[seed] = classifications

        if not seed_aggs:
            continue

        labels = ["success", "parse_failure", "wrong_function", "wrong_arguments"]
        mean_overall = {}
        for label in labels:
            counts = [sa["overall"][label]["count"] for sa in seed_aggs.values()]
            pcts = [sa["overall"][label]["pct"] for sa in seed_aggs.values()]
            mean_overall[label] = {
                "count_mean": float(np.mean(counts)),
                "count_std": float(np.std(counts)),
                "pct_mean": float(np.mean(pcts)),
                "pct_std": float(np.std(pcts)),
            }

        mean_per_category = {}
        for cat in NON_LIVE_CATEGORIES:
            mean_per_category[cat] = {}
            for label in labels:
                counts = [sa["per_category"].get(cat, {}).get(label, {}).get("count", 0) for sa in seed_aggs.values()]
                pcts = [sa["per_category"].get(cat, {}).get(label, {}).get("pct", 0) for sa in seed_aggs.values()]
                mean_per_category[cat][label] = {
                    "count_mean": float(np.mean(counts)),
                    "count_std": float(np.std(counts)),
                    "pct_mean": float(np.mean(pcts)),
                    "pct_std": float(np.std(pcts)),
                }
            totals = [sa["per_category"].get(cat, {}).get("total", 0) for sa in seed_aggs.values()]
            mean_per_category[cat]["total_mean"] = float(np.mean(totals))

        all_results[cond_dir] = {
            "label": cond_label,
            "seeds": list(seed_aggs.keys()),
            "mean_overall": mean_overall,
            "mean_per_category": mean_per_category,
            "per_seed": {str(s): a for s, a in seed_aggs.items()},
        }

    _print_summary_table(all_results, args.results_dirs)

    output_path = args.output or str(EXP_ROOT / "bfcl_cfg_diffusion" / "scores" / "failure_taxonomy_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return all_results


def _print_summary_table(all_results, cond_dirs):
    labels = ["success", "parse_failure", "wrong_function", "wrong_arguments"]
    label_display = {
        "success": "Success",
        "parse_failure": "Parse Failure",
        "wrong_function": "Wrong Function",
        "wrong_arguments": "Wrong Arguments",
    }

    header = f"{'Error Type':<20}"
    for cond in cond_dirs:
        r = all_results.get(cond)
        if r:
            header += f" | {r['label']:>20}"
    print("\n" + "=" * 80)
    print("FAILURE TAXONOMY (mean over seeds)")
    print("=" * 80)
    print(header)
    print("-" * len(header))

    for label in labels:
        row = f"{label_display[label]:<20}"
        for cond in cond_dirs:
            r = all_results.get(cond)
            if r:
                m = r["mean_overall"][label]
                row += f" | {m['count_mean']:>6.1f} ({m['pct_mean']:>5.1f}%)"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
