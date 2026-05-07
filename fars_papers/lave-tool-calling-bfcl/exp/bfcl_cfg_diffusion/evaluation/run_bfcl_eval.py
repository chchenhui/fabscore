"""
BFCL AST-based evaluation for unconstrained/constrained diffusion outputs.
Reads raw model outputs from JSONL files, runs BFCL evaluation functions directly,
computes per-category and overall success rates + AST parse rates.
Reusable for Conditions A, B, and C.
"""
import argparse
import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
GORILLA_ROOT = EXP_ROOT / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    ast_parse,
)
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.utils import is_java, is_js, is_relevance_or_irrelevance, load_file

DATA_DIR = GORILLA_ROOT / "bfcl_eval" / "data"
POSSIBLE_ANSWER_DIR = DATA_DIR / "possible_answer"

NON_LIVE_CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple", "irrelevance",
]

VERSION_PREFIX = "BFCL_v4"


def load_prompt_entries(category: str) -> dict:
    path = DATA_DIR / f"{VERSION_PREFIX}_{category}.json"
    entries = load_file(str(path), sort_by_id=False, use_lock=False)
    return {e["id"]: e for e in entries}


def load_ground_truth(category: str) -> dict:
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
    else:
        return Language.PYTHON, ReturnFormat.PYTHON


def check_ast_parseable(result_str: str) -> bool:
    cleaned = result_str.strip("`\n ")
    if not cleaned.startswith("["):
        cleaned = "[" + cleaned
    if not cleaned.endswith("]"):
        cleaned = cleaned + "]"
    try:
        ast.parse(cleaned)
        return True
    except (SyntaxError, ValueError):
        return False


def evaluate_single_entry(category, example_id, model_output, prompt_entry, gt_entry):
    if is_relevance_or_irrelevance(category):
        return evaluate_relevance_entry(category, model_output)

    language, return_format = get_language_and_format(category)

    try:
        decoded = default_decode_ast_prompting(model_output, return_format, has_tool_call_tag=False)
    except Exception:
        return False

    from bfcl_eval.model_handler.utils import is_function_calling_format_output
    if not is_function_calling_format_output(decoded):
        return False

    try:
        checker_result = ast_checker(
            prompt_entry["function"],
            decoded,
            gt_entry["ground_truth"],
            language,
            category,
            "llada_unconstrained",
        )
        return checker_result["valid"]
    except Exception:
        return False


def evaluate_relevance_entry(category, model_output):
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
    else:
        return contain_func_call


def evaluate_seed(result_path: str, sampled_ids: set = None):
    results = []
    with open(result_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            results.append(entry)

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    category_metrics = {}
    all_correct = 0
    all_total = 0
    all_ast_ok = 0

    for cat in NON_LIVE_CATEGORIES:
        cat_results = by_category.get(cat, [])
        if not cat_results:
            category_metrics[cat] = {"success_rate": 0.0, "ast_parse_rate": 0.0, "total": 0, "correct": 0}
            continue

        prompts = load_prompt_entries(cat)
        gt = load_ground_truth(cat)

        correct = 0
        ast_ok = 0
        total = len(cat_results)

        for r in cat_results:
            eid = r["id"]
            output = r["result"]

            if check_ast_parseable(output):
                ast_ok += 1

            prompt_entry = prompts.get(eid, {})
            gt_entry = gt.get(eid, {})

            if evaluate_single_entry(cat, eid, output, prompt_entry, gt_entry):
                correct += 1

        category_metrics[cat] = {
            "success_rate": correct / total if total > 0 else 0.0,
            "ast_parse_rate": ast_ok / total if total > 0 else 0.0,
            "total": total,
            "correct": correct,
        }
        all_correct += correct
        all_total += total
        all_ast_ok += ast_ok

    overall_success = all_correct / all_total if all_total > 0 else 0.0
    overall_ast = all_ast_ok / all_total if all_total > 0 else 0.0

    wall_times = [r.get("wall_time", 0) for r in results if "wall_time" in r]
    mean_time = np.mean(wall_times) if wall_times else 0.0

    return {
        "overall_success_rate": overall_success,
        "overall_ast_parse_rate": overall_ast,
        "total_examples": all_total,
        "total_correct": all_correct,
        "mean_inference_time": mean_time,
        "per_category": category_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with seed_*.jsonl files")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for aggregated results")
    parser.add_argument("--condition", type=str, default="condition_a",
                        help="Condition name for WandB logging")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seed_results = {}

    for seed in args.seeds:
        result_file = results_dir / f"seed_{seed}.jsonl"
        if not result_file.exists():
            print(f"WARNING: {result_file} not found, skipping seed {seed}")
            continue

        print(f"Evaluating seed {seed}...")
        metrics = evaluate_seed(str(result_file))
        seed_results[seed] = metrics

        print(f"  Seed {seed}: success={metrics['overall_success_rate']:.4f} "
              f"ast_parse={metrics['overall_ast_parse_rate']:.4f} "
              f"mean_time={metrics['mean_inference_time']:.2f}s")
        for cat, cm in metrics["per_category"].items():
            print(f"    {cat}: {cm['correct']}/{cm['total']} = {cm['success_rate']:.4f}")

    if not seed_results:
        print("ERROR: No seed results found")
        sys.exit(1)

    success_rates = [m["overall_success_rate"] for m in seed_results.values()]
    ast_rates = [m["overall_ast_parse_rate"] for m in seed_results.values()]
    times = [m["mean_inference_time"] for m in seed_results.values()]

    aggregated = {
        "condition": args.condition,
        "seeds": list(seed_results.keys()),
        "overall_success_rate_mean": float(np.mean(success_rates)),
        "overall_success_rate_std": float(np.std(success_rates)),
        "overall_ast_parse_rate_mean": float(np.mean(ast_rates)),
        "overall_ast_parse_rate_std": float(np.std(ast_rates)),
        "mean_inference_time": float(np.mean(times)),
        "per_seed": {},
        "per_category_mean": {},
    }

    for seed, metrics in seed_results.items():
        aggregated["per_seed"][str(seed)] = {
            "success_rate": metrics["overall_success_rate"],
            "ast_parse_rate": metrics["overall_ast_parse_rate"],
            "mean_inference_time": metrics["mean_inference_time"],
            "per_category": metrics["per_category"],
        }

    for cat in NON_LIVE_CATEGORIES:
        cat_success = []
        cat_ast = []
        for metrics in seed_results.values():
            cm = metrics["per_category"].get(cat, {})
            if cm.get("total", 0) > 0:
                cat_success.append(cm["success_rate"])
                cat_ast.append(cm["ast_parse_rate"])
        if cat_success:
            aggregated["per_category_mean"][cat] = {
                "success_rate_mean": float(np.mean(cat_success)),
                "success_rate_std": float(np.std(cat_success)),
                "ast_parse_rate_mean": float(np.mean(cat_ast)),
                "ast_parse_rate_std": float(np.std(cat_ast)),
            }

    print("\n=== Aggregated Results ===")
    print(f"Overall success rate: {aggregated['overall_success_rate_mean']:.4f} "
          f"+/- {aggregated['overall_success_rate_std']:.4f}")
    print(f"Overall AST parse rate: {aggregated['overall_ast_parse_rate_mean']:.4f} "
          f"+/- {aggregated['overall_ast_parse_rate_std']:.4f}")
    print(f"Mean inference time: {aggregated['mean_inference_time']:.2f}s")

    for cat, cm in aggregated["per_category_mean"].items():
        print(f"  {cat}: success={cm['success_rate_mean']:.4f} +/- {cm['success_rate_std']:.4f}")

    output_path = args.output or str(results_dir / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nResults saved to {output_path}")

    try:
        from dotenv import load_dotenv
        load_dotenv(EXP_ROOT / ".env")
        import wandb

        wandb_mode = os.environ.get("WANDB_MODE", "offline")
        wandb_project = os.environ.get("WANDB_PROJECT", "lave-tool-calling-bfcl")

        for seed, metrics in seed_results.items():
            run = wandb.init(
                project=wandb_project,
                name=f"{args.condition}_seed_{seed}",
                mode=wandb_mode,
                config={"seed": seed, "condition": args.condition},
                reinit=True,
            )
            run.log({
                "success_rate": metrics["overall_success_rate"],
                "ast_parse_rate": metrics["overall_ast_parse_rate"],
                "mean_inference_time": metrics["mean_inference_time"],
                **{f"success_rate/{cat}": cm["success_rate"]
                   for cat, cm in metrics["per_category"].items()},
                **{f"ast_parse_rate/{cat}": cm["ast_parse_rate"]
                   for cat, cm in metrics["per_category"].items()},
            })
            run.finish()

        summary_run = wandb.init(
            project=wandb_project,
            name=f"{args.condition}_summary",
            mode=wandb_mode,
            config={"condition": args.condition, "seeds": list(seed_results.keys())},
            reinit=True,
        )
        summary_run.log({
            "success_rate_mean": aggregated["overall_success_rate_mean"],
            "success_rate_std": aggregated["overall_success_rate_std"],
            "ast_parse_rate_mean": aggregated["overall_ast_parse_rate_mean"],
            "ast_parse_rate_std": aggregated["overall_ast_parse_rate_std"],
            "mean_inference_time": aggregated["mean_inference_time"],
            **{f"success_rate_mean/{cat}": cm["success_rate_mean"]
               for cat, cm in aggregated["per_category_mean"].items()},
        })
        summary_run.finish()
        print("WandB logging completed")
    except Exception as e:
        print(f"WandB logging failed (non-critical): {e}")

    return aggregated


if __name__ == "__main__":
    main()
