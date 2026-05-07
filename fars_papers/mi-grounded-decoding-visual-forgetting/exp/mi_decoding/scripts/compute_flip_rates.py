# Compute correct->wrong and wrong->correct flip rates between short (128) and
# long (512) budget runs for vanilla, visual replay, and adaptive MI decoding.
# Outputs a JSON with per-method, per-benchmark, and per-MMStar-category breakdowns.
import argparse
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


METHOD_CONFIGS = {
    "vanilla": {
        "short_dir": "mi_decoding/outputs/vanilla_vlaa_thinker_7b_short",
        "long_dir": "mi_decoding/outputs/vanilla_VLAA-Thinker-Qwen2.5VL-7B",
    },
    "visual_replay": {
        "short_dir": "mi_decoding/outputs/visual_replay_vlaa_thinker_7b_short",
        "long_dir": "mi_decoding/outputs/visual_replay_VLAA-Thinker-Qwen2.5VL-7B",
    },
    "adaptive_mi": {
        "short_dir": "mi_decoding/outputs/adaptive_mi_vlaa_thinker_7b_short",
        "long_dir": "mi_decoding/outputs/mi_optimized_v2",
    },
}

BENCHMARKS = ["mmstar", "hallusionbench"]


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            preds[item["id"]] = item
    return preds


def is_correct_mmstar(pred):
    return pred["extracted_answer"] == pred["gt_answer"]


def is_correct_hallusionbench(pred):
    gt = str(pred["gt_answer"])
    extracted = str(pred["extracted_answer"])
    if pred.get("category") == "VS" and str(pred.get("figure_id")) == "0":
        return extracted == gt or extracted == "2"
    return extracted == gt


def load_mmstar_categories():
    from datasets import load_dataset
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    cat_map = {}
    for i, row in enumerate(ds):
        item_id = row.get("index", i)
        cat_map[item_id] = {
            "category": row.get("category", ""),
            "l2_category": row.get("l2_category", ""),
        }
    return cat_map


def compute_flips(short_preds, long_preds, benchmark, max_items=None):
    is_correct = is_correct_mmstar if benchmark == "mmstar" else is_correct_hallusionbench

    common_ids = sorted(set(short_preds.keys()) & set(long_preds.keys()))
    if max_items is not None:
        common_ids = common_ids[:max_items]

    total = len(common_ids)
    cw = 0  # correct->wrong
    wc = 0  # wrong->correct
    cc = 0  # correct->correct
    ww = 0  # wrong->wrong

    per_item = []
    for item_id in common_ids:
        s_correct = is_correct(short_preds[item_id])
        l_correct = is_correct(long_preds[item_id])

        if s_correct and not l_correct:
            cw += 1
            flip_type = "correct_to_wrong"
        elif not s_correct and l_correct:
            wc += 1
            flip_type = "wrong_to_correct"
        elif s_correct and l_correct:
            cc += 1
            flip_type = "correct_to_correct"
        else:
            ww += 1
            flip_type = "wrong_to_wrong"

        per_item.append({"id": item_id, "flip_type": flip_type})

    result = {
        "total": total,
        "correct_to_wrong": cw,
        "wrong_to_correct": wc,
        "correct_to_correct": cc,
        "wrong_to_wrong": ww,
        "cw_rate": round(cw / total * 100, 2) if total > 0 else 0,
        "wc_rate": round(wc / total * 100, 2) if total > 0 else 0,
        "net_flip_rate": round((cw - wc) / total * 100, 2) if total > 0 else 0,
        "short_accuracy": round((cw + cc) / total * 100, 2) if total > 0 else 0,
        "long_accuracy": round((wc + cc) / total * 100, 2) if total > 0 else 0,
    }
    return result, per_item


def compute_category_flips(short_preds, long_preds, cat_map, max_items=None):
    common_ids = sorted(set(short_preds.keys()) & set(long_preds.keys()))
    if max_items is not None:
        common_ids = common_ids[:max_items]

    cat_counts = defaultdict(lambda: {"total": 0, "cw": 0, "wc": 0, "cc": 0, "ww": 0})

    for item_id in common_ids:
        info = cat_map.get(item_id, {})
        category = info.get("category", "unknown")

        s_correct = is_correct_mmstar(short_preds[item_id])
        l_correct = is_correct_mmstar(long_preds[item_id])

        cat_counts[category]["total"] += 1
        if s_correct and not l_correct:
            cat_counts[category]["cw"] += 1
        elif not s_correct and l_correct:
            cat_counts[category]["wc"] += 1
        elif s_correct and l_correct:
            cat_counts[category]["cc"] += 1
        else:
            cat_counts[category]["ww"] += 1

    result = {}
    for cat, counts in sorted(cat_counts.items()):
        t = counts["total"]
        result[cat] = {
            "total": t,
            "correct_to_wrong": counts["cw"],
            "wrong_to_correct": counts["wc"],
            "cw_rate": round(counts["cw"] / t * 100, 2) if t > 0 else 0,
            "wc_rate": round(counts["wc"] / t * 100, 2) if t > 0 else 0,
            "net_flip_rate": round((counts["cw"] - counts["wc"]) / t * 100, 2) if t > 0 else 0,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default=os.path.join(PROJECT_ROOT, "mi_decoding", "results", "flip_rate_analysis.json"))
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()

    print("Loading MMStar categories...")
    cat_map = load_mmstar_categories()
    print(f"  Loaded categories for {len(cat_map)} items")

    results = {"methods": {}, "mmstar_category_breakdown": {}}

    for method_name, config in METHOD_CONFIGS.items():
        results["methods"][method_name] = {}

        for benchmark in BENCHMARKS:
            short_path = os.path.join(PROJECT_ROOT, config["short_dir"], benchmark, "all_predictions.jsonl")
            long_path = os.path.join(PROJECT_ROOT, config["long_dir"], benchmark, "all_predictions.jsonl")

            if not os.path.exists(short_path):
                print(f"SKIP {method_name}/{benchmark}: short predictions not found at {short_path}")
                continue
            if not os.path.exists(long_path):
                print(f"SKIP {method_name}/{benchmark}: long predictions not found at {long_path}")
                continue

            short_preds = load_predictions(short_path)
            long_preds = load_predictions(long_path)
            print(f"\n{method_name}/{benchmark}: short={len(short_preds)} long={len(long_preds)} items")

            flip_result, per_item = compute_flips(short_preds, long_preds, benchmark, args.max_items)
            results["methods"][method_name][benchmark] = flip_result

            print(f"  C->W: {flip_result['cw_rate']}%  W->C: {flip_result['wc_rate']}%  Net: {flip_result['net_flip_rate']}%")
            print(f"  Short acc: {flip_result['short_accuracy']}%  Long acc: {flip_result['long_accuracy']}%")

            if benchmark == "mmstar":
                cat_result = compute_category_flips(short_preds, long_preds, cat_map, args.max_items)
                results["mmstar_category_breakdown"][method_name] = cat_result
                print(f"  Categories: {list(cat_result.keys())}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    print("\n=== Summary Table ===")
    print(f"{'Method':<20} {'Benchmark':<16} {'C->W%':>8} {'W->C%':>8} {'Net%':>8} {'Short%':>8} {'Long%':>8}")
    print("-" * 88)
    for method_name in METHOD_CONFIGS:
        for benchmark in BENCHMARKS:
            r = results["methods"].get(method_name, {}).get(benchmark)
            if r:
                print(f"{method_name:<20} {benchmark:<16} {r['cw_rate']:>8} {r['wc_rate']:>8} {r['net_flip_rate']:>8} {r['short_accuracy']:>8} {r['long_accuracy']:>8}")


if __name__ == "__main__":
    main()
