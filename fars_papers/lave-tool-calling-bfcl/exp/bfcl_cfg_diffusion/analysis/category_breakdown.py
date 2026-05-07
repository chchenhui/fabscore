"""Per-category success rate breakdown and inference timing comparison across conditions A/B/C.

Reads RESULTS.json from each condition's EXPERIMENT_RESULTS folder plus raw JSONL result files
for per-instance wall_time. Outputs a combined JSON to scores/category_breakdown_results.json
and prints markdown tables.

Usage:
  python -m bfcl_cfg_diffusion.analysis.category_breakdown
"""

import json
from pathlib import Path
import numpy as np

EXP_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = EXP_ROOT / "EXPERIMENT_RESULTS"
RAW_RESULTS = EXP_ROOT / "bfcl_cfg_diffusion" / "results"
SCORES_DIR = EXP_ROOT / "bfcl_cfg_diffusion" / "scores"

CONDITION_MAP = {
    "unconstrained": {
        "label": "(A) Unconstrained",
        "results_json": RESULTS_DIR / "condition_a_unconstrained" / "RESULTS.json",
        "raw_dir": RAW_RESULTS / "unconstrained",
    },
    "best_of_2": {
        "label": "(B) Best-of-2",
        "results_json": RESULTS_DIR / "condition_b_best_of_2" / "RESULTS.json",
        "raw_dir": RAW_RESULTS / "best_of_2",
    },
    "lave_cfg_v3": {
        "label": "(C) LAVE CFG",
        "results_json": RESULTS_DIR / "condition_c_lave_cfg" / "RESULTS.json",
        "raw_dir": RAW_RESULTS / "lave_cfg_v3",
    },
}

CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple", "irrelevance",
]

CATEGORY_DISPLAY = {
    "simple_python": "Simple (Python)",
    "simple_java": "Simple (Java)",
    "simple_javascript": "Simple (JavaScript)",
    "multiple": "Multiple",
    "parallel": "Parallel",
    "parallel_multiple": "Parallel Multiple",
    "irrelevance": "Irrelevance",
}

SEEDS = [42, 123, 456]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_per_category(cond_key, data):
    if cond_key == "unconstrained":
        return data.get("per_category_mean", {})
    elif cond_key == "best_of_2":
        return data.get("per_category_results", {})
    else:
        return data.get("per_category", {})


def get_success_rate(cat_data):
    return cat_data.get("success_rate_mean", 0.0), cat_data.get("success_rate_std", 0.0)


def compute_simple_merged(cond_key, data):
    simple_cats = ["simple_python", "simple_java", "simple_javascript"]
    if cond_key == "unconstrained":
        per_seed = data.get("per_seed", {})
        seed_rates = []
        for seed_str, sd in per_seed.items():
            total_correct = 0
            total_n = 0
            for sc in simple_cats:
                cat_d = sd.get("per_category", {}).get(sc, {})
                total_correct += cat_d.get("correct", 0)
                total_n += cat_d.get("total", 50)
            seed_rates.append(total_correct / total_n if total_n > 0 else 0)
        return float(np.mean(seed_rates)), float(np.std(seed_rates))
    elif cond_key == "best_of_2":
        per_cat = data.get("per_category_results", {})
        means = [per_cat.get(sc, {}).get("success_rate_mean", 0) for sc in simple_cats]
        stds = [per_cat.get(sc, {}).get("success_rate_std", 0) for sc in simple_cats]
        merged_mean = float(np.mean(means))
        merged_std = float(np.sqrt(np.mean(np.array(stds)**2)))
        return merged_mean, merged_std
    else:
        per_cat = data.get("per_category", {})
        means = [per_cat.get(sc, {}).get("success_rate_mean", 0) for sc in simple_cats]
        stds = [per_cat.get(sc, {}).get("success_rate_std", 0) for sc in simple_cats]
        merged_mean = float(np.mean(means))
        merged_std = float(np.sqrt(np.mean(np.array(stds)**2)))
        return merged_mean, merged_std


def compute_timing(raw_dir):
    seed_means = []
    for seed in SEEDS:
        fpath = raw_dir / f"seed_{seed}.jsonl"
        if not fpath.exists():
            continue
        records = load_jsonl(fpath)
        times = [r["wall_time"] for r in records if "wall_time" in r]
        if times:
            seed_means.append(np.mean(times))
    return seed_means


def compute_per_category_timing(raw_dir):
    cat_times = {cat: [] for cat in CATEGORIES}
    for seed in SEEDS:
        fpath = raw_dir / f"seed_{seed}.jsonl"
        if not fpath.exists():
            continue
        records = load_jsonl(fpath)
        seed_cat_times = {cat: [] for cat in CATEGORIES}
        for r in records:
            cat = r.get("category", "")
            if cat in seed_cat_times and "wall_time" in r:
                seed_cat_times[cat].append(r["wall_time"])
        for cat in CATEGORIES:
            if seed_cat_times[cat]:
                cat_times[cat].append(np.mean(seed_cat_times[cat]))
    result = {}
    for cat in CATEGORIES:
        if cat_times[cat]:
            result[cat] = {
                "mean": float(np.mean(cat_times[cat])),
                "std": float(np.std(cat_times[cat])),
            }
    return result


def main():
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {"per_category_success": {}, "timing": {}, "per_category_timing": {}}

    for cond_key, cond_info in CONDITION_MAP.items():
        data = load_json(cond_info["results_json"])
        per_cat = extract_per_category(cond_key, data)

        cat_results = {}
        for cat in CATEGORIES:
            cd = per_cat.get(cat, {})
            mean, std = get_success_rate(cd)
            cat_results[cat] = {"mean": mean, "std": std}

        sm_mean, sm_std = compute_simple_merged(cond_key, data)
        cat_results["simple_merged"] = {"mean": sm_mean, "std": sm_std}

        if cond_key == "unconstrained":
            overall = data.get("overall_success_rate_mean", 0), data.get("overall_success_rate_std", 0)
        elif cond_key == "best_of_2":
            overall = data.get("overall_results", {}).get("success_rate_mean", 0), data.get("overall_results", {}).get("success_rate_std", 0)
        else:
            overall = data.get("overall", {}).get("success_rate_mean", 0), data.get("overall", {}).get("success_rate_std", 0)
        cat_results["overall"] = {"mean": overall[0], "std": overall[1]}

        all_results["per_category_success"][cond_key] = cat_results

        seed_means = compute_timing(cond_info["raw_dir"])
        if seed_means:
            avg_time = float(np.mean(seed_means))
            std_time = float(np.std(seed_means))
            total_time = avg_time * 350
        else:
            avg_time, std_time, total_time = 0, 0, 0
        all_results["timing"][cond_key] = {
            "avg_time_per_instance": avg_time,
            "std_time_per_instance": std_time,
            "total_time_350": total_time,
            "per_seed_means": seed_means,
        }

        cat_timing = compute_per_category_timing(cond_info["raw_dir"])
        all_results["per_category_timing"][cond_key] = cat_timing

    baseline_time = all_results["timing"]["unconstrained"]["avg_time_per_instance"]
    for cond_key in CONDITION_MAP:
        t = all_results["timing"][cond_key]["avg_time_per_instance"]
        all_results["timing"][cond_key]["relative_overhead"] = round(t / baseline_time, 2) if baseline_time > 0 else 0

    out_path = SCORES_DIR / "category_breakdown_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {out_path}")

    print("\n## Per-Category Success Rate (%)\n")
    header = "| Category | # Examples | (A) Unconstrained | (B) Best-of-2 | (C) LAVE CFG |"
    sep = "|----------|-----------|-------------------|---------------|--------------|"
    print(header)
    print(sep)

    for cat in CATEGORIES:
        n = 50
        display = CATEGORY_DISPLAY[cat]
        vals = []
        for cond_key in ["unconstrained", "best_of_2", "lave_cfg_v3"]:
            d = all_results["per_category_success"][cond_key][cat]
            vals.append(f"{d['mean']*100:.1f} +/- {d['std']*100:.1f}")
        print(f"| {display} | {n} | {vals[0]} | {vals[1]} | {vals[2]} |")

    n_simple = 150
    vals = []
    for cond_key in ["unconstrained", "best_of_2", "lave_cfg_v3"]:
        d = all_results["per_category_success"][cond_key]["simple_merged"]
        vals.append(f"{d['mean']*100:.1f} +/- {d['std']*100:.1f}")
    print(f"| **Simple (merged)** | {n_simple} | {vals[0]} | {vals[1]} | {vals[2]} |")

    vals = []
    for cond_key in ["unconstrained", "best_of_2", "lave_cfg_v3"]:
        d = all_results["per_category_success"][cond_key]["overall"]
        vals.append(f"{d['mean']*100:.2f} +/- {d['std']*100:.2f}")
    print(f"| **Overall** | **350** | {vals[0]} | {vals[1]} | {vals[2]} |")

    print("\n## Inference Timing Comparison\n")
    header2 = "| Condition | Avg Time/Instance (s) | Total Time for 350 (s) | Relative Overhead vs (A) |"
    sep2 = "|-----------|----------------------|------------------------|--------------------------|"
    print(header2)
    print(sep2)
    for cond_key in ["unconstrained", "best_of_2", "lave_cfg_v3"]:
        label = CONDITION_MAP[cond_key]["label"]
        t = all_results["timing"][cond_key]
        time_str = f"{t['avg_time_per_instance']:.2f} +/- {t['std_time_per_instance']:.2f}"
        total_str = f"{t['total_time_350']:.0f}"
        overhead_str = f"{t['relative_overhead']:.2f}x"
        print(f"| {label} | {time_str} | {total_str} | {overhead_str} |")


if __name__ == "__main__":
    main()
