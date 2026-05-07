# Aggregate lm-eval results for proxy checkpoints across 12 datasets x 3 seeds.
# Computes per-dataset mean/std for GSM8K and MATH-500, composite score, and ranking.
# Usage: python aggregate_proxy_scores.py --regime proxy_std
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp")

DATASETS = [
    "AM-Thinking-v1-Distilled-math", "DeepMath-309K", "Maths-College",
    "OpenR1-Math", "QwQ-LongCoT-130K-math", "R1-Distill-SFT-math",
    "hkust-nlp__dart-math-hard", "mathplus", "numinamath-cot",
    "numinamath1_5", "openmathinstruct-2", "Magpie-Reasoning-V2-250K-CoT-QwQ-math",
]
SEEDS = [42, 123, 456]

LR_MAP = {"proxy_std": 5e-5, "proxy_tiny": 5e-6, "proxy_mid": 2e-5, "proxy_tiny_v2": 1e-5}


def find_results_json(result_dir: Path) -> dict | None:
    candidates = list(result_dir.rglob("results_*.json")) + list(result_dir.rglob("results.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    with open(candidates[0]) as fh:
        return json.load(fh)


def extract_scores(results: dict) -> tuple[float | None, float | None]:
    gsm8k_score = None
    math500_score = None

    r = results.get("results", {})

    for key in ["gsm8k", "gsm8k_cot_zeroshot"]:
        if key in r:
            for metric_key in ["exact_match,flexible-extract", "exact_match,strict-match", "acc,none"]:
                if metric_key in r[key]:
                    gsm8k_score = r[key][metric_key]
                    break
            break

    for key in ["minerva_math500", "minerva_math_500"]:
        if key in r:
            for metric_key in ["math_verify,none", "exact_match,none", "acc,none"]:
                if metric_key in r[key]:
                    math500_score = r[key][metric_key]
                    break
            break

    return gsm8k_score, math500_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, required=True, choices=["proxy_std", "proxy_tiny", "proxy_mid", "proxy_tiny_v2"])
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "tlr_proxy_sft" / "results" / args.regime
    output_dir = PROJECT_ROOT / "tlr_proxy_sft" / "results"
    all_scores = defaultdict(lambda: {"gsm8k": [], "math500": []})

    for dataset in DATASETS:
        for seed in SEEDS:
            result_dir = results_dir / dataset / f"seed_{seed}"
            if not result_dir.exists():
                print(f"MISSING: {dataset}/seed_{seed}")
                continue

            results = find_results_json(result_dir)
            if results is None:
                print(f"NO results.json: {dataset}/seed_{seed}")
                continue

            gsm8k, math500 = extract_scores(results)
            if gsm8k is not None:
                all_scores[dataset]["gsm8k"].append(gsm8k)
            if math500 is not None:
                all_scores[dataset]["math500"].append(math500)

            print(f"{dataset}/seed_{seed}: GSM8K={gsm8k}, MATH500={math500}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        s = all_scores[dataset]
        gsm8k_scores = s["gsm8k"]
        math500_scores = s["math500"]

        gsm8k_mean = float(np.mean(gsm8k_scores)) if gsm8k_scores else None
        gsm8k_std = float(np.std(gsm8k_scores)) if gsm8k_scores else None
        math500_mean = float(np.mean(math500_scores)) if math500_scores else None
        math500_std = float(np.std(math500_scores)) if math500_scores else None

        composite = None
        if gsm8k_mean is not None and math500_mean is not None:
            composite = (gsm8k_mean + math500_mean) / 2

        rows.append({
            "dataset": dataset,
            "gsm8k_mean": gsm8k_mean,
            "gsm8k_std": gsm8k_std,
            "math500_mean": math500_mean,
            "math500_std": math500_std,
            "composite_score": composite,
            "n_seeds": len(gsm8k_scores),
        })

    rows.sort(key=lambda x: x["composite_score"] if x["composite_score"] is not None else -1, reverse=True)

    csv_path = output_dir / f"{args.regime}_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")

    lr = LR_MAP.get(args.regime, None)
    ranking = {
        "ranking": [
            {
                "rank": i + 1,
                "dataset": r["dataset"],
                "composite_score": r["composite_score"],
                "gsm8k_mean": r["gsm8k_mean"],
                "gsm8k_std": r["gsm8k_std"],
                "math500_mean": r["math500_mean"],
                "math500_std": r["math500_std"],
                "n_seeds": r["n_seeds"],
            }
            for i, r in enumerate(rows)
        ],
        "metadata": {
            "base_model": "Qwen/Qwen2.5-1.5B",
            "finetuning_type": "lora",
            "lora_rank": 16,
            "max_steps": 500,
            "learning_rate": lr,
            "seeds": SEEDS,
            "eval_tasks": ["gsm8k", "minerva_math500"],
        },
    }
    ranking_path = output_dir / f"{args.regime}_ranking.json"
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"Saved {ranking_path}")

    print(f"\n=== {args.regime} Dataset Ranking ===")
    for r in rows:
        c = r["composite_score"]
        g = r["gsm8k_mean"]
        m = r["math500_mean"]
        if c:
            print(f"  {r['dataset']:45s}  composite={c:.4f}  gsm8k={g:.4f}  math500={m:.4f}")
        else:
            print(f"  {r['dataset']:45s}  INCOMPLETE")


if __name__ == "__main__":
    main()
