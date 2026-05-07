# Compute per-benchmark PDA/Spearman for all methods (GSM8K-only, MATH500-only, composite).
# Reads score CSVs and base_nll_scores.csv, outputs results/per_benchmark_pda.json
# and results/main_results_table.csv.
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_scores_csv(path: str) -> dict:
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            rows[ds] = {
                "gsm8k": float(row["gsm8k_mean"]) if "gsm8k_mean" in row else None,
                "math500": float(row["math500_mean"]) if "math500_mean" in row else None,
                "composite": float(row["composite_score"]) if "composite_score" in row else None,
            }
    return rows


def load_nll_scores(path: str) -> dict:
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row["dataset"]
            nll = float(row["mean_nll"])
            rows[ds] = {
                "gsm8k": None,
                "math500": None,
                "composite": -nll,
            }
    return rows


def compute_pda(scores_a: dict, scores_b: dict, datasets: list, key: str) -> float:
    pairs = list(combinations(datasets, 2))
    concordant = 0
    for d_i, d_j in pairs:
        diff_a = scores_a[d_i][key] - scores_a[d_j][key]
        diff_b = scores_b[d_i][key] - scores_b[d_j][key]
        if diff_a * diff_b > 0:
            concordant += 1
        elif diff_a == 0 or diff_b == 0:
            concordant += 0.5
    return concordant / len(pairs) if pairs else 0.5


def bootstrap_pda_ci(scores_a, scores_b, datasets, key, n_bootstrap=1000, ci=0.95):
    rng = np.random.RandomState(42)
    pairs = list(combinations(datasets, 2))
    n_pairs = len(pairs)
    conc = []
    for d_i, d_j in pairs:
        diff_a = scores_a[d_i][key] - scores_a[d_j][key]
        diff_b = scores_b[d_i][key] - scores_b[d_j][key]
        if diff_a * diff_b > 0:
            conc.append(1.0)
        elif diff_a == 0 or diff_b == 0:
            conc.append(0.5)
        else:
            conc.append(0.0)
    conc = np.array(conc)
    boot_pdas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        boot_pdas.append(np.mean(conc[idx]))
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_pdas, 100 * alpha)), float(np.percentile(boot_pdas, 100 * (1 - alpha)))


def compute_spearman(scores_a, scores_b, datasets, key):
    va = [scores_a[d][key] for d in datasets]
    vb = [scores_b[d][key] for d in datasets]
    rho, p = stats.spearmanr(va, vb)
    return float(rho), float(p)


def top1_match(scores_a, scores_b, datasets, key):
    top_a = max(datasets, key=lambda d: scores_a[d][key])
    top_b = max(datasets, key=lambda d: scores_b[d][key])
    return top_a == top_b, top_a, top_b


def compute_method_metrics(proxy_scores, target_scores, datasets, key):
    pda = compute_pda(proxy_scores, target_scores, datasets, key)
    lo, hi = bootstrap_pda_ci(proxy_scores, target_scores, datasets, key)
    rho, p = compute_spearman(proxy_scores, target_scores, datasets, key)
    match, ptop, ttop = top1_match(proxy_scores, target_scores, datasets, key)
    return {
        "pda": round(pda, 4),
        "pda_95ci_lower": round(lo, 4),
        "pda_95ci_upper": round(hi, 4),
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p, 6),
        "top1_match": match,
        "proxy_top1": ptop,
        "target_top1": ttop,
    }


def main():
    target = load_scores_csv(RESULTS_DIR / "target_scores.csv")
    std = load_scores_csv(RESULTS_DIR / "proxy_std_scores.csv")
    tiny = load_scores_csv(RESULTS_DIR / "proxy_tiny_v2_scores.csv")
    nll = load_nll_scores(RESULTS_DIR / "base_nll_scores.csv")

    datasets = sorted(set(target) & set(std) & set(tiny) & set(nll))
    print(f"Common datasets: {len(datasets)}")

    results = {}
    for method_name, proxy_scores in [("standard_lr", std), ("tiny_lr_v2", tiny), ("nll_baseline", nll)]:
        results[method_name] = {}
        for bench_key in ["composite", "gsm8k", "math500"]:
            if bench_key != "composite" and method_name == "nll_baseline":
                results[method_name][bench_key] = {"note": "NLL baseline has no per-benchmark scores"}
                continue
            if any(proxy_scores[d][bench_key] is None for d in datasets):
                results[method_name][bench_key] = {"note": "scores unavailable"}
                continue
            metrics = compute_method_metrics(proxy_scores, target, datasets, bench_key)
            results[method_name][bench_key] = metrics
            print(f"{method_name} / {bench_key}: PDA={metrics['pda']:.4f} [{metrics['pda_95ci_lower']:.4f}, {metrics['pda_95ci_upper']:.4f}] rho={metrics['spearman_rho']:.4f}")

    out_path = RESULTS_DIR / "per_benchmark_pda.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    build_main_table(results)


def build_main_table(per_bench):
    rows = []

    rows.append({
        "Method": "Random Baseline",
        "Proxy_Model": "-",
        "Target_Model": "Qwen2.5-7B",
        "Benchmark": "composite",
        "PDA": 0.5,
        "PDA_95CI": "[analytical]",
        "Spearman_rho": 0.0,
        "Spearman_p": "N/A",
        "Top1_Accuracy": "N/A",
    })

    method_display = {
        "nll_baseline": ("Training-Free NLL (C)", "Qwen2.5-1.5B (base)"),
        "standard_lr": ("Standard-LR Proxy SFT (A)", "Qwen2.5-1.5B"),
        "tiny_lr_v2": ("Tiny-LR Proxy SFT (B)", "Qwen2.5-1.5B"),
    }

    for method_key in ["nll_baseline", "standard_lr", "tiny_lr_v2"]:
        display_name, proxy_model = method_display[method_key]
        for bench in ["composite", "gsm8k", "math500"]:
            entry = per_bench[method_key].get(bench, {})
            if "note" in entry:
                rows.append({
                    "Method": display_name,
                    "Proxy_Model": proxy_model,
                    "Target_Model": "Qwen2.5-7B",
                    "Benchmark": bench,
                    "PDA": "N/A",
                    "PDA_95CI": "N/A",
                    "Spearman_rho": "N/A",
                    "Spearman_p": "N/A",
                    "Top1_Accuracy": "N/A",
                })
                continue
            ci_str = f"[{entry['pda_95ci_lower']:.4f}, {entry['pda_95ci_upper']:.4f}]"
            rows.append({
                "Method": display_name,
                "Proxy_Model": proxy_model,
                "Target_Model": "Qwen2.5-7B",
                "Benchmark": bench,
                "PDA": entry["pda"],
                "PDA_95CI": ci_str,
                "Spearman_rho": entry["spearman_rho"],
                "Spearman_p": entry["spearman_p"],
                "Top1_Accuracy": entry["top1_match"],
            })

    out_path = RESULTS_DIR / "main_results_table.csv"
    fieldnames = ["Method", "Proxy_Model", "Target_Model", "Benchmark", "PDA", "PDA_95CI", "Spearman_rho", "Spearman_p", "Top1_Accuracy"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMain results table saved to {out_path}")


if __name__ == "__main__":
    main()
