# Compute ranking agreement metrics between proxy and target rankings.
# PDA (Pairwise Direction Accuracy), Spearman rho, Top-1 accuracy, with bootstrap CI.
# Usage: python compute_ranking_metrics.py --proxy <proxy_ranking.json> --target <target_ranking.json> --output <output.json>
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


def load_ranking(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    score_map = {}
    for entry in data["ranking"]:
        score_map[entry["dataset"]] = entry["composite_score"]
    return score_map


def compute_pda(proxy_scores: dict, target_scores: dict, datasets: list) -> float:
    pairs = list(combinations(datasets, 2))
    concordant = 0
    for d_i, d_j in pairs:
        proxy_diff = proxy_scores[d_i] - proxy_scores[d_j]
        target_diff = target_scores[d_i] - target_scores[d_j]
        if proxy_diff * target_diff > 0:
            concordant += 1
        elif proxy_diff == 0 or target_diff == 0:
            concordant += 0.5
    return concordant / len(pairs)


def bootstrap_pda_ci(proxy_scores: dict, target_scores: dict, datasets: list,
                     n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    rng = np.random.RandomState(42)
    pairs = list(combinations(datasets, 2))
    n_pairs = len(pairs)

    pair_concordance = []
    for d_i, d_j in pairs:
        proxy_diff = proxy_scores[d_i] - proxy_scores[d_j]
        target_diff = target_scores[d_i] - target_scores[d_j]
        if proxy_diff * target_diff > 0:
            pair_concordance.append(1.0)
        elif proxy_diff == 0 or target_diff == 0:
            pair_concordance.append(0.5)
        else:
            pair_concordance.append(0.0)
    pair_concordance = np.array(pair_concordance)

    bootstrap_pdas = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_pairs, size=n_pairs, replace=True)
        bootstrap_pdas.append(np.mean(pair_concordance[indices]))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(bootstrap_pdas, 100 * alpha))
    upper = float(np.percentile(bootstrap_pdas, 100 * (1 - alpha)))
    return lower, upper


def compute_spearman(proxy_scores: dict, target_scores: dict, datasets: list) -> tuple:
    proxy_vals = [proxy_scores[d] for d in datasets]
    target_vals = [target_scores[d] for d in datasets]
    rho, pvalue = stats.spearmanr(proxy_vals, target_vals)
    return float(rho), float(pvalue)


def compute_top1_accuracy(proxy_scores: dict, target_scores: dict) -> bool:
    proxy_top = max(proxy_scores, key=proxy_scores.get)
    target_top = max(target_scores, key=target_scores.get)
    return proxy_top == target_top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    proxy_scores = load_ranking(args.proxy)
    target_scores = load_ranking(args.target)

    datasets = sorted(set(proxy_scores.keys()) & set(target_scores.keys()))
    n_datasets = len(datasets)
    n_pairs = n_datasets * (n_datasets - 1) // 2

    print(f"Datasets: {n_datasets}, Pairs: {n_pairs}")

    pda = compute_pda(proxy_scores, target_scores, datasets)
    pda_lower, pda_upper = bootstrap_pda_ci(proxy_scores, target_scores, datasets, args.n_bootstrap)
    spearman_rho, spearman_pvalue = compute_spearman(proxy_scores, target_scores, datasets)
    top1_match = compute_top1_accuracy(proxy_scores, target_scores)

    proxy_top = max(proxy_scores, key=proxy_scores.get)
    target_top = max(target_scores, key=target_scores.get)

    proxy_ranking = sorted(proxy_scores.keys(), key=lambda d: proxy_scores[d], reverse=True)
    target_ranking = sorted(target_scores.keys(), key=lambda d: target_scores[d], reverse=True)

    results = {
        "pda": pda,
        "pda_95ci_lower": pda_lower,
        "pda_95ci_upper": pda_upper,
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
        "top1_match": top1_match,
        "proxy_top1": proxy_top,
        "target_top1": target_top,
        "n_datasets": n_datasets,
        "n_pairs": n_pairs,
        "n_bootstrap": args.n_bootstrap,
        "proxy_ranking": proxy_ranking,
        "target_ranking": target_ranking,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Ranking Agreement ===")
    print(f"PDA: {pda:.4f}  (95% CI: [{pda_lower:.4f}, {pda_upper:.4f}])")
    print(f"Spearman rho: {spearman_rho:.4f}  (p={spearman_pvalue:.4f})")
    print(f"Top-1 match: {top1_match}  (proxy={proxy_top}, target={target_top})")
    print(f"\nProxy ranking:  {proxy_ranking}")
    print(f"Target ranking: {target_ranking}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
