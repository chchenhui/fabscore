# Run GRK-no-grounding ablation: fit on train, evaluate on global test.
# Compares against full GRK (gamma extension) via pairwise bootstrap.

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grk_music_arena.data.load_music_arena import load_and_split
from grk_music_arena.models.grk_no_ground_model import GRKNoGroundModel
from grk_music_arena.models.grk_model import GRKModel
from grk_music_arena.evaluation.metrics import (
    four_way_nll, per_class_nll, brier_score_bothbad, ece_bothbad
)
from grk_music_arena.evaluation.bootstrap import bootstrap_metric

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def pairwise_bootstrap(test_df, model_a, model_b, metric_fn,
                        n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(test_df)
    probs_a = model_a.predict_probs_batch(test_df)
    probs_b = model_b.predict_probs_batch(test_df)
    labels = test_df['preference'].values

    base_diff = metric_fn(probs_a, labels) - metric_fn(probs_b, labels)

    diffs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        val_a = metric_fn(probs_a[idx], labels[idx])
        val_b = metric_fn(probs_b[idx], labels[idx])
        diffs[b] = val_a - val_b

    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    excludes_zero = (ci_lower > 0) or (ci_upper < 0)

    return {
        'base_diff': float(base_diff),
        'mean_diff': float(diffs.mean()),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'excludes_zero': excludes_zero,
        'significant': excludes_zero,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Music Arena data...")
    splits, stats = load_and_split(train_frac=0.7)

    print(f"  Train: {len(splits['train'])} battles")
    print(f"  Test: {len(splits['test'])} battles")

    print("\n=== Fitting GRK-No-Grounding on Training Data ===")
    model = GRKNoGroundModel()
    model.fit(splits['train'])

    scores = model.get_scores()
    sorted_beta = sorted(scores['beta'].items(), key=lambda x: x[1], reverse=True)
    print("\nGRK-No-Grounding Leaderboard (beta):")
    for name, score in sorted_beta:
        print(f"  {name}: {score:.4f}")
    print(f"\nLambda: {scores['lambda']:.4f}")
    print(f"c: {scores['c']:.4f}")
    print(f"P(BOTH_BAD) constant: {scores['p_bothbad_constant']:.4f}")

    print("\n=== Evaluating on Global Test Set ===")
    probs = model.predict_probs_batch(splits['test'])
    labels = splits['test']['preference'].values

    nll = four_way_nll(probs, labels)
    cls_nll = per_class_nll(probs, labels)
    brier = brier_score_bothbad(probs, labels)
    ece = ece_bothbad(probs, labels)

    print(f"  4-way NLL: {nll:.4f}")
    print(f"  Per-class NLL: { {k: f'{v:.4f}' for k,v in cls_nll.items()} }")
    print(f"  BOTH_BAD Brier: {brier:.6f}")
    print(f"  BOTH_BAD ECE: {ece:.6f}")

    print("\n  Running bootstrap (1000 samples)...")
    boot_nll = bootstrap_metric(splits['test'], model, four_way_nll, n_bootstrap=1000, seed=42)
    boot_brier = bootstrap_metric(splits['test'], model, brier_score_bothbad, n_bootstrap=1000, seed=42)

    print(f"  4-way NLL 95% CI: [{boot_nll['ci_lower']:.4f}, {boot_nll['ci_upper']:.4f}]")
    print(f"  BOTH_BAD Brier 95% CI: [{boot_brier['ci_lower']:.6f}, {boot_brier['ci_upper']:.6f}]")

    grk_cv_path = os.path.join(RESULTS_DIR, 'grk_cv_results.json')
    use_gamma = True
    best_l2_beta = 0.0
    best_l2_gamma = 0.1
    if os.path.exists(grk_cv_path):
        with open(grk_cv_path) as f:
            grk_cv = json.load(f)
        best_l2_beta = grk_cv.get('best_l2', 0.0)
        best_l2_gamma = grk_cv.get('best_l2_gamma', 0.1)
        use_gamma = grk_cv.get('best_use_gamma', True)

    print(f"\n=== Fitting Full GRK (use_gamma={use_gamma}, l2_beta={best_l2_beta}, l2_gamma={best_l2_gamma}) for comparison ===")
    grk_full = GRKModel(use_gamma=use_gamma)
    grk_full.fit(splits['train'], l2_beta=best_l2_beta, l2_gamma=best_l2_gamma)

    grk_probs = grk_full.predict_probs_batch(splits['test'])
    grk_nll = four_way_nll(grk_probs, labels)
    print(f"  Full GRK 4-way NLL: {grk_nll:.4f}")

    print("\n  Pairwise bootstrap: GRK-No-Grounding vs Full GRK (NLL)...")
    pw_nll = pairwise_bootstrap(splits['test'], model, grk_full, four_way_nll)
    print(f"    NLL diff (NoGround - GRK): {pw_nll['base_diff']:.4f} "
          f"CI=[{pw_nll['ci_lower']:.4f}, {pw_nll['ci_upper']:.4f}] "
          f"sig={pw_nll['significant']}")

    print("  Pairwise bootstrap: GRK-No-Grounding vs Full GRK (Brier)...")
    pw_brier = pairwise_bootstrap(splits['test'], model, grk_full, brier_score_bothbad)
    print(f"    Brier diff (NoGround - GRK): {pw_brier['base_diff']:.6f} "
          f"CI=[{pw_brier['ci_lower']:.6f}, {pw_brier['ci_upper']:.6f}] "
          f"sig={pw_brier['significant']}")

    results = {
        'split': 'global',
        'n_battles': len(splits['test']),
        'four_way_nll': nll,
        'per_class_nll': cls_nll,
        'brier_score_bothbad': brier,
        'ece_bothbad': ece,
        'bootstrap_four_way_nll': boot_nll,
        'bootstrap_brier_bothbad': boot_brier,
        'grk_no_ground_scores': scores,
        'full_grk_config': {
            'use_gamma': use_gamma,
            'l2_beta': best_l2_beta,
            'l2_gamma': best_l2_gamma,
        },
        'full_grk_four_way_nll': grk_nll,
        'pairwise_bootstrap': {
            'noground_vs_grk_nll': pw_nll,
            'noground_vs_grk_brier': pw_brier,
        },
        'dataset_stats': stats,
    }

    out_path = os.path.join(RESULTS_DIR, 'grk_no_ground_global.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
