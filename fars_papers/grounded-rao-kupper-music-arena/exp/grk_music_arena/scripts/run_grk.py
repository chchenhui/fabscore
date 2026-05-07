# Run GRK model: fit on train, evaluate on global/instrumental/vocal test splits.
# Includes pairwise bootstrap comparisons vs BT and AB-MNL baselines.

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grk_music_arena.data.load_music_arena import load_and_split
from grk_music_arena.models.grk_model import GRKModel
from grk_music_arena.models.bt_model import BradleyTerryModel
from grk_music_arena.models.abmnl_model import ABMNLModel
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


def evaluate_split(model, test_df, label):
    if len(test_df) == 0:
        print(f"  [{label}] Empty split, skipping.")
        return None

    probs = model.predict_probs_batch(test_df)
    labels = test_df['preference'].values

    nll = four_way_nll(probs, labels)
    cls_nll = per_class_nll(probs, labels)
    brier = brier_score_bothbad(probs, labels)
    ece = ece_bothbad(probs, labels)

    print(f"  [{label}] 4-way NLL: {nll:.4f}")
    print(f"  [{label}] Per-class NLL: { {k: f'{v:.4f}' for k,v in cls_nll.items()} }")
    print(f"  [{label}] BOTH_BAD Brier: {brier:.6f}")
    print(f"  [{label}] BOTH_BAD ECE: {ece:.6f}")

    print(f"  [{label}] Running bootstrap (1000 samples)...")
    boot_nll = bootstrap_metric(test_df, model, four_way_nll, n_bootstrap=1000, seed=42)
    boot_brier = bootstrap_metric(test_df, model, brier_score_bothbad, n_bootstrap=1000, seed=42)

    print(f"  [{label}] 4-way NLL 95% CI: [{boot_nll['ci_lower']:.4f}, {boot_nll['ci_upper']:.4f}]")
    print(f"  [{label}] BOTH_BAD Brier 95% CI: [{boot_brier['ci_lower']:.6f}, {boot_brier['ci_upper']:.6f}]")

    return {
        'split': label,
        'n_battles': len(test_df),
        'four_way_nll': nll,
        'per_class_nll': cls_nll,
        'brier_score_bothbad': brier,
        'ece_bothbad': ece,
        'bootstrap_four_way_nll': boot_nll,
        'bootstrap_brier_bothbad': boot_brier,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Music Arena data...")
    splits, stats = load_and_split(train_frac=0.7)

    print("\n=== Dataset Statistics ===")
    for key, s in stats.items():
        if isinstance(s, dict) and 'total_battles' in s:
            print(f"  {s['label']}: {s['total_battles']} battles, "
                  f"{s['unique_systems']} systems, "
                  f"BOTH_BAD={s['bothbad_count']} ({s['bothbad_rate']:.2%})")

    grk_cv_path = os.path.join(RESULTS_DIR, 'grk_cv_results.json')
    use_gamma = False
    best_l2_beta = 0.0
    best_l2_gamma = 0.1
    if os.path.exists(grk_cv_path):
        with open(grk_cv_path) as f:
            grk_cv = json.load(f)
        best_l2_beta = grk_cv.get('best_l2', 0.0)
        best_l2_gamma = grk_cv.get('best_l2_gamma', 0.1)
        use_gamma = grk_cv.get('best_use_gamma', False)

    print(f"\n=== Fitting GRK on Full Training Data (use_gamma={use_gamma}, L2_beta={best_l2_beta}, L2_gamma={best_l2_gamma}) ===")
    grk = GRKModel(use_gamma=use_gamma)
    grk.fit(splits['train'], l2_beta=best_l2_beta, l2_gamma=best_l2_gamma)

    scores = grk.get_scores()
    sorted_beta = sorted(scores['beta'].items(), key=lambda x: x[1], reverse=True)
    print("\nGRK Leaderboard (beta):")
    for name, score in sorted_beta:
        print(f"  {name}: {score:.4f}")
    print(f"\nLambda (tie parameter): {scores['lambda']:.4f}")

    accept = grk.get_acceptability()
    print("\nPer-system acceptability P(BOTH_BAD | k vs avg):")
    for name, p in sorted(accept.items(), key=lambda x: x[1]):
        print(f"  {name}: {p:.4f}")

    print("\n=== Fitting BT and AB-MNL for pairwise comparisons ===")
    bt = BradleyTerryModel()
    bt.fit(splits['train'])
    print("  BT fitted.")

    cv_path = os.path.join(RESULTS_DIR, 'abmnl_cv_results.json')
    if os.path.exists(cv_path):
        with open(cv_path) as f:
            cv_output = json.load(f)
        best_l2 = cv_output['best_l2']
    else:
        best_l2 = 0.1
    abmnl = ABMNLModel()
    abmnl.fit(splits['train'], l2_rho=best_l2)
    print(f"  AB-MNL fitted (L2={best_l2}).")

    print("\n=== Evaluating on Global Test Set ===")
    global_results = evaluate_split(grk, splits['test'], 'global')

    print("\n  Pairwise bootstrap: GRK vs BT (NLL)...")
    pw_bt_nll = pairwise_bootstrap(splits['test'], grk, bt, four_way_nll)
    print(f"    NLL diff (GRK - BT): {pw_bt_nll['base_diff']:.4f} "
          f"CI=[{pw_bt_nll['ci_lower']:.4f}, {pw_bt_nll['ci_upper']:.4f}] "
          f"sig={pw_bt_nll['significant']}")

    print("  Pairwise bootstrap: GRK vs AB-MNL (NLL)...")
    pw_abmnl_nll = pairwise_bootstrap(splits['test'], grk, abmnl, four_way_nll)
    print(f"    NLL diff (GRK - ABMNL): {pw_abmnl_nll['base_diff']:.4f} "
          f"CI=[{pw_abmnl_nll['ci_lower']:.4f}, {pw_abmnl_nll['ci_upper']:.4f}] "
          f"sig={pw_abmnl_nll['significant']}")

    print("  Pairwise bootstrap: GRK vs BT (Brier)...")
    pw_bt_brier = pairwise_bootstrap(splits['test'], grk, bt, brier_score_bothbad)
    print(f"    Brier diff (GRK - BT): {pw_bt_brier['base_diff']:.6f} "
          f"CI=[{pw_bt_brier['ci_lower']:.6f}, {pw_bt_brier['ci_upper']:.6f}] "
          f"sig={pw_bt_brier['significant']}")

    print("  Pairwise bootstrap: GRK vs AB-MNL (Brier)...")
    pw_abmnl_brier = pairwise_bootstrap(splits['test'], grk, abmnl, brier_score_bothbad)
    print(f"    Brier diff (GRK - ABMNL): {pw_abmnl_brier['base_diff']:.6f} "
          f"CI=[{pw_abmnl_brier['ci_lower']:.6f}, {pw_abmnl_brier['ci_upper']:.6f}] "
          f"sig={pw_abmnl_brier['significant']}")

    global_results['grk_scores'] = scores
    global_results['acceptability'] = accept
    global_results['dataset_stats'] = stats
    global_results['pairwise_bootstrap'] = {
        'grk_vs_bt_nll': pw_bt_nll,
        'grk_vs_abmnl_nll': pw_abmnl_nll,
        'grk_vs_bt_brier': pw_bt_brier,
        'grk_vs_abmnl_brier': pw_abmnl_brier,
    }

    print("\n=== Evaluating on Instrumental Test Subset ===")
    instrumental_results = evaluate_split(grk, splits['test_instrumental'], 'instrumental')

    print("\n=== Evaluating on Vocal Test Subset ===")
    vocal_results = evaluate_split(grk, splits['test_vocal'], 'vocal')

    global_path = os.path.join(RESULTS_DIR, 'grk_global.json')
    with open(global_path, 'w') as f:
        json.dump(global_results, f, indent=2, default=str)
    print(f"\nSaved global results to {global_path}")

    if instrumental_results:
        inst_path = os.path.join(RESULTS_DIR, 'grk_instrumental.json')
        with open(inst_path, 'w') as f:
            json.dump(instrumental_results, f, indent=2, default=str)
        print(f"Saved instrumental results to {inst_path}")

    if vocal_results:
        vocal_path = os.path.join(RESULTS_DIR, 'grk_vocal.json')
        with open(vocal_path, 'w') as f:
            json.dump(vocal_results, f, indent=2, default=str)
        print(f"Saved vocal results to {vocal_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
