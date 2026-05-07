# AB-MNL regularization sensitivity: sweep L2 penalty on rho to trace
# test-set NLL and corr(rho, -beta) as a function of regularization strength.
# Tests whether strong regularization pushes AB-MNL toward GRK-like behavior.

import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grk_music_arena.data.load_music_arena import load_and_split
from grk_music_arena.models.abmnl_model import ABMNLModel
from grk_music_arena.evaluation.metrics import four_way_nll, brier_score_bothbad

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

L2_VALUES = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


def fit_and_evaluate(train_df, test_df, l2_rho):
    model = ABMNLModel()
    model.fit(train_df, l2_rho=l2_rho)

    probs = model.predict_probs_batch(test_df)
    labels = test_df['preference'].values
    nll = four_way_nll(probs, labels)
    brier = brier_score_bothbad(probs, labels)

    scores = model.get_scores()
    systems = sorted(scores['beta'].keys())
    beta_arr = np.array([scores['beta'][s] for s in systems])
    rho_arr = np.array([scores['rho'][s] for s in systems])

    neg_beta = -beta_arr
    pearson_corr, pearson_p = pearsonr(rho_arr, neg_beta)
    spearman_corr, spearman_p = spearmanr(rho_arr, neg_beta)

    return {
        'l2_rho': l2_rho,
        'four_way_nll': float(nll),
        'brier_score_bothbad': float(brier),
        'pearson_corr_rho_neg_beta': float(pearson_corr),
        'pearson_p': float(pearson_p),
        'spearman_corr_rho_neg_beta': float(spearman_corr),
        'spearman_p': float(spearman_p),
        'rho': {s: float(rho_arr[i]) for i, s in enumerate(systems)},
        'beta': {s: float(beta_arr[i]) for i, s in enumerate(systems)},
        'tau': float(scores['tau']),
        'kappa': float(scores['kappa']),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Music Arena data...")
    splits, stats = load_and_split(train_frac=0.7)

    print(f"  Train: {len(splits['train'])} battles")
    print(f"  Test: {len(splits['test'])} battles")

    print("\n=== AB-MNL Regularization Sensitivity ===")
    print(f"L2 values: {L2_VALUES}\n")

    results_list = []
    for l2 in L2_VALUES:
        print(f"  L2={l2}...", end=' ')
        res = fit_and_evaluate(splits['train'], splits['test'], l2)
        results_list.append(res)
        print(f"NLL={res['four_way_nll']:.4f}, Brier={res['brier_score_bothbad']:.6f}, "
              f"Pearson(rho,-beta)={res['pearson_corr_rho_neg_beta']:.4f}, "
              f"Spearman(rho,-beta)={res['spearman_corr_rho_neg_beta']:.4f}")

    grk_path = os.path.join(RESULTS_DIR, 'grk_global.json')
    grk_nll = None
    if os.path.exists(grk_path):
        with open(grk_path) as f:
            grk_data = json.load(f)
        grk_nll = grk_data['four_way_nll']
        print(f"\nFull GRK 4-way NLL for reference: {grk_nll:.4f}")

    output = {
        'l2_values': L2_VALUES,
        'results': results_list,
        'grk_reference_nll': grk_nll,
        'dataset_stats': stats,
    }

    out_path = os.path.join(RESULTS_DIR, 'abmnl_reg_sensitivity.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
