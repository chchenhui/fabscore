# 5-fold CV to tune L2 regularization for GRK model.
# Compares base GRK (l2_beta only) vs extended GRK (l2_beta + l2_gamma).
# Folds are contiguous time-ordered chunks of the training split.
# Saves CV results to grk_music_arena/results/grk_cv_results.json.

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grk_music_arena.data.load_music_arena import load_and_split
from grk_music_arena.models.grk_model import GRKModel
from grk_music_arena.evaluation.metrics import four_way_nll

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
N_FOLDS = 5


def time_ordered_folds(df, n_folds=5):
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n
        folds.append(df.iloc[start:end].reset_index(drop=True))
    return folds


def run_cv(train_df):
    folds = time_ordered_folds(train_df, N_FOLDS)

    configs = [
        {'use_gamma': False, 'l2_beta': 0.0, 'l2_gamma': 0.0, 'label': 'base_l2b=0'},
        {'use_gamma': True, 'l2_beta': 0.0, 'l2_gamma': 0.01, 'label': 'gamma_l2g=0.01'},
        {'use_gamma': True, 'l2_beta': 0.0, 'l2_gamma': 0.02, 'label': 'gamma_l2g=0.02'},
        {'use_gamma': True, 'l2_beta': 0.0, 'l2_gamma': 0.05, 'label': 'gamma_l2g=0.05'},
        {'use_gamma': True, 'l2_beta': 0.0, 'l2_gamma': 0.1, 'label': 'gamma_l2g=0.1'},
        {'use_gamma': True, 'l2_beta': 0.0, 'l2_gamma': 0.2, 'label': 'gamma_l2g=0.2'},
        {'use_gamma': True, 'l2_beta': 0.0, 'l2_gamma': 0.5, 'label': 'gamma_l2g=0.5'},
        {'use_gamma': True, 'l2_beta': 0.0, 'l2_gamma': 1.0, 'label': 'gamma_l2g=1.0'},
    ]

    results = {}
    for cfg in configs:
        fold_nlls = []
        for hold_out_idx in range(N_FOLDS):
            train_folds = [folds[j] for j in range(N_FOLDS) if j != hold_out_idx]
            cv_train = pd.concat(train_folds, ignore_index=True)
            cv_val = folds[hold_out_idx]

            model = GRKModel(use_gamma=cfg['use_gamma'])
            model.fit(cv_train, l2_beta=cfg['l2_beta'], l2_gamma=cfg['l2_gamma'])

            probs = model.predict_probs_batch(cv_val)
            labels = cv_val['preference'].values
            nll = four_way_nll(probs, labels)
            fold_nlls.append(nll)
            print(f"  {cfg['label']}, fold={hold_out_idx}, NLL={nll:.4f}")

        mean_nll = float(np.mean(fold_nlls))
        std_nll = float(np.std(fold_nlls))
        results[cfg['label']] = {
            'config': cfg,
            'fold_nlls': fold_nlls,
            'mean_nll': mean_nll,
            'std_nll': std_nll,
        }
        print(f"  {cfg['label']} => mean NLL={mean_nll:.4f} +/- {std_nll:.4f}")

    best_key = min(results, key=lambda k: results[k]['mean_nll'])
    best_info = results[best_key]

    cv_output = {
        'candidates': results,
        'best_config': best_info['config'],
        'best_label': best_key,
        'best_mean_nll': best_info['mean_nll'],
        'best_l2': best_info['config']['l2_beta'],
        'best_l2_gamma': best_info['config'].get('l2_gamma', 0.0),
        'best_use_gamma': best_info['config']['use_gamma'],
    }
    return cv_output


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Music Arena data...")
    splits, stats = load_and_split(train_frac=0.7)
    train_df = splits['train']

    print(f"\n=== 5-Fold CV for GRK hyperparameters (n_train={len(train_df)}) ===")
    cv_output = run_cv(train_df)

    print(f"\nBest config: {cv_output['best_label']} (mean NLL={cv_output['best_mean_nll']:.4f})")

    out_path = os.path.join(RESULTS_DIR, 'grk_cv_results.json')
    with open(out_path, 'w') as f:
        json.dump(cv_output, f, indent=2)
    print(f"Saved CV results to {out_path}")

    return cv_output


if __name__ == '__main__':
    main()
