# 5-fold CV to tune L2 regularization on rho for AB-MNL model.
# Folds are contiguous time-ordered chunks of the training split.
# Saves CV results to grk_music_arena/results/abmnl_cv_results.json.

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grk_music_arena.data.load_music_arena import load_and_split
from grk_music_arena.models.abmnl_model import ABMNLModel
from grk_music_arena.evaluation.metrics import four_way_nll

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
L2_CANDIDATES = [0, 0.001, 0.01, 0.1, 1.0, 10.0]
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
    results = {}

    for l2 in L2_CANDIDATES:
        fold_nlls = []
        for hold_out_idx in range(N_FOLDS):
            train_folds = [folds[j] for j in range(N_FOLDS) if j != hold_out_idx]
            import pandas as pd
            cv_train = pd.concat(train_folds, ignore_index=True)
            cv_val = folds[hold_out_idx]

            model = ABMNLModel()
            model.fit(cv_train, l2_rho=l2)

            probs = model.predict_probs_batch(cv_val)
            labels = cv_val['preference'].values
            nll = four_way_nll(probs, labels)
            fold_nlls.append(nll)
            print(f"  l2={l2:.4f}, fold={hold_out_idx}, NLL={nll:.4f}")

        mean_nll = float(np.mean(fold_nlls))
        std_nll = float(np.std(fold_nlls))
        results[str(l2)] = {
            'l2_rho': l2,
            'fold_nlls': fold_nlls,
            'mean_nll': mean_nll,
            'std_nll': std_nll,
        }
        print(f"  l2={l2:.4f} => mean NLL={mean_nll:.4f} +/- {std_nll:.4f}")

    best_l2 = min(results, key=lambda k: results[k]['mean_nll'])
    best_info = results[best_l2]

    cv_output = {
        'candidates': {k: v for k, v in results.items()},
        'best_l2': best_info['l2_rho'],
        'best_mean_nll': best_info['mean_nll'],
    }
    return cv_output


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Music Arena data...")
    splits, stats = load_and_split(train_frac=0.7)
    train_df = splits['train']

    print(f"\n=== 5-Fold CV for AB-MNL L2 on rho (n_train={len(train_df)}) ===")
    cv_output = run_cv(train_df)

    print(f"\nBest L2: {cv_output['best_l2']} (mean NLL={cv_output['best_mean_nll']:.4f})")

    out_path = os.path.join(RESULTS_DIR, 'abmnl_cv_results.json')
    with open(out_path, 'w') as f:
        json.dump(cv_output, f, indent=2)
    print(f"Saved CV results to {out_path}")

    return cv_output


if __name__ == '__main__':
    main()
