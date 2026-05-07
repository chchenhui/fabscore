# Run BT baseline: load data, fit model, evaluate on global/instrumental/vocal splits.
# Outputs JSON results to grk_music_arena/results/.

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from grk_music_arena.data.load_music_arena import load_and_split
from grk_music_arena.models.bt_model import BradleyTerryModel
from grk_music_arena.evaluation.metrics import (
    four_way_nll, per_class_nll, brier_score_bothbad, ece_bothbad
)
from grk_music_arena.evaluation.bootstrap import bootstrap_metric

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


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
            print(f"    Outcomes: {s['outcome_counts']}")
        elif isinstance(s, dict) and 'underpowered' in s:
            print(f"  Underpower gate: {'FLAGGED' if s['underpowered'] else 'OK'}")
            if s['underpowered']:
                for r in s['reasons']:
                    print(f"    - {r}")

    print("\n=== Fitting BT Model on Training Data ===")
    model = BradleyTerryModel()
    model.fit(splits['train'])

    scores = model.get_scores()
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("\nBT Leaderboard (top 10):")
    for name, score in sorted_scores[:10]:
        print(f"  {name}: {score:.4f}")

    print("\n=== Evaluating on Global Test Set ===")
    global_results = evaluate_split(model, splits['test'], 'global')
    global_results['bt_scores'] = scores
    global_results['dataset_stats'] = stats

    print("\n=== Evaluating on Instrumental Test Subset ===")
    instrumental_results = evaluate_split(model, splits['test_instrumental'], 'instrumental')

    print("\n=== Evaluating on Vocal Test Subset ===")
    vocal_results = evaluate_split(model, splits['test_vocal'], 'vocal')

    global_path = os.path.join(RESULTS_DIR, 'bt_baseline_global.json')
    with open(global_path, 'w') as f:
        json.dump(global_results, f, indent=2, default=str)
    print(f"\nSaved global results to {global_path}")

    if instrumental_results:
        inst_path = os.path.join(RESULTS_DIR, 'bt_baseline_instrumental.json')
        with open(inst_path, 'w') as f:
            json.dump(instrumental_results, f, indent=2, default=str)
        print(f"Saved instrumental results to {inst_path}")

    if vocal_results:
        vocal_path = os.path.join(RESULTS_DIR, 'bt_baseline_vocal.json')
        with open(vocal_path, 'w') as f:
            json.dump(vocal_results, f, indent=2, default=str)
        print(f"Saved vocal results to {vocal_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
