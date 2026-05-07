# Script to train MLP probes: layer sweep then multi-seed training on best layer.
# Evaluates TPR@FPR=1e-3 and 1e-4 on HarmBench test, calibrated on 52k Alpaca.

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from key_search_bypass.monitor.train import layer_sweep, train_multi_seed
from key_search_bypass.metrics.safety import evaluate_probe, aggregate_results

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
ACT_DIR = os.path.join(PROJ, "key_search_bypass", "outputs", "activations")
MONITOR_DIR = os.path.join(PROJ, "key_search_bypass", "outputs", "monitors", "unencrypted")
RESULTS_DIR = os.path.join(PROJ, "key_search_bypass", "results")
LAYERS = list(range(21, 28))
SEEDS = [42, 123, 456]


def main():
    print("=" * 60)
    print("Phase 1: Layer Sweep")
    print("=" * 60)
    best_layer, sweep_results = layer_sweep(
        activation_dir=ACT_DIR,
        layers=LAYERS,
        input_dim=3584,
        hidden_dim=32,
        epochs=5000,
        seed=42,
        use_wandb=True,
    )
    print(f"\nLayer sweep results:")
    for layer_idx, res in sorted(sweep_results.items()):
        print(f"  Layer {layer_idx}: AUROC={res['auroc']:.4f} (best epoch {res['best_epoch']})")
    print(f"Best layer: {best_layer}")

    print("\n" + "=" * 60)
    print(f"Phase 2: Multi-Seed Training on Layer {best_layer}")
    print("=" * 60)
    probes = train_multi_seed(
        activation_dir=ACT_DIR,
        layer_idx=best_layer,
        seeds=SEEDS,
        output_dir=MONITOR_DIR,
        input_dim=3584,
        hidden_dim=32,
        epochs=5000,
        use_wandb=True,
    )

    print("\n" + "=" * 60)
    print("Phase 3: Evaluation")
    print("=" * 60)
    alpaca_full_acts = np.load(os.path.join(ACT_DIR, f"layer_{best_layer}", "alpaca_full.npy"))
    test_acts = np.load(os.path.join(ACT_DIR, f"layer_{best_layer}", "test.npy"))
    test_labels = np.load(os.path.join(ACT_DIR, "test_labels.npy"))

    all_results = []
    for probe, seed, auroc, best_epoch in probes:
        print(f"\nEvaluating seed {seed}...")
        res = evaluate_probe(
            probe=probe,
            alpaca_full_acts=alpaca_full_acts,
            test_acts=test_acts,
            test_labels=test_labels,
            fpr_targets=(1e-3, 1e-4),
            device="cuda",
        )
        res["seed"] = seed
        res["best_epoch"] = best_epoch
        all_results.append(res)
        print(f"  AUROC: {res['auroc']:.4f}")
        print(f"  TPR@FPR=1e-3: {res['fpr_0.001']['tpr']:.4f} (actual FPR={res['fpr_0.001']['actual_fpr']:.6f})")
        print(f"  TPR@FPR=1e-4: {res['fpr_0.0001']['tpr']:.4f} (actual FPR={res['fpr_0.0001']['actual_fpr']:.6f})")

    agg = aggregate_results(all_results)
    print("\n" + "=" * 60)
    print("Aggregated Results (mean +/- std over 3 seeds)")
    print("=" * 60)
    print(f"  AUROC: {agg['auroc_mean']:.4f} +/- {agg['auroc_std']:.4f}")
    print(f"  TPR@FPR=1e-3: {agg['fpr_0.001_tpr_mean']:.4f} +/- {agg['fpr_0.001_tpr_std']:.4f}")
    print(f"  TPR@FPR=1e-4: {agg['fpr_0.0001_tpr_mean']:.4f} +/- {agg['fpr_0.0001_tpr_std']:.4f}")

    if agg['fpr_0.001_tpr_mean'] < 0.80:
        print("\nWARNING: TPR@FPR=1e-3 is below 0.80! Investigate layer choice or probe capacity.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "baseline_unencrypted_monitor.json")
    output = {
        "best_layer": best_layer,
        "layer_sweep": {str(k): v for k, v in sweep_results.items()},
        "per_seed_results": all_results,
        "aggregated": agg,
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "hidden_dim": 32,
            "lr": 1e-4,
            "weight_decay": 1.0,
            "epochs": 5000,
            "seeds": SEEDS,
            "layers_swept": LAYERS,
        },
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
