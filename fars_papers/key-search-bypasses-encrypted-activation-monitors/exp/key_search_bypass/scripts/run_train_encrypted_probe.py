# Train MLP probes on encrypted activations and evaluate K=1 detection rate.
# Uses same architecture/hyperparams as baseline: hidden_dim=32, lr=1e-4, wd=1.0, 5000 epochs.
# Trains 3 seeds, calibrates thresholds on 52k encrypted Alpaca, reports TPR@FPR.

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_DIR))

from key_search_bypass.monitor.train import train_probe
from key_search_bypass.monitor.probe import MLPProbe
from key_search_bypass.metrics.safety import evaluate_probe, aggregate_results

ACT_DIR = PROJ_DIR / "key_search_bypass" / "outputs" / "activations" / "encrypted_train"
LAYER_IDX = 27
MONITOR_DIR = PROJ_DIR / "key_search_bypass" / "outputs" / "monitors" / "encrypted"
RESULTS_DIR = PROJ_DIR / "key_search_bypass" / "results"
SEEDS = [42, 123, 456]


def main():
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    layer_dir = ACT_DIR / f"layer_{LAYER_IDX}"
    train_acts = np.load(layer_dir / "train.npy")
    train_labels = np.load(ACT_DIR / "train_labels.npy")
    test_acts = np.load(layer_dir / "test.npy")
    test_labels = np.load(ACT_DIR / "test_labels.npy")
    alpaca_full_acts = np.load(layer_dir / "alpaca_full.npy")

    print(f"Train: {train_acts.shape}, Test: {test_acts.shape}, Alpaca: {alpaca_full_acts.shape}")
    print(f"Train labels: {train_labels.shape} (pos={train_labels.sum()}, neg={len(train_labels)-train_labels.sum()})")
    print(f"Test labels: {test_labels.shape} (pos={test_labels.sum()}, neg={len(test_labels)-test_labels.sum()})")

    probes = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Training probe seed={seed}, layer={LAYER_IDX}")
        print(f"{'='*60}")
        probe, auroc, best_epoch = train_probe(
            train_acts, train_labels,
            test_acts, test_labels,
            input_dim=3584,
            hidden_dim=32,
            lr=1e-4,
            weight_decay=1.0,
            epochs=5000,
            seed=seed,
            run_name=f"encrypted_L{LAYER_IDX}_s{seed}",
            use_wandb=True,
        )
        ckpt_path = MONITOR_DIR / f"probe_L{LAYER_IDX}_s{seed}.pt"
        torch.save(probe.state_dict(), ckpt_path)
        print(f"  Saved to {ckpt_path}")
        probes.append((probe, seed, auroc, best_epoch))

    print(f"\n{'='*60}")
    print("Evaluation: K=1 encrypted monitor performance")
    print(f"{'='*60}")

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
        print(f"  TPR@FPR=1e-3: {res['fpr_0.001']['tpr']:.4f} (threshold={res['fpr_0.001']['threshold']:.6f})")
        print(f"  TPR@FPR=1e-4: {res['fpr_0.0001']['tpr']:.4f} (threshold={res['fpr_0.0001']['threshold']:.6f})")

    agg = aggregate_results(all_results)
    print(f"\n{'='*60}")
    print("Aggregated K=1 Results (mean +/- std over 3 seeds)")
    print(f"{'='*60}")
    print(f"  AUROC: {agg['auroc_mean']:.4f} +/- {agg['auroc_std']:.4f}")
    print(f"  TPR@FPR=1e-3: {agg['fpr_0.001_tpr_mean']:.4f} +/- {agg['fpr_0.001_tpr_std']:.4f}")
    print(f"  TPR@FPR=1e-4: {agg['fpr_0.0001_tpr_mean']:.4f} +/- {agg['fpr_0.0001_tpr_std']:.4f}")

    output = {
        "layer": LAYER_IDX,
        "encryptor": "seed_123_opt2/last_checkpoint.pt",
        "per_seed_results": all_results,
        "aggregated": agg,
        "config": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "hidden_dim": 32,
            "lr": 1e-4,
            "weight_decay": 1.0,
            "epochs": 5000,
            "seeds": SEEDS,
        },
    }
    results_path = RESULTS_DIR / "encrypted_monitor_k1.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
