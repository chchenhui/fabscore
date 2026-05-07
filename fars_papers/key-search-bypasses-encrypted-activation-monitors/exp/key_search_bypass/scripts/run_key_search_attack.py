# Run key-search attack evaluation on all HarmBench test harmful prompts.
# For each of 3 monitor seeds and K in {1,2,4,8,16,32,64}, computes TPR@FPR.
# Saves results to results/key_search_attack.json.

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from key_search_bypass.encryptor.model import KeyConditionedEncryptor
from key_search_bypass.monitor.probe import MLPProbe
from key_search_bypass.metrics.safety import calibrate_threshold, get_scores
from key_search_bypass.attack.key_search import run_key_search_eval
from key_search_bypass.data.harmbench import load_harmbench_tokenized

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 27
KEY_DIM = 128
ENCRYPTOR_CKPT = PROJ_DIR / "key_search_bypass" / "outputs" / "encryptor" / "seed_123_opt2" / "last_checkpoint.pt"
MONITOR_DIR = PROJ_DIR / "key_search_bypass" / "outputs" / "monitors" / "encrypted"
ACT_DIR = PROJ_DIR / "key_search_bypass" / "outputs" / "activations" / "encrypted_train"
RESULTS_DIR = PROJ_DIR / "key_search_bypass" / "results"
SEEDS = [42, 123, 456]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_dim = model.config.hidden_size

    print(f"Loading encryptor from {ENCRYPTOR_CKPT}...")
    ckpt = torch.load(ENCRYPTOR_CKPT, map_location="cpu", weights_only=True)
    encryptor = KeyConditionedEncryptor(hidden_dim=hidden_dim, key_dim=KEY_DIM)
    encryptor.load_state_dict(ckpt["encryptor_state_dict"])
    encryptor = encryptor.to(dtype=torch.bfloat16, device="cuda")
    encryptor.eval()

    print("Loading HarmBench test set (harmful only)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    harm_test = load_harmbench_tokenized(tokenizer=tokenizer, csv_name="harmbench_behaviors_text_test.csv")
    harmful_ids = torch.tensor(harm_test["input_ids"], dtype=torch.long)
    harmful_mask = torch.tensor(harm_test["attention_mask"], dtype=torch.long)
    n_harmful = harmful_ids.shape[0]
    print(f"  {n_harmful} harmful test prompts")

    alpaca_full_acts = np.load(ACT_DIR / f"layer_{LAYER_IDX}" / "alpaca_full.npy")
    print(f"  Alpaca full activations: {alpaca_full_acts.shape}")

    all_seed_results = {}
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        probe = MLPProbe(input_dim=3584, hidden_dim=32)
        probe_path = MONITOR_DIR / f"probe_L{LAYER_IDX}_s{seed}.pt"
        probe.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=True))
        probe = probe.to("cuda")
        probe.eval()

        alpaca_scores = get_scores(probe, alpaca_full_acts, device="cuda")
        tau_1e3 = calibrate_threshold(alpaca_scores, 1e-3)
        tau_1e4 = calibrate_threshold(alpaca_scores, 1e-4)
        print(f"  Thresholds: FPR=1e-3 -> tau={tau_1e3:.6f}, FPR=1e-4 -> tau={tau_1e4:.6f}")

        thresholds = {
            "fpr_0.001": tau_1e3,
            "fpr_0.0001": tau_1e4,
        }

        results = run_key_search_eval(
            harmful_ids, harmful_mask,
            model, encryptor, probe,
            thresholds=thresholds,
            K_values=K_VALUES,
            layer_idx=LAYER_IDX,
            key_dim=KEY_DIM,
            key_batch=8,
        )

        all_seed_results[seed] = {
            "thresholds": {k: float(v) for k, v in thresholds.items()},
            "per_K": {str(k): v for k, v in results.items()},
        }

        print(f"\n  Results for seed {seed}:")
        print(f"  {'K':>4} | TPR@FPR=1e-3 | TPR@FPR=1e-4")
        print(f"  {'-'*40}")
        for k_val in K_VALUES:
            r = results[k_val]
            print(f"  {k_val:>4} | {r['fpr_0.001']['tpr']:.4f}       | {r['fpr_0.0001']['tpr']:.4f}")

        probe = probe.cpu()
        del probe
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Aggregating results across seeds")
    print(f"{'='*60}")

    aggregated = {}
    for k_val in K_VALUES:
        k_agg = {}
        for fpr_key in ["fpr_0.001", "fpr_0.0001"]:
            tprs = [all_seed_results[s]["per_K"][str(k_val)][fpr_key]["tpr"] for s in SEEDS]
            k_agg[fpr_key] = {
                "tpr_mean": float(np.mean(tprs)),
                "tpr_std": float(np.std(tprs)),
                "tpr_per_seed": {str(s): t for s, t in zip(SEEDS, tprs)},
            }
        aggregated[str(k_val)] = k_agg

    print(f"\n  {'K':>4} | TPR@FPR=1e-3 (mean+/-std) | TPR@FPR=1e-4 (mean+/-std)")
    print(f"  {'-'*65}")
    for k_val in K_VALUES:
        a = aggregated[str(k_val)]
        print(f"  {k_val:>4} | {a['fpr_0.001']['tpr_mean']:.4f} +/- {a['fpr_0.001']['tpr_std']:.4f}       "
              f"| {a['fpr_0.0001']['tpr_mean']:.4f} +/- {a['fpr_0.0001']['tpr_std']:.4f}")

    output = {
        "K_values": K_VALUES,
        "seeds": SEEDS,
        "encryptor": "seed_123_opt2/last_checkpoint.pt",
        "layer": LAYER_IDX,
        "per_seed_results": {str(s): v for s, v in all_seed_results.items()},
        "aggregated": aggregated,
    }

    results_path = RESULTS_DIR / "key_search_attack.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
