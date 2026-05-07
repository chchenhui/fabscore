# Validates optimized v2 encryptor checkpoints (last only, step 5000) on held-out data.
# Only evaluates last checkpoints since best are too early (step 500, no privacy).

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from key_search_bypass.encryptor.model import KeyConditionedEncryptor
from key_search_bypass.metrics.utility import compute_mean_kl
from key_search_bypass.metrics.privacy import compute_knn_asr, compute_key_diversity

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEEDS = [42, 123, 456]
KEY_DIM = 128
MAX_LENGTH = 128
VAL_SIZE = 1000


def load_validation_data(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    texts = [ex["instruction"] for ex in ds]
    rng = np.random.RandomState(seed=999)
    all_indices = rng.permutation(len(texts))
    val_indices = all_indices[10000:10000 + VAL_SIZE]
    val_texts = [texts[i] for i in val_indices]
    val_enc = tokenizer(val_texts, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    return val_enc


def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_dim = model.config.hidden_size

    print("Loading validation data...")
    val_enc = load_validation_data(tokenizer)
    val_ids = val_enc["input_ids"]
    val_mask = val_enc["attention_mask"]
    print(f"Validation set: {val_ids.shape[0]} samples")

    results = {}
    for seed in SEEDS:
        ckpt_dir = PROJ_DIR / "key_search_bypass" / "outputs" / "encryptor" / f"seed_{seed}_opt2"
        ckpt_path = ckpt_dir / "last_checkpoint.pt"
        if not ckpt_path.exists():
            print(f"WARNING: No checkpoint for seed {seed}")
            continue

        print(f"\n=== Seed {seed} (last checkpoint) ===")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        print(f"  Step: {ckpt['step']}, Train eval KL: {ckpt['eval_kl']:.6f}")

        encryptor = KeyConditionedEncryptor(hidden_dim=hidden_dim, key_dim=KEY_DIM)
        encryptor.load_state_dict(ckpt["encryptor_state_dict"])
        encryptor = encryptor.to(dtype=torch.bfloat16, device="cuda")
        encryptor.eval()

        print("  Computing KL divergence...")
        kl = compute_mean_kl(model, encryptor, val_ids, val_mask, key_dim=KEY_DIM, batch_size=4)
        print(f"  KL divergence: {kl:.6f}")

        print("  Computing KNN ASR@10...")
        asr = compute_knn_asr(model, encryptor, val_ids, val_mask, key_dim=KEY_DIM, k=10, batch_size=4)
        print(f"  KNN ASR@10: {asr:.4f}")

        print("  Computing key diversity...")
        mean_dist, std_dist = compute_key_diversity(
            model, encryptor, val_ids, val_mask,
            key_dim=KEY_DIM, n_prompts=200, n_keys=32, batch_size=4
        )
        print(f"  Key diversity: mean L2={mean_dist:.4f}, std={std_dist:.4f}")

        results[f"seed_{seed}"] = {
            "step": ckpt["step"],
            "train_eval_kl": ckpt["eval_kl"],
            "val_kl_divergence": kl,
            "knn_asr_at_10": asr,
            "key_diversity_mean_l2": mean_dist,
            "key_diversity_std_l2": std_dist,
            "meets_utility_bound": kl <= 0.02,
            "meets_privacy_bound": asr <= 0.20,
        }

    kl_values = [r["val_kl_divergence"] for r in results.values()]
    asr_values = [r["knn_asr_at_10"] for r in results.values()]
    div_values = [r["key_diversity_mean_l2"] for r in results.values()]

    results["summary"] = {
        "mean_kl": float(np.mean(kl_values)),
        "std_kl": float(np.std(kl_values)),
        "mean_asr": float(np.mean(asr_values)),
        "std_asr": float(np.std(asr_values)),
        "mean_key_diversity": float(np.mean(div_values)),
        "std_key_diversity": float(np.std(div_values)),
        "all_meet_utility_bound": all(r["meets_utility_bound"] for r in results.values()),
        "all_meet_privacy_bound": all(r["meets_privacy_bound"] for r in results.values()),
    }

    output_path = PROJ_DIR / "key_search_bypass" / "results" / "encryptor_opt2_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"\nSummary:")
    print(f"  KL divergence: {results['summary']['mean_kl']:.6f} +/- {results['summary']['std_kl']:.6f}")
    print(f"  KNN ASR@10: {results['summary']['mean_asr']:.4f} +/- {results['summary']['std_asr']:.4f}")
    print(f"  Key diversity: {results['summary']['mean_key_diversity']:.4f} +/- {results['summary']['std_key_diversity']:.4f}")
    print(f"  Utility bound met: {results['summary']['all_meet_utility_bound']}")
    print(f"  Privacy bound met: {results['summary']['all_meet_privacy_bound']}")


if __name__ == "__main__":
    main()
