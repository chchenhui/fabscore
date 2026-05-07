# Collect per-prompt score distributions across 64 keys for harmful and harmless prompts.
# For each prompt, samples 64 keys and records the full monitor score vector.
# Uses the best diverse encryptor (seed 123, last_checkpoint) and best probe (seed 123).

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer
from key_search_bypass.encryptor.model import KeyConditionedEncryptor
from key_search_bypass.monitor.probe import MLPProbe
from key_search_bypass.attack.key_search import key_search_scores
from key_search_bypass.data.harmbench import load_harmbench_tokenized
from key_search_bypass.data.alpaca import load_alpaca_subset

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 27
KEY_DIM = 128
K = 64
KEY_BATCH = 8

ENCRYPTOR_PATH = os.path.join(BASE, "outputs", "encryptor", "seed_123_diverse", "last_checkpoint.pt")
PROBE_PATH = os.path.join(BASE, "outputs", "monitors", "encrypted_diverse", "probe_L27_s123.pt")
THRESHOLD_FILE = os.path.join(BASE, "results", "encrypted_diverse_monitor_k1.json")
OUTPUT_DIR = os.path.join(BASE, "outputs", "score_distributions")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print("Loading encryptor...")
    encryptor = KeyConditionedEncryptor(hidden_dim=3584, key_dim=KEY_DIM)
    ckpt = torch.load(ENCRYPTOR_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("encryptor_state_dict", ckpt.get("model_state_dict", ckpt))
    encryptor.load_state_dict(state_dict)
    encryptor = encryptor.to(device).to(torch.bfloat16)
    encryptor.eval()

    print("Loading probe...")
    probe = MLPProbe(input_dim=3584, hidden_dim=32)
    probe_ckpt = torch.load(PROBE_PATH, map_location="cpu", weights_only=True)
    probe.load_state_dict(probe_ckpt)
    probe = probe.to(device)
    probe.eval()

    with open(THRESHOLD_FILE) as f:
        monitor_results = json.load(f)
    seed_123_results = [s for s in monitor_results["per_seed"] if s["seed"] == 123][0]
    threshold_1e3 = seed_123_results["fpr_0.001"]["threshold"]
    threshold_1e4 = seed_123_results["fpr_0.0001"]["threshold"]
    print(f"Thresholds: FPR=1e-3 -> {threshold_1e3:.4f}, FPR=1e-4 -> {threshold_1e4:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nLoading HarmBench test set...")
    harmful_data = load_harmbench_tokenized(tokenizer=tokenizer, csv_name="harmbench_behaviors_text_test.csv")
    n_harmful = harmful_data["input_ids"].shape[0]
    print(f"  {n_harmful} harmful prompts")

    print("Loading Alpaca subset (200)...")
    harmless_data = load_alpaca_subset(n=200, seed=42, tokenizer=tokenizer)
    n_harmless = harmless_data["input_ids"].shape[0]
    print(f"  {n_harmless} harmless prompts")

    harmful_ids = torch.tensor(harmful_data["input_ids"], dtype=torch.long)
    harmful_mask = torch.tensor(harmful_data["attention_mask"], dtype=torch.long)
    harmless_ids = torch.tensor(harmless_data["input_ids"], dtype=torch.long)
    harmless_mask = torch.tensor(harmless_data["attention_mask"], dtype=torch.long)

    print(f"\n--- Collecting {K} scores per prompt for {n_harmful} harmful prompts ---")
    harmful_scores = np.zeros((n_harmful, K), dtype=np.float32)
    for i in range(n_harmful):
        ids = harmful_ids[i:i+1].to(device)
        mask = harmful_mask[i:i+1].to(device)
        scores = key_search_scores(
            ids, mask, model, encryptor, probe,
            K=K, layer_idx=LAYER_IDX, key_dim=KEY_DIM, key_batch=KEY_BATCH,
        )
        harmful_scores[i] = scores
        if (i + 1) % 20 == 0:
            print(f"  Harmful: {i+1}/{n_harmful}")

    print(f"\n--- Collecting {K} scores per prompt for {n_harmless} harmless prompts ---")
    harmless_scores = np.zeros((n_harmless, K), dtype=np.float32)
    for i in range(n_harmless):
        ids = harmless_ids[i:i+1].to(device)
        mask = harmless_mask[i:i+1].to(device)
        scores = key_search_scores(
            ids, mask, model, encryptor, probe,
            K=K, layer_idx=LAYER_IDX, key_dim=KEY_DIM, key_batch=KEY_BATCH,
        )
        harmless_scores[i] = scores
        if (i + 1) % 20 == 0:
            print(f"  Harmless: {i+1}/{n_harmless}")

    np.savez(
        os.path.join(OUTPUT_DIR, "harmful_scores.npz"),
        scores=harmful_scores,
        texts=np.array(harmful_data["texts"], dtype=object),
    )
    np.savez(
        os.path.join(OUTPUT_DIR, "harmless_scores.npz"),
        scores=harmless_scores,
        texts=np.array(harmless_data["texts"], dtype=object),
    )
    threshold_info = {
        "fpr_1e3": float(threshold_1e3),
        "fpr_1e4": float(threshold_1e4),
        "probe_seed": 123,
        "encryptor_seed": 123,
        "K": K,
    }
    with open(os.path.join(OUTPUT_DIR, "threshold.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  harmful_scores.npz: shape {harmful_scores.shape}")
    print(f"  harmless_scores.npz: shape {harmless_scores.shape}")
    print(f"  threshold.json: {threshold_info}")

    print("\n--- Quick stats ---")
    print(f"Harmful scores: mean={harmful_scores.mean():.4f}, std={harmful_scores.std():.4f}")
    print(f"Harmless scores: mean={harmless_scores.mean():.4f}, std={harmless_scores.std():.4f}")
    per_prompt_std_harmful = harmful_scores.std(axis=1)
    per_prompt_std_harmless = harmless_scores.std(axis=1)
    print(f"Per-prompt std (harmful): mean={per_prompt_std_harmful.mean():.4f}, max={per_prompt_std_harmful.max():.4f}")
    print(f"Per-prompt std (harmless): mean={per_prompt_std_harmless.mean():.4f}, max={per_prompt_std_harmless.max():.4f}")

    mins_32 = np.min(harmful_scores[:, :32], axis=1)
    bypass_rate_32 = (mins_32 < threshold_1e3).mean()
    print(f"Bypass rate at K=32 (FPR=1e-3): {bypass_rate_32:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
