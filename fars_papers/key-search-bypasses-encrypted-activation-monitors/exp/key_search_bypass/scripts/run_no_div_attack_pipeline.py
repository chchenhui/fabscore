# Adapted attack pipeline for the no-diversity encryptor (lambda2=0) ablation.
# Same 5-phase structure as run_full_attack_pipeline.py but with:
#   - encryptor checkpoints from outputs/encryptor_no_div/
#   - K_VALUES limited to [1,2,4,8,16,32,64]
#   - No gradient attack (skip_gradient_attack always True)
#   - Output files prefixed with "no_div"
import sys
import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from key_search_bypass.encryptor.model import KeyConditionedEncryptor
from key_search_bypass.monitor.probe import MLPProbe
from key_search_bypass.monitor.train import train_probe
from key_search_bypass.metrics.safety import evaluate_probe, aggregate_results, calibrate_threshold, get_scores
from key_search_bypass.metrics.privacy import compute_knn_asr, compute_key_diversity
from key_search_bypass.metrics.utility import compute_mean_kl
from key_search_bypass.attack.key_search import run_key_search_eval
from key_search_bypass.data import build_balanced_dataset
from key_search_bypass.data.alpaca import load_alpaca_full, load_alpaca_subset
from key_search_bypass.data.harmbench import load_harmbench_tokenized

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 27
KEY_DIM = 128
SEEDS = [42, 123, 456]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
ENCRYPTOR_DIR = PROJ_DIR / "key_search_bypass" / "outputs" / "encryptor_no_div"


def select_best_encryptor(model, tokenizer, hidden_dim):
    print("=" * 60)
    print("Phase 1: Validate no-diversity encryptor checkpoints")
    print("=" * 60)

    val_data = load_alpaca_subset(n=500, seed=9999, tokenizer=tokenizer)
    val_ids = torch.tensor(val_data["input_ids"], dtype=torch.long)
    val_mask = torch.tensor(val_data["attention_mask"], dtype=torch.long)

    results = {}
    for seed in SEEDS:
        ckpt_dir = ENCRYPTOR_DIR / f"seed_{seed}"
        for ckpt_name in ["best_checkpoint.pt", "last_checkpoint.pt"]:
            ckpt_path = ckpt_dir / ckpt_name
            if not ckpt_path.exists():
                continue

            print(f"\nValidating {ckpt_name} for seed {seed}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            encryptor = KeyConditionedEncryptor(hidden_dim=hidden_dim, key_dim=KEY_DIM)
            encryptor.load_state_dict(ckpt["encryptor_state_dict"])
            encryptor = encryptor.to(dtype=torch.bfloat16, device="cuda")
            encryptor.eval()

            kl = compute_mean_kl(model, encryptor, val_ids, val_mask, KEY_DIM, batch_size=4)
            asr = compute_knn_asr(model, encryptor, val_ids, val_mask, KEY_DIM, k=10, batch_size=4)
            div_mean, div_std = compute_key_diversity(model, encryptor, val_ids, val_mask, KEY_DIM, n_prompts=100, n_keys=32)

            key = f"seed_{seed}_{ckpt_name.replace('.pt','')}"
            results[key] = {
                "seed": seed,
                "ckpt": ckpt_name,
                "kl": kl,
                "asr": asr,
                "div_mean": div_mean,
                "div_std": div_std,
                "path": str(ckpt_path),
            }
            print(f"  KL={kl:.4f}, ASR@10={asr:.4f}, Diversity L2={div_mean:.4f}+/-{div_std:.4f}")

            encryptor = encryptor.cpu()
            del encryptor
            torch.cuda.empty_cache()

    best_key = min(results.keys(), key=lambda k: results[k]["kl"])
    print(f"\nSelected: {best_key} (KL={results[best_key]['kl']:.4f}, Div={results[best_key]['div_mean']:.4f})")
    return results[best_key], results


def extract_encrypted_activations(input_ids, attention_mask, model, encryptor, layer_idx, batch_size=8, device="cuda"):
    captured = []
    _batch_mask = [None]

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        mask = _batch_mask[0]
        seq_lengths = mask.sum(dim=1) - 1
        bs = hidden.shape[0]
        last_tok = hidden[torch.arange(bs, device=hidden.device), seq_lengths.to(hidden.device)]
        captured.append(last_tok.detach().float().cpu())

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    n = input_ids.shape[0]
    ids_t = torch.tensor(input_ids, dtype=torch.long) if not isinstance(input_ids, torch.Tensor) else input_ids.long()
    mask_t = torch.tensor(attention_mask, dtype=torch.long) if not isinstance(attention_mask, torch.Tensor) else attention_mask.long()

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids = ids_t[start:end].to(device)
            mask = mask_t[start:end].to(device)
            _batch_mask[0] = mask

            clean_embeds = model.model.embed_tokens(ids)
            k = KeyConditionedEncryptor.sample_key(ids.shape[0], KEY_DIM, device=device, dtype=clean_embeds.dtype)
            enc_embeds = encryptor(clean_embeds, k)

            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 1)
            model.model(
                inputs_embeds=enc_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
            )
            if (start // batch_size) % 100 == 0:
                print(f"  Batch {start // batch_size + 1}/{(n + batch_size - 1) // batch_size}")

    handle.remove()
    return torch.cat(captured, dim=0).numpy()


def main():
    os.environ["WANDB_MODE"] = "offline"

    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    hidden_dim = model.config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    best_enc_info, all_enc_results = select_best_encryptor(model, tokenizer, hidden_dim)

    results_dir = PROJ_DIR / "key_search_bypass" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "encryptor_no_div_validation.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in all_enc_results.items()}, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Phase 2: Extract encrypted activations with no-diversity encryptor")
    print("=" * 60)

    enc_ckpt = torch.load(best_enc_info["path"], map_location="cpu", weights_only=True)
    encryptor = KeyConditionedEncryptor(hidden_dim=hidden_dim, key_dim=KEY_DIM)
    encryptor.load_state_dict(enc_ckpt["encryptor_state_dict"])
    encryptor = encryptor.to(dtype=torch.bfloat16, device="cuda")
    encryptor.eval()

    train_data, test_data = build_balanced_dataset(tokenizer=tokenizer)
    alpaca_full = load_alpaca_full(tokenizer=tokenizer)

    act_dir = PROJ_DIR / "key_search_bypass" / "outputs" / "activations" / "encrypted_no_div"
    layer_dir = act_dir / f"layer_{LAYER_IDX}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting train set...")
    train_acts = extract_encrypted_activations(
        train_data["input_ids"], train_data["attention_mask"],
        model, encryptor, LAYER_IDX, batch_size=8,
    )
    np.save(layer_dir / "train.npy", train_acts)
    print(f"  Saved: {train_acts.shape}")

    print("Extracting test set...")
    test_acts = extract_encrypted_activations(
        test_data["input_ids"], test_data["attention_mask"],
        model, encryptor, LAYER_IDX, batch_size=8,
    )
    np.save(layer_dir / "test.npy", test_acts)
    print(f"  Saved: {test_acts.shape}")

    print("Extracting Alpaca full...")
    alpaca_acts = extract_encrypted_activations(
        alpaca_full["input_ids"], alpaca_full["attention_mask"],
        model, encryptor, LAYER_IDX, batch_size=8,
    )
    np.save(layer_dir / "alpaca_full.npy", alpaca_acts)
    print(f"  Saved: {alpaca_acts.shape}")

    np.save(act_dir / "train_labels.npy", train_data["labels"])
    np.save(act_dir / "test_labels.npy", test_data["labels"])

    print("\n" + "=" * 60)
    print("Phase 3: Train encrypted monitor probes (no-div)")
    print("=" * 60)

    monitor_dir = PROJ_DIR / "key_search_bypass" / "outputs" / "monitors" / "encrypted_no_div"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    probes = []
    for seed in SEEDS:
        print(f"\nTraining probe seed={seed}...")
        probe, auroc, best_epoch = train_probe(
            train_acts, train_data["labels"],
            test_acts, test_data["labels"],
            input_dim=3584, hidden_dim=32,
            lr=1e-4, weight_decay=1.0, epochs=5000,
            seed=seed,
            run_name=f"enc_no_div_L{LAYER_IDX}_s{seed}",
            use_wandb=True,
        )
        ckpt_path = monitor_dir / f"probe_L{LAYER_IDX}_s{seed}.pt"
        torch.save(probe.state_dict(), ckpt_path)
        probes.append((probe, seed, auroc, best_epoch))

    print("\nEvaluating K=1 monitor performance...")
    k1_results = []
    for probe, seed, auroc, best_epoch in probes:
        res = evaluate_probe(
            probe=probe, alpaca_full_acts=alpaca_acts,
            test_acts=test_acts, test_labels=test_data["labels"],
            fpr_targets=(1e-3, 1e-4), device="cuda",
        )
        res["seed"] = seed
        res["best_epoch"] = best_epoch
        k1_results.append(res)
        print(f"  Seed {seed}: AUROC={res['auroc']:.4f}, TPR@1e-3={res['fpr_0.001']['tpr']:.4f}, TPR@1e-4={res['fpr_0.0001']['tpr']:.4f}")

    k1_agg = aggregate_results(k1_results)
    with open(results_dir / "encrypted_no_div_monitor_k1.json", "w") as f:
        json.dump({"per_seed": k1_results, "aggregated": k1_agg}, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 4: Key-search attack (random only, no gradient)")
    print("=" * 60)

    harm_test = load_harmbench_tokenized(tokenizer=tokenizer, csv_name="harmbench_behaviors_text_test.csv")
    harmful_ids = torch.tensor(harm_test["input_ids"], dtype=torch.long)
    harmful_mask = torch.tensor(harm_test["attention_mask"], dtype=torch.long)
    print(f"  {harmful_ids.shape[0]} harmful test prompts")

    all_seed_random_results = {}
    for probe, seed, auroc, best_epoch in probes:
        print(f"\n--- Seed {seed} ---")
        probe = probe.to("cuda")
        probe.eval()

        alpaca_scores = get_scores(probe, alpaca_acts, device="cuda")
        tau_1e3 = calibrate_threshold(alpaca_scores, 1e-3)
        tau_1e4 = calibrate_threshold(alpaca_scores, 1e-4)
        print(f"  Thresholds: FPR=1e-3 -> tau={tau_1e3:.6f}, FPR=1e-4 -> tau={tau_1e4:.6f}")
        thresholds = {"fpr_0.001": tau_1e3, "fpr_0.0001": tau_1e4}

        print(f"  Running random key search (K up to {max(K_VALUES)})...")
        random_results = run_key_search_eval(
            harmful_ids, harmful_mask, model, encryptor, probe,
            thresholds=thresholds, K_values=K_VALUES,
            layer_idx=LAYER_IDX, key_dim=KEY_DIM, key_batch=8,
        )
        all_seed_random_results[seed] = {
            "thresholds": {k: float(v) for k, v in thresholds.items()},
            "per_K": {str(k): v for k, v in random_results.items()},
        }
        print(f"  Random results:")
        for k_val in K_VALUES:
            r = random_results[k_val]
            print(f"    K={k_val:>4}: TPR@1e-3={r['fpr_0.001']['tpr']:.4f}, TPR@1e-4={r['fpr_0.0001']['tpr']:.4f}")

        probe = probe.cpu()
        del probe
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Phase 5: Aggregate and save results")
    print("=" * 60)

    random_agg = {}
    for k_val in K_VALUES:
        k_agg = {}
        for fpr_key in ["fpr_0.001", "fpr_0.0001"]:
            tprs = [all_seed_random_results[s]["per_K"][str(k_val)][fpr_key]["tpr"] for s in SEEDS]
            k_agg[fpr_key] = {
                "tpr_mean": float(np.mean(tprs)),
                "tpr_std": float(np.std(tprs)),
                "tpr_per_seed": {str(s): t for s, t in zip(SEEDS, tprs)},
            }
        random_agg[str(k_val)] = k_agg

    print(f"\nRandom key search results (no-diversity encryptor):")
    print(f"  {'K':>4} | TPR@FPR=1e-3 (mean+/-std) | TPR@FPR=1e-4 (mean+/-std)")
    print(f"  {'-'*65}")
    for k_val in K_VALUES:
        a = random_agg[str(k_val)]
        print(f"  {k_val:>4} | {a['fpr_0.001']['tpr_mean']:.4f} +/- {a['fpr_0.001']['tpr_std']:.4f}       "
              f"| {a['fpr_0.0001']['tpr_mean']:.4f} +/- {a['fpr_0.0001']['tpr_std']:.4f}")

    k1_tpr_1e3 = random_agg["1"]["fpr_0.001"]["tpr_mean"]
    k32_tpr_1e3 = random_agg["32"]["fpr_0.001"]["tpr_mean"]
    k64_tpr_1e3 = random_agg["64"]["fpr_0.001"]["tpr_mean"]
    print(f"\nTPR drop K=1->K=32 @FPR=1e-3: {(k1_tpr_1e3 - k32_tpr_1e3)*100:.1f}pp")
    print(f"TPR drop K=1->K=64 @FPR=1e-3: {(k1_tpr_1e3 - k64_tpr_1e3)*100:.1f}pp")

    output = {
        "encryptor": {k: v for k, v in best_enc_info.items() if k != "path"},
        "all_encryptor_results": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in all_enc_results.items()},
        "K_values": K_VALUES,
        "seeds": SEEDS,
        "layer": LAYER_IDX,
        "k1_monitor": {"per_seed": k1_results, "aggregated": k1_agg},
        "random_attack": {
            "per_seed": {str(s): v for s, v in all_seed_random_results.items()},
            "aggregated": random_agg,
        },
        "tpr_drops": {
            "k1_to_k32_fpr_1e3": float(k1_tpr_1e3 - k32_tpr_1e3),
            "k1_to_k64_fpr_1e3": float(k1_tpr_1e3 - k64_tpr_1e3),
        },
    }

    out_path = results_dir / "key_search_no_div.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
