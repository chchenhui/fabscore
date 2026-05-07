# Extract encrypted activations from frozen Qwen2.5-7B-Instruct for monitor training.
# Uses best opt2 encryptor (seed_123, lowest val KL=0.0226) to encrypt embeddings,
# then hooks layer 27 to capture last-token hidden states.
# Saves train/test/alpaca_full activations + labels to outputs/activations/encrypted_train/.

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from key_search_bypass.data import build_balanced_dataset
from key_search_bypass.data.alpaca import load_alpaca_full
from key_search_bypass.encryptor.model import KeyConditionedEncryptor

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LAYER_IDX = 27
KEY_DIM = 128
ENCRYPTOR_CKPT = PROJ_DIR / "key_search_bypass" / "outputs" / "encryptor" / "seed_123_opt2" / "last_checkpoint.pt"
OUTPUT_DIR = PROJ_DIR / "key_search_bypass" / "outputs" / "activations" / "encrypted_train" / f"layer_{LAYER_IDX}"

_batch_mask = None


def extract_encrypted_activations(
    input_ids, attention_mask, model, encryptor, layer_idx, batch_size=8, device="cuda",
):
    global _batch_mask
    captured = []

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        mask = _batch_mask
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
            _batch_mask = mask

            clean_embeds = model.model.embed_tokens(ids)
            k = KeyConditionedEncryptor.sample_key(
                ids.shape[0], KEY_DIM, device=device, dtype=clean_embeds.dtype
            )
            enc_embeds = encryptor(clean_embeds, k)

            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 1)
            model.model(
                inputs_embeds=enc_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
            )

            if (start // batch_size) % 50 == 0:
                print(f"  Batch {start // batch_size + 1}/{(n + batch_size - 1) // batch_size}")

    handle.remove()
    all_acts = torch.cat(captured, dim=0).numpy()
    return all_acts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples for sanity check (0=all)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

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
    print(f"  Encryptor loaded (step {ckpt['step']}, eval_kl={ckpt['eval_kl']:.6f})")

    print("Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, test_data = build_balanced_dataset(tokenizer=tokenizer)
    alpaca_full = load_alpaca_full(tokenizer=tokenizer)

    if args.max_samples > 0:
        print(f"  SANITY CHECK MODE: limiting to {args.max_samples} samples per split")
        for d in [train_data, test_data, alpaca_full]:
            for k in ["input_ids", "attention_mask", "labels"]:
                d[k] = d[k][:args.max_samples]

    print(f"  Train: {len(train_data['labels'])} samples")
    print(f"  Test: {len(test_data['labels'])} samples")
    print(f"  Alpaca full: {len(alpaca_full['labels'])} samples")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_base = OUTPUT_DIR.parent

    print(f"\nExtracting encrypted activations (layer {LAYER_IDX})...")

    print(f"\n--- Train set ---")
    train_acts = extract_encrypted_activations(
        train_data["input_ids"], train_data["attention_mask"],
        model, encryptor, LAYER_IDX, batch_size=args.batch_size,
    )
    np.save(OUTPUT_DIR / "train.npy", train_acts)
    print(f"  Saved train: shape {train_acts.shape}")

    print(f"\n--- Test set ---")
    test_acts = extract_encrypted_activations(
        test_data["input_ids"], test_data["attention_mask"],
        model, encryptor, LAYER_IDX, batch_size=args.batch_size,
    )
    np.save(OUTPUT_DIR / "test.npy", test_acts)
    print(f"  Saved test: shape {test_acts.shape}")

    print(f"\n--- Alpaca full ---")
    alpaca_acts = extract_encrypted_activations(
        alpaca_full["input_ids"], alpaca_full["attention_mask"],
        model, encryptor, LAYER_IDX, batch_size=args.batch_size,
    )
    np.save(OUTPUT_DIR / "alpaca_full.npy", alpaca_acts)
    print(f"  Saved alpaca_full: shape {alpaca_acts.shape}")

    np.save(out_base / "train_labels.npy", train_data["labels"])
    np.save(out_base / "test_labels.npy", test_data["labels"])
    print(f"\nSaved labels to {out_base}")

    print("\nActivation stats:")
    for name, acts in [("train", train_acts), ("test", test_acts), ("alpaca_full", alpaca_acts)]:
        print(f"  {name}: mean={acts.mean():.4f}, std={acts.std():.4f}, "
              f"min={acts.min():.4f}, max={acts.max():.4f}, "
              f"any_nan={np.isnan(acts).any()}, any_inf={np.isinf(acts).any()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
