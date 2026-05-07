# Activation extraction from frozen LLM: hooks into specified transformer
# layers to collect last-token hidden states. Memory-efficient: extracts
# last-token immediately per batch in the hook rather than storing full seqs.

import os
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_batch_mask = None


def extract_activations(
    input_ids,
    attention_mask,
    layers,
    model=None,
    batch_size=32,
    output_dir=None,
    split_name="data",
    device="cuda",
):
    global _batch_mask

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()

    captured = {layer_idx: [] for layer_idx in layers}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            mask = _batch_mask
            seq_lengths = mask.sum(dim=1) - 1
            batch_size_cur = hidden.shape[0]
            last_tok = hidden[torch.arange(batch_size_cur, device=hidden.device), seq_lengths.to(hidden.device)]
            captured[layer_idx].append(last_tok.detach().float().cpu())
        return hook_fn

    handles = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    n = input_ids.shape[0]
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_t = torch.tensor(attention_mask, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ids = input_ids_t[start:end].to(device)
            mask = attention_mask_t[start:end]
            _batch_mask = mask
            model(input_ids=ids, attention_mask=mask.to(device))
            if (start // batch_size) % 50 == 0:
                print(f"  Extracted batch {start//batch_size + 1}/{(n + batch_size - 1)//batch_size}")

    for h in handles:
        h.remove()

    results = {}
    for layer_idx in layers:
        all_last = torch.cat(captured[layer_idx], dim=0).numpy()
        results[layer_idx] = all_last

        if output_dir is not None:
            layer_dir = Path(output_dir) / f"layer_{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            np.save(layer_dir / f"{split_name}.npy", all_last)
            print(f"  Saved layer {layer_idx} {split_name}: shape {all_last.shape}")

    return results


def run_extraction_pipeline(
    train_data,
    test_data,
    alpaca_full_data,
    layers,
    output_dir,
    batch_size=32,
    device="cuda",
):
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print(f"\nExtracting train set ({train_data['input_ids'].shape[0]} samples)...")
    extract_activations(
        train_data["input_ids"], train_data["attention_mask"],
        layers, model=model, batch_size=batch_size,
        output_dir=output_dir, split_name="train", device=device,
    )

    print(f"\nExtracting test set ({test_data['input_ids'].shape[0]} samples)...")
    extract_activations(
        test_data["input_ids"], test_data["attention_mask"],
        layers, model=model, batch_size=batch_size,
        output_dir=output_dir, split_name="test", device=device,
    )

    print(f"\nExtracting Alpaca full ({alpaca_full_data['input_ids'].shape[0]} samples)...")
    extract_activations(
        alpaca_full_data["input_ids"], alpaca_full_data["attention_mask"],
        layers, model=model, batch_size=batch_size,
        output_dir=output_dir, split_name="alpaca_full", device=device,
    )

    np.save(Path(output_dir) / "train_labels.npy", train_data["labels"])
    np.save(Path(output_dir) / "test_labels.npy", test_data["labels"])
    print("\nExtraction complete.")
