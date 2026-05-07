# Debug: verify new MIGU register_hook approach works with DeepSpeed ZeRO-2.

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.trace_data import DEFAULT_ORDER, build_dataloaders
from training.utils import set_seed, load_config, to_device, print_rank_0, load_model_and_tokenizer, get_optimizer_grouped_parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    local_rank = args.local_rank
    set_seed(42)

    if local_rank == -1:
        device = torch.device("cuda")
        global_rank = 0
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        deepspeed.init_distributed()
        global_rank = dist.get_rank()

    model, tokenizer = load_model_and_tokenizer(config["model_path"], bf16=True, gradient_checkpointing=True)

    with open(config["deepspeed_config"]) as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = 2
    ws = dist.get_world_size() if dist.is_initialized() else 1
    ds_config["train_batch_size"] = 2 * ws
    ds_config["gradient_accumulation_steps"] = 1

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, 0.0)
    from deepspeed.ops.adam import FusedAdam
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=1e-5, betas=(0.9, 0.95))

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config,
        args=argparse.Namespace(local_rank=local_rank),
    )

    all_train_dataloaders = build_dataloaders(
        config["data_path"], DEFAULT_ORDER, tokenizer, split="train",
        batch_size=2, max_prompt_len=512, max_ans_len=256,
        shuffle=True, distributed=dist.is_initialized(), local_rank=local_rank,
    )
    train_dataloader = all_train_dataloaders[DEFAULT_ORDER[0]]
    model_engine.train()

    print_rank_0("=== Step 1: Train step WITHOUT MIGU (baseline) ===", global_rank)
    batch = next(iter(train_dataloader))
    for key in ["sources", "gts"]:
        if key in batch:
            del batch[key]
    batch = to_device(batch, device)

    unwrapped = model_engine.module if hasattr(model_engine, 'module') else model_engine
    w0_before = {}
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, nn.Linear):
            w0_before[name] = mod.weight.data.clone()
            break

    outputs = model_engine(**batch, use_cache=False)
    loss = outputs.loss
    print_rank_0(f"  Loss: {loss.item():.4f}", global_rank)
    model_engine.backward(loss)
    model_engine.step()

    w0_after = {}
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, nn.Linear):
            w0_after[name] = mod.weight.data.clone()
            break

    for name in w0_before:
        diff = (w0_after[name] - w0_before[name]).abs()
        nonzero = (diff > 0).sum().item()
        total = diff.numel()
        print_rank_0(f"  {name}: {nonzero}/{total} params changed ({100*nonzero/total:.1f}%)", global_rank)

    print_rank_0("\n=== Step 2: Train step WITH MIGU hooks ===", global_rank)
    from methods.migu import MIGUHooks
    hooks = MIGUHooks(model_engine, threshold=0.7)
    print_rank_0(f"  Registered forward hooks + backward param hooks", global_rank)

    w1_before = {}
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, nn.Linear):
            w1_before[name] = mod.weight.data.clone()

    batch2 = next(iter(train_dataloader))
    for key in ["sources", "gts"]:
        if key in batch2:
            del batch2[key]
    batch2 = to_device(batch2, device)

    outputs2 = model_engine(**batch2, use_cache=False)
    loss2 = outputs2.loss
    print_rank_0(f"  Loss: {loss2.item():.4f}", global_rank)
    print_rank_0(f"  Activations cached: {len(hooks.temporal_activation_sum)} layers", global_rank)
    model_engine.backward(loss2)
    model_engine.step()
    hooks.step_done()

    masked_layers = 0
    full_update_layers = 0
    sample_count = 0
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, nn.Linear) and name in w1_before:
            diff = (mod.weight.data - w1_before[name]).abs()
            rows_with_update = (diff.sum(dim=-1) > 0).sum().item()
            total_rows = diff.shape[0]
            frac = rows_with_update / total_rows
            if frac < 0.95:
                masked_layers += 1
            else:
                full_update_layers += 1
            if sample_count < 5:
                print_rank_0(f"  {name}: {rows_with_update}/{total_rows} rows updated ({100*frac:.1f}%)", global_rank)
                sample_count += 1

    print_rank_0(f"\n  Summary: {masked_layers} layers partially updated, {full_update_layers} layers fully updated", global_rank)
    if masked_layers > 0:
        print_rank_0("  MIGU gradient masking is WORKING!", global_rank)
    else:
        print_rank_0("  WARNING: MIGU masking appears NOT to be working!", global_rank)

    hooks.remove()
    print_rank_0("\n=== Debug v3 complete ===", global_rank)

if __name__ == "__main__":
    main()
