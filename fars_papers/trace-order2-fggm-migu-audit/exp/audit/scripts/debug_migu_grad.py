# Debug script: verify MIGU gradient masking works with DeepSpeed ZeRO-2.
# Tests whether module.weight.grad is accessible and modifiable after backward().

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.trace_data import TASK_EPOCHS, DEFAULT_ORDER, TraceDataset, TraceDataCollator, build_dataloaders, load_tokenizer
from training.utils import set_seed, load_config, to_device, print_rank_0, save_hf_format, load_model_and_tokenizer, get_optimizer_grouped_parameters

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

    model_path = config["model_path"]
    ds_config_path = config["deepspeed_config"]
    data_path = config["data_path"]

    print_rank_0(f"Loading model from {model_path}", global_rank)
    model, tokenizer = load_model_and_tokenizer(model_path, bf16=True, gradient_checkpointing=True)

    with open(ds_config_path) as f:
        ds_config = json.load(f)
    ds_config["train_micro_batch_size_per_gpu"] = 2
    ds_config["train_batch_size"] = 2 * (dist.get_world_size() if dist.is_initialized() else 1)
    ds_config["gradient_accumulation_steps"] = 1

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, 0.0)
    from deepspeed.ops.adam import FusedAdam
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=1e-5, betas=(0.9, 0.95))

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer,
        config=ds_config,
        args=argparse.Namespace(local_rank=local_rank),
    )

    task_order = DEFAULT_ORDER
    all_train_dataloaders = build_dataloaders(
        data_path, task_order, tokenizer, split="train",
        batch_size=2, max_prompt_len=512, max_ans_len=256,
        shuffle=True, distributed=dist.is_initialized(), local_rank=local_rank,
    )

    train_dataloader = all_train_dataloaders[task_order[0]]
    model_engine.train()

    batch = next(iter(train_dataloader))
    for key in ["sources", "gts"]:
        if key in batch:
            del batch[key]
    batch = to_device(batch, device)

    print_rank_0("=== Step 1: Forward + backward WITHOUT MIGU hooks ===", global_rank)
    outputs = model_engine(**batch, use_cache=False)
    loss = outputs.loss
    model_engine.backward(loss)

    none_count = 0
    has_grad_count = 0
    unwrapped = model_engine.module if hasattr(model_engine, 'module') else model_engine
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, nn.Linear):
            if mod.weight.grad is None:
                none_count += 1
            else:
                has_grad_count += 1
                if has_grad_count <= 3:
                    print_rank_0(f"  [has grad] {name}: grad shape={mod.weight.grad.shape}, norm={mod.weight.grad.norm().item():.6f}", global_rank)

    print_rank_0(f"  Linear layers with grad: {has_grad_count}, without grad (None): {none_count}", global_rank)
    model_engine.step()

    print_rank_0("\n=== Step 2: Forward + backward WITH MIGU hooks ===", global_rank)
    from methods.migu import MIGUHooks
    hooks = MIGUHooks(model_engine, threshold=0.7)
    print_rank_0(f"  Registered hooks on {len(hooks._module_map)} linear layers", global_rank)

    batch2 = next(iter(train_dataloader))
    for key in ["sources", "gts"]:
        if key in batch2:
            del batch2[key]
    batch2 = to_device(batch2, device)

    outputs2 = model_engine(**batch2, use_cache=False)
    loss2 = outputs2.loss
    model_engine.backward(loss2)

    print_rank_0(f"  Activation cache: {len(hooks.temporal_activation_sum)} layers", global_rank)

    none_count2 = 0
    has_grad_count2 = 0
    grad_norms_before = {}
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, nn.Linear):
            if mod.weight.grad is None:
                none_count2 += 1
            else:
                has_grad_count2 += 1
                grad_norms_before[name] = mod.weight.grad.norm().item()

    print_rank_0(f"  Before masking - Linear layers with grad: {has_grad_count2}, without grad: {none_count2}", global_rank)

    hooks.apply_gradient_mask()

    masked_count = 0
    unchanged_count = 0
    for name, mod in unwrapped.named_modules():
        if isinstance(mod, nn.Linear) and mod.weight.grad is not None:
            grad_norm_after = mod.weight.grad.norm().item()
            grad_norm_before = grad_norms_before.get(name, 0)
            if abs(grad_norm_after - grad_norm_before) > 1e-8:
                masked_count += 1
                if masked_count <= 5:
                    print_rank_0(f"  [MASKED] {name}: norm before={grad_norm_before:.6f}, after={grad_norm_after:.6f}, ratio={grad_norm_after/(grad_norm_before+1e-12):.4f}", global_rank)
            else:
                unchanged_count += 1

    print_rank_0(f"  After masking - MASKED: {masked_count}, UNCHANGED: {unchanged_count}", global_rank)
    if masked_count == 0 and unchanged_count > 0:
        print_rank_0("  *** WARNING: NO GRADIENTS WERE MODIFIED BY MIGU! Masking is NOT working! ***", global_rank)
    elif none_count2 > 0:
        print_rank_0(f"  *** WARNING: {none_count2} linear layers had None grad - masking cannot apply to these! ***", global_rank)

    model_engine.step()
    hooks.remove()
    print_rank_0("\n=== Debug complete ===", global_rank)

if __name__ == "__main__":
    main()
