# Debug script v2: explore DeepSpeed ZeRO-2 gradient storage to find where grads live.

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.trace_data import TASK_EPOCHS, DEFAULT_ORDER, build_dataloaders, load_tokenizer
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

    batch = next(iter(train_dataloader))
    for key in ["sources", "gts"]:
        if key in batch:
            del batch[key]
    batch = to_device(batch, device)

    outputs = model_engine(**batch, use_cache=False)
    loss = outputs.loss
    model_engine.backward(loss)

    unwrapped = model_engine.module if hasattr(model_engine, 'module') else model_engine

    print_rank_0("=== Checking param.grad vs param.main_grad ===", global_rank)
    checked = 0
    for name, param in unwrapped.named_parameters():
        if not param.requires_grad:
            continue
        has_grad = param.grad is not None
        has_main_grad = hasattr(param, 'main_grad') and param.main_grad is not None
        has_ds_grad = hasattr(param, 'ds_grad') and param.ds_grad is not None
        if checked < 5:
            print_rank_0(f"  {name}: .grad={has_grad}, .main_grad={has_main_grad}, .ds_grad={has_ds_grad}, shape={param.shape}", global_rank)
        checked += 1

    print_rank_0(f"  Total checked: {checked}", global_rank)

    print_rank_0("\n=== Checking model_engine attributes ===", global_rank)
    for attr in ['optimizer', '_global_grad_norm', 'grad_partitions_flat_buffer']:
        has_attr = hasattr(model_engine, attr)
        print_rank_0(f"  model_engine.{attr} exists: {has_attr}", global_rank)

    print_rank_0("\n=== Checking optimizer internals ===", global_rank)
    opt = model_engine.optimizer
    print_rank_0(f"  Optimizer type: {type(opt)}", global_rank)
    for attr in ['grad_partitions_flat_buffer', 'averaged_gradients', 'ipg_buffer', 'single_partition_of_fp32_groups', 'fp16_groups', 'fp16_partitioned_groups', 'fp32_partitioned_groups_flat', 'grad_position']:
        has_attr = hasattr(opt, attr)
        print_rank_0(f"  optimizer.{attr} exists: {has_attr}", global_rank)

    if hasattr(opt, 'averaged_gradients'):
        print_rank_0(f"\n=== averaged_gradients info ===", global_rank)
        for gidx, grads in enumerate(opt.averaged_gradients):
            if grads:
                print_rank_0(f"  Group {gidx}: {len(grads)} grads, first shape: {grads[0].shape if grads[0] is not None else None}", global_rank)

    print_rank_0("\n=== Trying backward hook approach ===", global_rank)
    grad_data = {}
    def make_grad_hook(name):
        def hook(grad):
            grad_data[name] = grad.norm().item()
            return grad
        return hook

    handles = []
    for name, param in unwrapped.named_parameters():
        if param.requires_grad:
            h = param.register_hook(make_grad_hook(name))
            handles.append(h)
            if len(handles) >= 200:
                break

    batch2 = next(iter(train_dataloader))
    for key in ["sources", "gts"]:
        if key in batch2:
            del batch2[key]
    batch2 = to_device(batch2, device)

    model_engine.step()

    outputs2 = model_engine(**batch2, use_cache=False)
    loss2 = outputs2.loss
    model_engine.backward(loss2)

    print_rank_0(f"  Gradient hooks fired for {len(grad_data)} params (out of {len(handles)} registered)", global_rank)
    for i, (name, norm) in enumerate(list(grad_data.items())[:5]):
        print_rank_0(f"    {name}: grad norm = {norm:.6f}", global_rank)

    for h in handles:
        h.remove()

    model_engine.step()
    print_rank_0("\n=== Debug v2 complete ===", global_rank)

if __name__ == "__main__":
    main()
