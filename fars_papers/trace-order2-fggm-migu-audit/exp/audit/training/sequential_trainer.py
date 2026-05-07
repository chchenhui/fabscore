# Sequential task-by-task training loop for TRACE continual learning.
# Matches TRACE CL_Base_Model pattern: single DeepSpeed engine + optimizer across
# all tasks, constant LR after warmup, save final checkpoint per task (no best
# checkpoint selection between tasks).

import os
import sys
import json
import math
import argparse
import torch
import torch.distributed as dist
import deepspeed
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.trace_data import (
    TASK_EPOCHS, DEFAULT_ORDER, ORDER_2, TraceDataset, TraceDataCollator,
    build_dataloaders, load_tokenizer,
)
from training.utils import (
    set_seed, load_config, to_device, print_rank_0, get_all_reduce_mean,
    save_hf_format, load_model_and_tokenizer, get_optimizer_grouped_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def evaluate_on_eval_set(model, eval_dataloader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in eval_dataloader:
        for key in ["sources", "gts"]:
            if key in batch:
                del batch[key]
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
            total_loss += outputs.loss.float().item()
            total_steps += 1
    model.train()
    avg_loss = total_loss / max(total_steps, 1)
    return avg_loss


def run_sequential_training(config: dict, local_rank: int):
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

    seed = config.get("seed", 42)
    set_seed(seed)

    method = config.get("method", "sft")
    task_order = config.get("task_order", DEFAULT_ORDER)
    data_path = config["data_path"]
    model_path = config["model_path"]
    output_dir = config["output_dir"]
    ds_config_path = config["deepspeed_config"]
    max_prompt_len = config.get("max_prompt_len", 1024)
    max_ans_len = config.get("max_ans_len", 512)
    per_device_train_batch_size = config.get("per_device_train_batch_size", 16)
    per_device_eval_batch_size = config.get("per_device_eval_batch_size", 16)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    learning_rate = config.get("learning_rate", 1e-5)
    weight_decay = config.get("weight_decay", 0.0)
    eval_steps = config.get("eval_steps", 150)
    gradient_checkpointing = config.get("gradient_checkpointing", True)
    bf16 = config.get("bf16", True)
    num_warmup_steps = config.get("num_warmup_steps", 0)

    if local_rank == -1:
        device = torch.device("cuda")
        global_rank = 0
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        deepspeed.init_distributed()
        global_rank = dist.get_rank()

    import wandb
    wandb_project = os.environ.get("WANDB_PROJECT", "trace-order2-fggm-migu-audit")
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    if global_rank == 0:
        wandb.init(
            project=wandb_project,
            name=f"{method}_{'_'.join(task_order[:2])}..._{config.get('order_name', 'default')}_seed{seed}",
            config=config,
            mode=wandb_mode,
        )

    print_rank_0(f"Loading model from {model_path}", global_rank)
    model, tokenizer = load_model_and_tokenizer(
        model_path, bf16=bf16, gradient_checkpointing=gradient_checkpointing
    )

    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["train_batch_size"] = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * (dist.get_world_size() if dist.is_initialized() else 1)
    )
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps

    method_hooks = None
    if method == "sft":
        pass
    elif method == "migu":
        from methods.migu import get_hooks
        method_hooks = get_hooks(config)
    elif method == "fggm":
        from methods.fggm import get_hooks
        method_hooks = get_hooks(config)

    epoch_override = config.get("epoch_override", None)

    # Build all dataloaders upfront
    all_train_dataloaders = build_dataloaders(
        data_path, task_order, tokenizer, split="train",
        batch_size=per_device_train_batch_size,
        max_prompt_len=max_prompt_len, max_ans_len=max_ans_len,
        shuffle=True, distributed=dist.is_initialized(), local_rank=local_rank,
    )
    all_eval_dataloaders = build_dataloaders(
        data_path, task_order, tokenizer, split="eval",
        batch_size=per_device_eval_batch_size,
        max_prompt_len=max_prompt_len, max_ans_len=max_ans_len,
        shuffle=False, distributed=False,
    )

    # Single optimizer and LR scheduler across all tasks (matching TRACE reference)
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, weight_decay)
    from deepspeed.ops.adam import FusedAdam
    optimizer = FusedAdam(
        optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.95),
    )

    from transformers import get_constant_schedule_with_warmup
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
        args=argparse.Namespace(local_rank=local_rank),
    )

    global_step = 0

    # Resume: find the last completed checkpoint and load its weights
    last_completed_idx = -1
    for tidx in range(len(task_order)):
        ckpt_dir = os.path.join(output_dir, str(tidx))
        if os.path.exists(ckpt_dir) and os.path.exists(os.path.join(ckpt_dir, "config.json")):
            last_completed_idx = tidx
        else:
            break
    if last_completed_idx >= 0:
        resume_ckpt = os.path.join(output_dir, str(last_completed_idx))
        print_rank_0(f"RESUME: Loading weights from last completed checkpoint {last_completed_idx} at {resume_ckpt}", global_rank)
        from transformers import AutoModelForCausalLM
        state_dict = AutoModelForCausalLM.from_pretrained(resume_ckpt, torch_dtype=torch.bfloat16).state_dict()
        model_engine.module.load_state_dict(state_dict, strict=True)
        del state_dict
        torch.cuda.empty_cache()
        print_rank_0(f"RESUME: Weights loaded successfully from checkpoint {last_completed_idx}", global_rank)

    for task_idx, task_name in enumerate(task_order):
        epochs = epoch_override if epoch_override is not None else TASK_EPOCHS[task_name]
        print_rank_0(f"\n{'='*60}", global_rank)
        print_rank_0(f"Task {task_idx}/{len(task_order)-1}: {task_name} ({epochs} epochs)", global_rank)
        print_rank_0(f"{'='*60}", global_rank)

        # Resume: skip if final checkpoint already exists
        final_checkpoint_dir = os.path.join(output_dir, str(task_idx))
        if os.path.exists(final_checkpoint_dir) and os.path.exists(os.path.join(final_checkpoint_dir, "config.json")):
            print_rank_0(f"  RESUME: Skipping task {task_idx} ({task_name}), checkpoint exists at {final_checkpoint_dir}", global_rank)
            continue

        train_dataloader = all_train_dataloaders[task_name]
        eval_dataloader = all_eval_dataloaders[task_name]

        total_steps_this_task = epochs * len(train_dataloader)

        hooks = None
        if method_hooks is not None:
            hooks = method_hooks(model_engine, task_idx,
                                  train_dataloader=train_dataloader, device=device)

        progress_bar = tqdm(total=total_steps_this_task, leave=True, disable=(global_rank != 0))
        step_in_task = 0

        model_engine.train()
        for epoch in range(epochs):
            print_rank_0(f"Epoch {epoch+1}/{epochs}, batches: {len(train_dataloader)}", global_rank)
            for step, batch in enumerate(train_dataloader):
                if "sources" in batch:
                    del batch["sources"]
                if "gts" in batch:
                    del batch["gts"]
                batch = to_device(batch, device)

                outputs = model_engine(**batch, use_cache=False)
                loss = outputs.loss

                model_engine.backward(loss)
                model_engine.step()
                if hooks is not None and hasattr(hooks, 'step_done'):
                    hooks.step_done()

                global_step += 1
                step_in_task += 1

                loss_val = loss.float().item()

                if global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Task {task_idx}: {task_name} | Epoch {epoch+1} | Loss: {loss_val:.4f}"
                    )
                    wandb.log({
                        "train/loss": loss_val,
                        "train/lr": lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, 'get_last_lr') else learning_rate,
                        "train/task_idx": task_idx,
                        "train/global_step": global_step,
                    }, step=global_step)

                if step_in_task % eval_steps == 0 and step_in_task > 0:
                    print_rank_0(f"  [Step {step_in_task}] Running eval on {task_name} eval set...", global_rank)
                    eval_loss = evaluate_on_eval_set(model_engine, eval_dataloader, device)
                    print_rank_0(f"  [Step {step_in_task}] Eval loss: {eval_loss:.4f}", global_rank)
                    if global_rank == 0:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/task_idx": task_idx,
                            "eval/task_name": task_name,
                        }, step=global_step)
                    model_engine.train()

        # End-of-task eval
        eval_loss = evaluate_on_eval_set(model_engine, eval_dataloader, device)
        print_rank_0(f"  [End of task {task_idx}] Final eval loss: {eval_loss:.4f}", global_rank)
        if global_rank == 0:
            wandb.log({
                "eval/loss": eval_loss,
                "eval/task_idx": task_idx,
                "eval/task_name": task_name,
                "eval/end_of_task": True,
            }, step=global_step)

        # Save final checkpoint (matching TRACE: just save after each task, no best selection)
        if global_rank == 0:
            save_hf_format(model_engine, tokenizer, output_dir, sub_folder=str(task_idx))

        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

        if hooks is not None and hasattr(hooks, "remove"):
            hooks.remove()

        progress_bar.close()
        print_rank_0(f"  Task {task_idx} ({task_name}) complete. Checkpoint saved to {final_checkpoint_dir}", global_rank)

    if global_rank == 0:
        wandb.finish()
    print_rank_0("Sequential training complete!", global_rank)


def main():
    args = parse_args()
    config = load_config(args.config)
    run_sequential_training(config, args.local_rank)


if __name__ == "__main__":
    main()
