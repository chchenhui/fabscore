"""
Minimal test to verify run_1.py code path works and generates all_results.npy.
Runs only x_div_y dataset with 1 seed and 500 steps (instead of 7500).
Saves output to workspace dir.
"""
import sys
import os

# Add parent dir to path so we can import from run_1.py
sys.path.insert(0, "/home/chenhui/fabscore/aiscientist_papers/20240802_012342_predictive_uncertainty")

import json
import numpy as np
import torch

# Import from run_1.py by executing the module in a controlled way
# We'll replicate the key parts of run_1.py but with fewer steps

import abc
import random
from itertools import permutations
from typing import Set
from einops import rearrange, repeat
from torch import nn, Tensor
from torch.utils.data import IterableDataset

# Copy dataset classes from run_1.py
exec(open("/home/chenhui/fabscore/aiscientist_papers/20240802_012342_predictive_uncertainty/run_1.py").read().split("if __name__")[0])

# Run minimal test
out_dir = "/home/chenhui/fabscore/aiscientist_papers/20240802_012342_predictive_uncertainty/fabscore_claude/workspace/test_run_out"
os.makedirs(out_dir, exist_ok=True)

# Patch num_total_updates to 500 for quick test
import builtins
_orig_run = run

def run_minimal(out_dir, dataset, seed_offset):
    """Same as run but with 500 steps instead of 7500."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed_offset)
    random.seed(seed_offset)
    np.random.seed(seed_offset)

    # Use same setup as run_1.py but with 500 steps
    train_dataset = get_dataset(dataset, frac_train=0.3, seed=seed_offset)
    val_dataset = get_dataset(dataset, frac_train=0.3, seed=seed_offset, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512)

    model = Transformer(
        num_layers=2,
        d_vocab=train_dataset.n_vocab,
        d_model=128,
        d_mlp=512,
        num_heads=4,
        act_type="relu",
        use_cache=False,
        seq_len=5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.5)
    num_train_batches = 10
    num_eval_batches = 8
    num_total_updates = 500  # reduced from 7500
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: min(s/warmup_steps, 1))

    final_info, train_log_info, val_log_info = [], [], []
    step_val_acc_99 = num_total_updates

    for ep in range(num_total_updates // num_train_batches):
        train_metrics = train(model, train_loader, optimizer, scheduler, device, num_train_batches)
        val_metrics = evaluate(model, val_loader, device, num_eval_batches)
        train_metrics["step"] = (ep + 1) * num_train_batches
        val_metrics["step"] = (ep + 1) * num_train_batches
        if step_val_acc_99 == num_total_updates and val_metrics["val_accuracy"] > 0.99:
            step_val_acc_99 = val_metrics["step"]
        train_log_info.append(train_metrics)
        val_log_info.append(val_metrics)

    final_info = {
        "final_train_loss": train_metrics["train_loss"],
        "final_val_loss": val_metrics["val_loss"],
        "final_train_acc": train_metrics["train_accuracy"],
        "final_val_acc": val_metrics["val_accuracy"],
        "step_val_acc_99": step_val_acc_99,
    }
    print(f"Final info: {final_info}")
    with open(os.path.join(out_dir, f"final_info_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(final_info, f)
    return final_info, train_log_info, val_log_info

# Run just x_div_y with 1 seed
all_results = {}
final_info, train_info, val_info = run_minimal(out_dir, "x_div_y", 0)
all_results["x_div_y_0_final_info"] = final_info
all_results["x_div_y_0_train_info"] = train_info
all_results["x_div_y_0_val_info"] = val_info

with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
    np.save(f, all_results)

print(f"\nall_results.npy saved to {out_dir}")
print(f"Keys: {list(all_results.keys())}")
print(f"Train info length: {len(train_info)} steps")
print(f"First train acc: {train_info[0]['train_accuracy']:.4f}")
print(f"Last train acc: {train_info[-1]['train_accuracy']:.4f}")
print(f"First val acc: {val_info[0]['val_accuracy']:.4f}")
print(f"Last val acc: {val_info[-1]['val_accuracy']:.4f}")
