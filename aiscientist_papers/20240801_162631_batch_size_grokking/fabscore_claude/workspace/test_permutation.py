"""
Minimal verification script for claim 39:
Tests that experiment code can generate all_results.npy with permutation train_acc/train_loss data.
Uses fewer steps (200 instead of 5000) to run quickly.
"""
import sys
import os
import json
import numpy as np

# Add repo root to path
sys.path.insert(0, '/home/chenhui/fabscore/aiscientist_papers/20240801_162631_batch_size_grokking')

import torch
from experiment import get_data, Transformer, train, evaluate

out_dir = '/home/chenhui/fabscore/aiscientist_papers/20240801_162631_batch_size_grokking/fabscore_claude/workspace/perm_run'
os.makedirs(out_dir, exist_ok=True)

def quick_run(out_dir, dataset, seed_offset, initial_batch_size=32, doubling_interval=1000):
    """Same as experiment.run() but with only 200 steps"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337 + seed_offset)
    batch_size = initial_batch_size
    train_loader, val_loader, n_vocab, n_output = get_data(
        operation=dataset, prime=97, training_fraction=0.5, batch_size=batch_size
    )

    model = Transformer(
        num_layers=2, dim_model=128, num_heads=4,
        vocab_size=n_vocab, output_size=n_output, seq_len=5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.5)
    num_train_batches = 5
    num_eval_batches = 4
    num_total_updates = 200  # only 200 steps
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: min(s / warmup_steps, 1))

    final_info, train_log_info, val_log_info = [], [], []
    step_val_acc_99 = num_total_updates
    train_metrics, val_metrics = {}, {}
    for ep in range(num_total_updates // num_train_batches):
        if ep * num_train_batches % doubling_interval == 0 and ep > 0:
            batch_size *= 2
            train_loader, val_loader, n_vocab, n_output = get_data(
                operation=dataset, prime=97, training_fraction=0.5, batch_size=batch_size
            )

        train_metrics = train(model, train_loader, optimizer, scheduler, device, num_train_batches)
        val_metrics = evaluate(model, val_loader, device, num_eval_batches)
        train_metrics["step"] = (ep + 1) * num_train_batches
        val_metrics["step"] = (ep + 1) * num_train_batches

        if step_val_acc_99 == num_total_updates and val_metrics["val_accuracy"] > 0.99:
            step_val_acc_99 = val_metrics["step"]
        train_log_info.append(train_metrics)
        val_log_info.append(val_metrics)

    final_info = {
        "final_train_loss": train_metrics.get("train_loss", 0),
        "final_val_loss": val_metrics.get("val_loss", 0),
        "final_train_acc": train_metrics.get("train_accuracy", 0),
        "final_val_acc": val_metrics.get("val_accuracy", 0),
        "step_val_acc_99": step_val_acc_99,
    }
    return final_info, train_log_info, val_log_info

print("Running quick permutation test (200 steps, 1 seed)...")
final_info, train_info, val_info = quick_run(out_dir, "permutation", 0, initial_batch_size=32)
print(f"Final info: {final_info}")

# Build a minimal all_results dict
all_results = {
    "permutation_0_final_info": final_info,
    "permutation_0_train_info": train_info,
    "permutation_0_val_info": val_info,
}

# Save all_results.npy
npy_path = os.path.join(out_dir, "all_results.npy")
with open(npy_path, "wb") as f:
    np.save(f, all_results)
print(f"Saved all_results.npy to {npy_path}")

# Verify structure
loaded = np.load(npy_path, allow_pickle=True).item()
print(f"Keys in all_results: {list(loaded.keys())}")

train_sample = loaded["permutation_0_train_info"][0]
print(f"Train info sample (first step): {train_sample}")
val_sample = loaded["permutation_0_val_info"][0]
print(f"Val info sample (first step): {val_sample}")

# Verify required fields for Figure 3
assert "train_accuracy" in train_sample, "Missing train_accuracy"
assert "train_loss" in train_sample, "Missing train_loss"
assert "step" in train_sample, "Missing step"

print("\nAll required fields for Figure 3 (train_acc and train_loss for permutation) are present!")
print(f"Number of steps recorded: {len(train_info)}")
train_accs = [t["train_accuracy"] for t in train_info]
train_losses = [t["train_loss"] for t in train_info]
print(f"Train accuracy range: {min(train_accs):.4f} to {max(train_accs):.4f}")
print(f"Train loss range: {min(train_losses):.4f} to {max(train_losses):.4f}")
print("SUCCESS: Code generates valid training curves for permutation task")
