"""
Quick test for claim 33: Verify that run_1.py code produces training accuracy data
for x_div_y dataset. Uses reduced steps (100 instead of 7500) to test code path.
"""
import sys
import os
import json
import numpy as np
import torch
import abc
import random
from itertools import permutations
from typing import Set

# Add repo root to path
REPO_ROOT = "/home/chenhui/fabscore/aiscientist_papers/20240802_012342_predictive_uncertainty"
sys.path.insert(0, REPO_ROOT)

# Read and exec the module-level definitions from run_1.py (everything before __main__)
with open(os.path.join(REPO_ROOT, "run_1.py"), "r") as f:
    source = f.read()

# Split at if __name__ == "__main__":
module_code = source.split("if __name__")[0]
exec(module_code, globals())

print(f"Functions available: run, train, evaluate, get_data, Transformer")

# Monkey-patch: run a quick test with just x_div_y, 1 seed, 100 steps
OUT_DIR = "/home/chenhui/fabscore/aiscientist_papers/20240802_012342_predictive_uncertainty/fabscore_claude/workspace/run_quick_out"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get data for x_div_y
train_loader, val_loader, n_vocab, n_output = get_data(
    operation="x_div_y",
    prime=97,
    training_fraction=0.5,
    batch_size=512,
)
print(f"Dataset: x_div_y, n_vocab={n_vocab}, n_output={n_output}")

# Create model (same as run_1.py)
model = Transformer(
    num_layers=2,
    dim_model=128,
    num_heads=4,
    vocab_size=n_vocab,
    output_size=n_output,
    seq_len=5,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.5)
num_train_batches = 10
num_eval_batches = 8
num_total_updates = 300  # reduced from 7500 for quick test
warmup_steps = 50
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: min(s/warmup_steps, 1))

train_log_info, val_log_info = [], []
step_val_acc_99 = num_total_updates

print(f"Running {num_total_updates} steps (reduced from 7500)...")
for ep in range(num_total_updates // num_train_batches):
    train_metrics = train(model, train_loader, optimizer, scheduler, device, num_train_batches)
    val_metrics = evaluate(model, val_loader, device, num_eval_batches)
    train_metrics["step"] = (ep + 1) * num_train_batches
    val_metrics["step"] = (ep + 1) * num_train_batches
    if step_val_acc_99 == num_total_updates and val_metrics["val_accuracy"] > 0.99:
        step_val_acc_99 = val_metrics["step"]
    train_log_info.append(train_metrics)
    val_log_info.append(val_metrics)
    if (ep + 1) % 5 == 0:
        print(f"  Step {train_metrics['step']}: train_acc={train_metrics['train_accuracy']:.4f}, val_acc={val_metrics['val_accuracy']:.4f}")

final_info = {
    "final_train_loss": train_metrics["train_loss"],
    "final_val_loss": val_metrics["val_loss"],
    "final_train_acc": train_metrics["train_accuracy"],
    "final_val_acc": val_metrics["val_accuracy"],
    "step_val_acc_99": step_val_acc_99,
}
print(f"\nFinal info: {final_info}")

# Save all_results.npy (just for x_div_y, seed 0)
all_results = {
    "x_div_y_0_final_info": final_info,
    "x_div_y_0_train_info": train_log_info,
    "x_div_y_0_val_info": val_log_info,
}
with open(os.path.join(OUT_DIR, "all_results.npy"), "wb") as f:
    np.save(f, all_results)

print(f"\nall_results.npy saved to {OUT_DIR}")
print(f"Training accuracy at step 10: {train_log_info[0]['train_accuracy']:.4f}")
print(f"Training accuracy at step {train_log_info[-1]['step']}: {train_log_info[-1]['train_accuracy']:.4f}")
print(f"Val accuracy at step 10: {val_log_info[0]['val_accuracy']:.4f}")
print(f"Val accuracy at step {val_log_info[-1]['step']}: {val_log_info[-1]['val_accuracy']:.4f}")

# Verify data structure matches what plot.py expects
loaded = np.load(os.path.join(OUT_DIR, "all_results.npy"), allow_pickle=True).item()
print(f"\nLoaded all_results.npy keys: {list(loaded.keys())}")
assert "x_div_y_0_train_info" in loaded
assert "x_div_y_0_val_info" in loaded
assert "train_accuracy" in loaded["x_div_y_0_train_info"][0]
assert "val_accuracy" in loaded["x_div_y_0_val_info"][0]
print("Data structure matches plot.py expectations: OK")
print("\nVERIFICATION COMPLETE: Code generates valid all_results.npy with training accuracy data")
