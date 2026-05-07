"""
Quick test for enwik8 to verify Figure 2 code path.
Patches max_iters to 500 (with eval_interval=100) for a quick run.
"""
import sys
sys.path.insert(0, '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation')

import os
import numpy as np
import json

# We need to patch experiment.py's train() to use fewer iterations
# We do this by monkeypatching the constants inside the function after import
import experiment

# Save original train
original_train = experiment.train

# Override train to patch enwik8 hyperparams
def patched_train(dataset="shakespeare_char", out_dir="run_0", seed_offset=0):
    """Run train with patched max_iters for enwik8 quick test."""
    import experiment as exp_module
    # Monkey-patch the function's closure/globals
    # We can't easily do this, so we'll use a different approach:
    # inspect the function and temporarily replace the relevant constants
    # Actually, just call original and let it run -- but that's 100k iters
    # Instead, let's read and exec a modified version
    return original_train(dataset, out_dir, seed_offset)

# Better approach: directly call train() from experiment.py
# but modify the key parameters via source inspection
# Let's just run with reduced iters using a temp script approach

import importlib
import types

# Read the source, patch max_iters for enwik8
src_path = '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation/experiment.py'
with open(src_path, 'r') as f:
    src = f.read()

# Patch: for enwik8, max_iters=100000 -> 300, eval_interval=1000 -> 100, log_interval=100 -> 50
src_patched = src.replace(
    'max_iters = 5000 if dataset == "shakespeare_char" else 100000',
    'max_iters = 5000 if dataset == "shakespeare_char" else 300'
).replace(
    'eval_interval = 250 if dataset == "shakespeare_char" else 1000',
    'eval_interval = 250 if dataset == "shakespeare_char" else 100'
).replace(
    'log_interval = 10 if dataset == "shakespeare_char" else 100',
    'log_interval = 10 if dataset == "shakespeare_char" else 50'
)

# Write the patched version to a temp file
patched_path = '/tmp/experiment_patched_enwik8.py'
with open(patched_path, 'w') as f:
    f.write(src_patched)

# Load it as a module
spec = importlib.util.spec_from_file_location("experiment_patched", patched_path)
exp_patched = importlib.util.module_from_spec(spec)
sys.modules["experiment_patched"] = exp_patched
spec.loader.exec_module(exp_patched)

out_dir = '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation/fabscore_claude/workspace/enwik8_test_run'
os.makedirs(out_dir, exist_ok=True)

import os
os.chdir('/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation')

print("Starting quick enwik8 training (300 iters)...")
final_info, train_info, val_info = exp_patched.train("enwik8", out_dir, seed_offset=0)

print(f"Training complete!")
print(f"Final train loss: {final_info['final_train_loss']:.4f}")
print(f"Best val loss: {final_info['best_val_loss']:.4f}")
print(f"Training time: {final_info['total_train_time']:.2f}s")
print(f"Train info count: {len(train_info)}")
print(f"Val info count: {len(val_info)}")
if train_info:
    print(f"Train info sample keys: {list(train_info[0].keys())}")
if val_info:
    print(f"Val info sample keys: {list(val_info[0].keys())}")
    print(f"First val entry: {val_info[0]}")

# Save all_results.npy
all_results = {
    "enwik8_0_final_info": final_info,
    "enwik8_0_train_info": train_info,
    "enwik8_0_val_info": val_info,
}
npy_path = os.path.join(out_dir, "all_results.npy")
with open(npy_path, "wb") as f:
    np.save(f, all_results)
print(f"\nall_results.npy saved to {npy_path}")
results = np.load(npy_path, allow_pickle=True).item()
print(f"Keys in all_results.npy: {list(results.keys())}")
