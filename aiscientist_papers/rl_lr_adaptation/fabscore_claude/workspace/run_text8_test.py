"""
Quick test for text8 to verify Figure 3 code path.
Patches max_iters to 300 (with eval_interval=100) for a quick run.
"""
import sys
sys.path.insert(0, '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation')

import os
import numpy as np
import importlib
import importlib.util

# Read the source, patch max_iters for text8
src_path = '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation/experiment.py'
with open(src_path, 'r') as f:
    src = f.read()

# Patch: for enwik8/text8, max_iters=100000 -> 300, eval_interval=1000 -> 100, log_interval=100 -> 50
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
patched_path = '/tmp/experiment_patched_text8.py'
with open(patched_path, 'w') as f:
    f.write(src_patched)

# Load it as a module
spec = importlib.util.spec_from_file_location("experiment_patched_text8", patched_path)
exp_patched = importlib.util.module_from_spec(spec)
sys.modules["experiment_patched_text8"] = exp_patched
spec.loader.exec_module(exp_patched)

out_dir = '/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation/fabscore_claude/workspace/text8_test_run'
os.makedirs(out_dir, exist_ok=True)

os.chdir('/home/chenhui/fabscore/aiscientist_papers/rl_lr_adaptation')

print("Starting quick text8 training (300 iters)...")
final_info, train_info, val_info = exp_patched.train("text8", out_dir, seed_offset=0)

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
    "text8_0_final_info": final_info,
    "text8_0_train_info": train_info,
    "text8_0_val_info": val_info,
}
npy_path = os.path.join(out_dir, "all_results.npy")
with open(npy_path, "wb") as f:
    np.save(f, all_results)
print(f"\nall_results.npy saved to {npy_path}")
results = np.load(npy_path, allow_pickle=True).item()
print(f"Keys in all_results.npy: {list(results.keys())}")
