"""Minimal verification script for Claim 2 (Figure 2a: Modular Division dynamics).
Runs just x_div_y with seed 0 to get training loss, val accuracy, and invariance score over time.
Saves results to workspace for verification.
"""

import sys
import os

# Add the repo root to path
repo_root = "/home/chenhui/fabscore/aiscientist_papers/20240804_065223_invariance_learning_grokking"
sys.path.insert(0, repo_root)
os.chdir(repo_root)

import abc
import random
from itertools import permutations
from typing import Set
import json
import numpy as np
from einops import rearrange, repeat
import torch
from torch.utils.data import IterableDataset
from torch import nn, Tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import all needed classes/functions from run_4.py
import importlib.util
spec = importlib.util.spec_from_file_location("run_4", os.path.join(repo_root, "run_4.py"))
run4_mod = importlib.util.load_from_spec = None

# Instead, let's directly import by exec
exec_globals = {}
with open(os.path.join(repo_root, "run_4.py"), "r") as f:
    source = f.read()

# We'll only execute the function definitions, not the main block
# Find the main block and stop before it
lines = source.split('\n')
main_start = None
for i, line in enumerate(lines):
    if line.strip() == 'if __name__ == "__main__":' or line.strip().startswith('parser = argparse'):
        main_start = i
        break

if main_start is not None:
    func_source = '\n'.join(lines[:main_start])
else:
    func_source = source

exec(func_source, exec_globals)

# Get the needed functions
run = exec_globals['run']
Transformer = exec_globals['Transformer']

# Run just x_div_y with seed 0
out_dir = "/home/chenhui/fabscore/aiscientist_papers/20240804_065223_invariance_learning_grokking/fabscore_claude/workspace"
dataset = "x_div_y"
seed_offset = 0

print(f"Running {dataset} with seed offset {seed_offset}")
print("This verifies the learning dynamics for Figure 2(a): Modular Division")

final_info, train_info, val_info, invariance_scores = run(out_dir, dataset, seed_offset)

print(f"\n=== Results for {dataset}, seed {seed_offset} ===")
print(f"Final train loss: {final_info['final_train_loss']:.6f}")
print(f"Final val loss: {final_info['final_val_loss']:.6f}")
print(f"Final train acc: {final_info['final_train_acc']:.4f}")
print(f"Final val acc: {final_info['final_val_acc']:.4f}")
print(f"Grokking point (step when val_acc >= 99%): {final_info['grokking_point']}")
print(f"Invariance point: {final_info['invariance_point']}")

# Sample of training dynamics
print(f"\nFirst 5 training steps logged:")
for info in train_info[:5]:
    print(f"  step={info['step']}: train_loss={info['train_loss']:.4f}, train_acc={info['train_accuracy']:.4f}")

print(f"\nFirst 5 validation steps logged:")
for info in val_info[:5]:
    print(f"  step={info['step']}: val_loss={info['val_loss']:.4f}, val_acc={info['val_accuracy']:.4f}")

print(f"\nFirst 5 invariance scores:")
for info in invariance_scores[:5]:
    print(f"  step={info['step']}: invariance_score={info['score']:.4f}")

# Find some key dynamics
# Training loss: should decrease early
# Val accuracy: should jump late (grokking)
# Invariance score: should be 1.0 throughout (for x_div_y)

# Save results
results = {
    "final_info": final_info,
    "train_info": train_info[:20],  # First 20 steps
    "val_info": val_info[:20],
    "invariance_scores": invariance_scores[:20],
    "grokking_step_val_info": [v for v in val_info if v.get('val_accuracy', 0) >= 0.99][:5],
}

with open(os.path.join(out_dir, "claim2_verification_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to claim2_verification_results.json")

# Plot the figure to verify Figure 2(a) content
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

steps_train = [info['step'] for info in train_info]
train_losses = [info['train_loss'] for info in train_info]
train_accs = [info['train_accuracy'] for info in train_info]

steps_val = [info['step'] for info in val_info]
val_losses = [info['val_loss'] for info in val_info]
val_accs = [info['val_accuracy'] for info in val_info]

steps_inv = [info['step'] for info in invariance_scores]
inv_scores = [info['score'] for info in invariance_scores]

axes[0].plot(steps_train, train_losses, 'b-', label='Train Loss')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Training Loss (x_div_y, seed 0)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(steps_val, val_accs, 'g-', label='Val Accuracy')
axes[1].set_ylabel('Validation Accuracy')
axes[1].set_title('Validation Accuracy (x_div_y, seed 0)')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(steps_inv, inv_scores, 'r-', label='Invariance Score')
axes[2].set_ylabel('Invariance Score')
axes[2].set_xlabel('Training Steps')
axes[2].set_title('Invariance Score (x_div_y, seed 0)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "claim2_figure2a_verification.png"))
print(f"Figure saved to claim2_figure2a_verification.png")
