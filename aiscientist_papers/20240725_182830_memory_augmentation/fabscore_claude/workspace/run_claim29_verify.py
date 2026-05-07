"""
Verification script for Claim 29: Figure 1(a) - Training loss on shakespeare_char.
Imports train function from experiment.py (baseline/run_0) and run_1.py (MAT),
runs just shakespeare_char for 1 seed each to verify training curves are produced.
"""
import sys
import os
import numpy as np
import json

REPO_DIR = "/home/chenhui/fabscore/aiscientist_papers/20240725_182830_memory_augmentation"
WORKSPACE = "/home/chenhui/fabscore/aiscientist_papers/20240725_182830_memory_augmentation/fabscore_claude/workspace"

sys.path.insert(0, REPO_DIR)

print("=" * 60)
print("Step 1: Running Baseline (experiment.py) on shakespeare_char")
print("=" * 60)

# Patch sys.argv for argparse at module level
sys.argv = ['experiment.py', '--out_dir', os.path.join(WORKSPACE, 'run_0_test')]

import experiment as exp0
baseline_out = os.path.join(WORKSPACE, 'run_0_test')
os.makedirs(baseline_out, exist_ok=True)

print("Running baseline train on shakespeare_char (seed 0)...")
final_info_0, train_info_0, val_info_0 = exp0.train("shakespeare_char", baseline_out, 0)
print(f"Baseline final_train_loss: {final_info_0['final_train_loss']:.6f}")
print(f"Baseline best_val_loss: {final_info_0['best_val_loss']:.6f}")
print(f"Number of training log points: {len(train_info_0)}")
print(f"Number of val log points: {len(val_info_0)}")
if len(val_info_0) > 0:
    print(f"First val log: {val_info_0[0]}")
    print(f"Last val log: {val_info_0[-1]}")

# Save partial all_results for run_0
all_results_0 = {
    "shakespeare_char_0_final_info": final_info_0,
    "shakespeare_char_0_train_info": train_info_0,
    "shakespeare_char_0_val_info": val_info_0,
}
npy_path_0 = os.path.join(baseline_out, "all_results_partial.npy")
np.save(npy_path_0, all_results_0)
print(f"Saved partial all_results to {npy_path_0}")

print("\n" + "=" * 60)
print("Step 2: Running MAT (run_1.py) on shakespeare_char")
print("=" * 60)

# Need to reload with fresh sys.argv for run_1's argparse
mat_out = os.path.join(WORKSPACE, 'run_1_test')
os.makedirs(mat_out, exist_ok=True)
sys.argv = ['run_1.py', '--out_dir', mat_out]

import run_1 as r1
print("Running MAT train on shakespeare_char (seed 0)...")
final_info_1, train_info_1, val_info_1 = r1.train("shakespeare_char", mat_out, 0)
print(f"MAT final_train_loss: {final_info_1['final_train_loss']:.6f}")
print(f"MAT best_val_loss: {final_info_1['best_val_loss']:.6f}")
print(f"Number of training log points: {len(train_info_1)}")
print(f"Number of val log points: {len(val_info_1)}")
if len(val_info_1) > 0:
    print(f"First val log: {val_info_1[0]}")
    print(f"Last val log: {val_info_1[-1]}")

# Save partial all_results for run_1
all_results_1 = {
    "shakespeare_char_0_final_info": final_info_1,
    "shakespeare_char_0_train_info": train_info_1,
    "shakespeare_char_0_val_info": val_info_1,
}
npy_path_1 = os.path.join(mat_out, "all_results_partial.npy")
np.save(npy_path_1, all_results_1)
print(f"Saved partial all_results to {npy_path_1}")

print("\n" + "=" * 60)
print("Summary for Figure 1(a) verification:")
print("=" * 60)
print(f"Baseline training loss at end: {final_info_0['final_train_loss']:.4f}")
print(f"MAT training loss at end: {final_info_1['final_train_loss']:.4f}")

# Compare with existing final_info.json
existing_0 = json.load(open(os.path.join(REPO_DIR, "run_0/final_info_shakespeare_char_0.json")))
print(f"\nExisting run_0 final_train_loss: {existing_0['final_train_loss']:.4f}")

print("\nTraining loss curves (val log format for figure):")
print("Iter -> Baseline train_loss:")
for entry in val_info_0:
    print(f"  iter={entry['iter']}: train={entry['train/loss']:.4f}, val={entry['val/loss']:.4f}")

print("\nVerification complete! Training curves produced successfully.")
print("Figure 1(a) can be generated from these training curves.")
