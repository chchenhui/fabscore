"""
Verification script for Claim 33: Figure 3(a) - Training loss on text8 dataset.
Runs experiment.py (baseline) and run_1.py (MAT) on text8 for a small number
of iterations (3000) to confirm training curves are produced.
"""
import sys
import os
import numpy as np
import json
import inspect
import textwrap

REPO_DIR = "/home/chenhui/fabscore/aiscientist_papers/20240725_182830_memory_augmentation"
WORKSPACE = "/home/chenhui/fabscore/aiscientist_papers/20240725_182830_memory_augmentation/fabscore_claude/workspace"

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

print("=" * 60)
print("Step 1: Loading experiment.py (Baseline)")
print("=" * 60)

sys.argv = ['experiment.py', '--out_dir', os.path.join(WORKSPACE, 'run_0_text8_test')]
import experiment as exp0

# Patch to reduce max_iters for text8 to 3000
src = inspect.getsource(exp0.train)
src_patched = src.replace(
    "max_iters = 5000 if dataset == \"shakespeare_char\" else 100000",
    "max_iters = 5000 if dataset == \"shakespeare_char\" else 3000"
)
src_patched = textwrap.dedent(src_patched)
ns = {k: v for k, v in vars(exp0).items()}
exec(compile(src_patched, "<patched_train>", "exec"), ns)
patched_train = ns['train']

baseline_out = os.path.join(WORKSPACE, 'run_0_text8_test')
os.makedirs(baseline_out, exist_ok=True)

print("Running baseline (experiment.py) on text8 (3000 iters)...")
try:
    final_info, train_info, val_info = patched_train("text8", baseline_out, 0)
    print(f"Baseline text8 final_train_loss: {final_info['final_train_loss']:.6f}")
    print(f"Baseline text8 best_val_loss: {final_info['best_val_loss']:.6f}")
    print(f"Number of val log points: {len(val_info)}")
    if len(val_info) > 0:
        print(f"First val log: {val_info[0]}")
        print(f"Last val log: {val_info[-1]}")

    all_results_0 = {
        "text8_0_final_info": final_info,
        "text8_0_train_info": train_info,
        "text8_0_val_info": val_info,
    }
    npy_path = os.path.join(baseline_out, "all_results_partial.npy")
    np.save(npy_path, all_results_0)
    print(f"Saved partial all_results to {npy_path}")
    baseline_success = True
    baseline_val_info = val_info
    baseline_final = final_info
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
    baseline_success = False
    baseline_val_info = []
    baseline_final = {}

print("\n" + "=" * 60)
print("Step 2: Running MAT (run_1.py) on text8")
print("=" * 60)

mat_out = os.path.join(WORKSPACE, 'run_1_text8_test')
os.makedirs(mat_out, exist_ok=True)
sys.argv = ['run_1.py', '--out_dir', mat_out]

import run_1 as r1
src_r1 = inspect.getsource(r1.train)
src_r1_patched = src_r1.replace(
    "max_iters = 5000 if dataset == \"shakespeare_char\" else 100000",
    "max_iters = 5000 if dataset == \"shakespeare_char\" else 3000"
)
src_r1_patched = textwrap.dedent(src_r1_patched)
ns_r1 = {k: v for k, v in vars(r1).items()}
exec(compile(src_r1_patched, "<patched_train_r1>", "exec"), ns_r1)
patched_train_r1 = ns_r1['train']

print("Running MAT (run_1.py) on text8 (3000 iters)...")
try:
    final_info_r1, train_info_r1, val_info_r1 = patched_train_r1("text8", mat_out, 0)
    print(f"MAT text8 final_train_loss: {final_info_r1['final_train_loss']:.6f}")
    print(f"MAT text8 best_val_loss: {final_info_r1['best_val_loss']:.6f}")
    print(f"Number of val log points: {len(val_info_r1)}")
    if len(val_info_r1) > 0:
        print(f"First val log: {val_info_r1[0]}")
        print(f"Last val log: {val_info_r1[-1]}")

    all_results_r1 = {
        "text8_0_final_info": final_info_r1,
        "text8_0_train_info": train_info_r1,
        "text8_0_val_info": val_info_r1,
    }
    npy_path_r1 = os.path.join(mat_out, "all_results_partial.npy")
    np.save(npy_path_r1, all_results_r1)
    print(f"Saved partial all_results to {npy_path_r1}")
    mat_success = True
    mat_val_info = val_info_r1
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
    mat_success = False
    mat_val_info = []

print("\n" + "=" * 60)
print("Summary for Figure 3(a) verification (text8 training loss):")
print("=" * 60)
print(f"Baseline text8 training - Success: {baseline_success}")
print(f"MAT text8 training - Success: {mat_success}")

if baseline_success and len(baseline_val_info) > 0:
    print("\nBaseline training loss curve (from val_info):")
    for entry in baseline_val_info:
        print(f"  iter={entry['iter']}: train_loss={entry['train/loss']:.4f}, val_loss={entry['val/loss']:.4f}")
    train_losses = [e['train/loss'] for e in baseline_val_info]
    print(f"\nFirst train_loss: {train_losses[0]:.4f}, Last train_loss: {train_losses[-1]:.4f}")
    print(f"Training loss decreasing: {train_losses[0] > train_losses[-1]}")

if mat_success and len(mat_val_info) > 0:
    print("\nMAT training loss curve (from val_info):")
    for entry in mat_val_info:
        print(f"  iter={entry['iter']}: train_loss={entry['train/loss']:.4f}, val_loss={entry['val/loss']:.4f}")
    train_losses_mat = [e['train/loss'] for e in mat_val_info]
    print(f"\nFirst train_loss: {train_losses_mat[0]:.4f}, Last train_loss: {train_losses_mat[-1]:.4f}")
    print(f"Training loss decreasing: {train_losses_mat[0] > train_losses_mat[-1]}")

# Compare with existing final_info_text8_0.json
existing_text8 = json.load(open(os.path.join(REPO_DIR, "run_0/final_info_text8_0.json")))
print(f"\nExisting run_0 text8 final_train_loss: {existing_text8['final_train_loss']:.4f}")
print(f"Existing run_0 text8 best_val_loss: {existing_text8['best_val_loss']:.4f}")

print("\nVerification complete!")
