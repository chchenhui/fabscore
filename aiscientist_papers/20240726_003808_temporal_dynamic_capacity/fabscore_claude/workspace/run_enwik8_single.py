"""
Run only run_1 (Basic Gating Mechanism) for enwik8 with full 100k iters.
This verifies Figure 2(a) training code works for enwik8.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=1 python fabscore_claude/workspace/run_enwik8_single.py
"""
import sys
import os
import numpy as np

sys.argv = ['run_enwik8_single.py', '--out_dir', 'tmp_workspace']

repo_root = '/home/chenhui/fabscore/aiscientist_papers/20240726_003808_temporal_dynamic_capacity'
workspace_dir = os.path.join(repo_root, 'fabscore_claude', 'workspace')

os.chdir(repo_root)
sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")
print(f"Workspace: {workspace_dir}")

out_dir = os.path.join(workspace_dir, 'run_1_enwik8')
os.makedirs(out_dir, exist_ok=True)

# Run run_1 for enwik8
import run_1 as m

print("Starting run_1 enwik8 training (100k iters, ~14 min)...")
import time
t0 = time.time()
final_info, train_info, val_info = m.train('enwik8', out_dir, 0)
elapsed = time.time() - t0

run_all_results = {
    'enwik8_0_final_info': final_info,
    'enwik8_0_train_info': train_info,
    'enwik8_0_val_info': val_info,
}

npy_path = os.path.join(out_dir, 'all_results.npy')
np.save(npy_path, run_all_results)

print(f"\nDone in {elapsed:.1f}s")
print(f"Final train loss: {final_info['final_train_loss']:.4f}")
print(f"Best val loss: {final_info['best_val_loss']:.4f}")
print(f"Eval points: {len(val_info)}")
print(f"Saved to: {npy_path}")
