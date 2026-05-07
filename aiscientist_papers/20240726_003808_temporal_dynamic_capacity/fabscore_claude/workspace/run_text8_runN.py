"""
Run a single run script for text8.
Usage:
  CUDA_VISIBLE_DEVICES=X python fabscore_claude/workspace/run_text8_runN.py --run RUN_NAME

E.g.:
  CUDA_VISIBLE_DEVICES=0 python fabscore_claude/workspace/run_text8_runN.py --run run_1
"""
import sys
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True, help='Run name, e.g. run_2')
args_local = parser.parse_args()

run_name = args_local.run

# Patch sys.argv before importing run scripts
sys.argv = [f'{run_name}.py', '--out_dir', 'tmp_workspace']

repo_root = '/home/chenhui/fabscore/aiscientist_papers/20240726_003808_temporal_dynamic_capacity'
workspace_dir = os.path.join(repo_root, 'fabscore_claude', 'workspace')

os.chdir(repo_root)
sys.path.insert(0, repo_root)

out_dir = os.path.join(workspace_dir, f'{run_name}_text8')
os.makedirs(out_dir, exist_ok=True)

print(f"Running {run_name} for text8...")
module = __import__(run_name)

import time
t0 = time.time()
final_info, train_info, val_info = module.train('text8', out_dir, 0)
elapsed = time.time() - t0

run_all_results = {
    'text8_0_final_info': final_info,
    'text8_0_train_info': train_info,
    'text8_0_val_info': val_info,
}

npy_path = os.path.join(out_dir, 'all_results.npy')
np.save(npy_path, run_all_results)

print(f"\nDone in {elapsed:.1f}s")
print(f"Final train loss: {final_info['final_train_loss']:.4f}")
print(f"Best val loss: {final_info['best_val_loss']:.4f}")
print(f"Eval points: {len(val_info)}")
print(f"Saved to: {npy_path}")
