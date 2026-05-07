"""
Minimal wrapper to run only shakespeare_char training for all run implementations.
This script imports train() functions from each run script and runs only shakespeare_char.
It saves all_results.npy to the workspace directory for Figure 1(a) verification.

Usage: Run from the repo root directory:
  python fabscore_claude/workspace/run_shakespeare_only.py
"""
import sys
import os
import numpy as np

# Patch sys.argv to avoid argparse issues when importing run scripts
sys.argv = ['run_shakespeare_only.py', '--out_dir', 'tmp_workspace']

# Change to repo root so data paths work
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
workspace_dir = os.path.dirname(os.path.abspath(__file__))

os.chdir(repo_root)
sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")
print(f"Workspace: {workspace_dir}")
print(f"Data path: {os.path.join('../../../data', 'shakespeare_char')}")

# We'll run each run script's train function for shakespeare_char only
run_scripts = ['run_1', 'run_2', 'run_3', 'run_4', 'run_5']
run_labels = {
    'run_1': 'Basic Gating Mechanism',
    'run_2': 'Enhanced Gating Mechanism',
    'run_3': 'Advanced Gating Mechanism with Derivatives',
    'run_4': 'Complex Gating Mechanism with Second Derivatives',
    'run_5': 'Dynamic Neuron Adjustment',
}

all_runs_results = {}
num_seeds = 3  # shakespeare_char uses 3 seeds

for run_name in run_scripts:
    print(f"\n{'='*60}")
    print(f"Running {run_name}: {run_labels[run_name]}")
    print(f"{'='*60}")

    # Import the module fresh each time
    if run_name in sys.modules:
        del sys.modules[run_name]

    module = __import__(run_name)
    train_fn = module.train

    run_all_results = {}
    out_dir = os.path.join(workspace_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    for seed_offset in range(num_seeds):
        print(f"\n  Seed {seed_offset}/{num_seeds-1}")
        try:
            final_info, train_info, val_info = train_fn('shakespeare_char', out_dir, seed_offset)
            run_all_results[f"shakespeare_char_{seed_offset}_final_info"] = final_info
            run_all_results[f"shakespeare_char_{seed_offset}_train_info"] = train_info
            run_all_results[f"shakespeare_char_{seed_offset}_val_info"] = val_info
            print(f"    Final train loss: {final_info['final_train_loss']:.4f}")
            print(f"    Best val loss: {final_info['best_val_loss']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results for this run
    npy_path = os.path.join(out_dir, 'all_results.npy')
    with open(npy_path, 'wb') as f:
        np.save(f, run_all_results)
    print(f"\n  Saved to: {npy_path}")

    all_runs_results[run_name] = run_all_results

# Save combined results summary
summary_path = os.path.join(workspace_dir, 'shakespeare_char_all_runs_summary.json')
import json
summary = {}
for run_name, results in all_runs_results.items():
    train_losses = []
    for seed_offset in range(num_seeds):
        key = f"shakespeare_char_{seed_offset}_final_info"
        if key in results:
            train_losses.append(results[key]['final_train_loss'])
    summary[run_name] = {
        'label': run_labels[run_name],
        'final_train_losses': train_losses,
        'mean_final_train_loss': float(np.mean(train_losses)) if train_losses else None,
    }

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n\nSummary saved to: {summary_path}")
print("\nFinal results:")
for run_name, data in summary.items():
    print(f"  {run_name} ({data['label']}): mean={data['mean_final_train_loss']:.4f}")
