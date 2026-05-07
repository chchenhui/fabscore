"""
Generate reproduced Figure 3(a): Training Loss Across Runs for text8 Dataset
Uses all_results.npy from each run's text8 training in the workspace.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

repo_root = '/home/chenhui/fabscore/aiscientist_papers/20240726_003808_temporal_dynamic_capacity'
workspace_dir = os.path.join(repo_root, 'fabscore_claude', 'workspace')

labels = {
    "run_1": "Basic Gating Mechanism",
    "run_2": "Enhanced Gating Mechanism",
    "run_3": "Advanced Gating Mechanism with Derivatives",
    "run_4": "Complex Gating Mechanism with Second Derivatives",
    "run_5": "Dynamic Neuron Adjustment",
}

def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

colors_full = generate_color_palette(6)
# run_1 to run_5 map to indices 1-5 (run_0 baseline would be index 0)
run_colors = {f"run_{i}": colors_full[i] for i in range(1, 6)}

summary = {}
plt.figure(figsize=(10, 6))

for run_name in ["run_1", "run_2", "run_3", "run_4", "run_5"]:
    npy_path = os.path.join(workspace_dir, f'{run_name}_text8', 'all_results.npy')
    d = np.load(npy_path, allow_pickle=True).item()

    val_info_key = 'text8_0_val_info'
    val_info = d[val_info_key]

    iters = [info['iter'] for info in val_info]
    train_losses = [info['train/loss'] for info in val_info]
    final_train = d['text8_0_final_info']['final_train_loss']
    best_val = d['text8_0_final_info']['best_val_loss']

    summary[run_name] = {
        'final_train_loss': final_train,
        'best_val_loss': best_val,
        'eval_points': len(val_info),
        'iters_range': [iters[0], iters[-1]],
    }

    plt.plot(iters, train_losses, label=labels[run_name], color=run_colors[run_name])

plt.title("Training Loss Across Runs for text8 Dataset")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()

fig_path = os.path.join(workspace_dir, 'figure3a_reproduced.png')
plt.savefig(fig_path)
plt.close()

print(f"Figure saved to: {fig_path}")
print("\nSummary:")
for run_name, info in summary.items():
    print(f"  {run_name}: final_train_loss={info['final_train_loss']:.4f}, best_val_loss={info['best_val_loss']:.4f}, eval_pts={info['eval_points']}")

# Save summary JSON
summary_path = os.path.join(workspace_dir, 'text8_train_loss_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {summary_path}")

# Reference values from original final_info files
print("\nReference values from original final_info files:")
ref_vals = {
    'run_0 (Baseline)': 1.0013301372528076,
    'run_1 (Basic Gating)': 0.9982973337173462,
    'run_2 (Enhanced Gating)': 1.0034825801849365,
    'run_3 (Advanced+Deriv)': 1.0013269186019897,
    'run_4 (Complex+2ndDeriv)': 0.9943627119064331,
    'run_5 (Dynamic Neuron Adj)': 0.9567222595214844,
}
for k, v in ref_vals.items():
    print(f"  {k}: {v:.4f}")
