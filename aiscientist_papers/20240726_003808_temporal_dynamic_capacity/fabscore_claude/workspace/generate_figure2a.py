"""
Generate Figure 2(a): Training Loss Across Runs for enwik8 Dataset.
Uses all_results.npy files from workspace/run_X_enwik8/ directories.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

workspace_dir = '/home/chenhui/fabscore/aiscientist_papers/20240726_003808_temporal_dynamic_capacity/fabscore_claude/workspace'

# Labels from plot.py
labels = {
    "run_1": "Basic Gating Mechanism",
    "run_2": "Enhanced Gating Mechanism",
    "run_3": "Advanced Gating Mechanism with Derivatives",
    "run_4": "Complex Gating Mechanism with Second Derivatives",
    "run_5": "Dynamic Neuron Adjustment",
}

# Colors from plot.py (tab20 cmap, 6 runs, we're missing run_0=Baseline)
all_runs_for_color = ["run_0", "run_1", "run_2", "run_3", "run_4", "run_5"]
cmap = plt.get_cmap('tab20')
colors = {r: mcolors.rgb2hex(cmap(i)) for i, r in enumerate(
    np.linspace(0, 1, len(all_runs_for_color)), start=0
)}
colors = {}
for i, r in enumerate(all_runs_for_color):
    colors[r] = mcolors.rgb2hex(cmap(np.linspace(0, 1, len(all_runs_for_color))[i]))

dataset = 'enwik8'
results_info = {}

for run_name in ["run_1", "run_2", "run_3", "run_4", "run_5"]:
    npy_path = os.path.join(workspace_dir, f'{run_name}_enwik8', 'all_results.npy')
    if not os.path.exists(npy_path):
        print(f"Missing: {npy_path}")
        continue

    results_dict = np.load(npy_path, allow_pickle=True).item()

    val_losses = []
    train_losses = []
    iters = None
    for k in results_dict.keys():
        if dataset in k and "val_info" in k:
            iters = [info["iter"] for info in results_dict[k]]
            val_losses.append([info["val/loss"] for info in results_dict[k]])
            train_losses.append([info["train/loss"] for info in results_dict[k]])

    if train_losses:
        mean_train = np.mean(train_losses, axis=0)
        stderr_train = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))
        results_info[run_name] = {
            "iters": iters,
            "train_loss": mean_train,
            "train_loss_sterr": stderr_train,
        }
        print(f"{run_name}: {len(iters)} eval points, final train loss = {mean_train[-1]:.4f}")

# Plot Figure 2(a)
plt.figure(figsize=(10, 6))
for run_name in ["run_1", "run_2", "run_3", "run_4", "run_5"]:
    if run_name not in results_info:
        continue
    info = results_info[run_name]
    iters = info["iters"]
    mean = info["train_loss"]
    sterr = info["train_loss_sterr"]
    color = colors[run_name]
    plt.plot(iters, mean, label=labels[run_name], color=color)
    plt.fill_between(iters, mean - sterr, mean + sterr, color=color, alpha=0.2)

plt.title(f"Training Loss Across Runs for enwik8 Dataset")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
out_path = os.path.join(workspace_dir, 'figure2a_reproduced.png')
plt.savefig(out_path)
plt.close()
print(f"\nFigure saved to: {out_path}")
