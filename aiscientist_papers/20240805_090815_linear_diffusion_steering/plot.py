import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp
import pickle

# LOAD FINAL RESULTS:
datasets = ["circle", "dino", "line", "moons"]
folders = os.listdir("./")
final_results = {}
train_info = {}


def smooth(x, window_len=10, window='hanning'):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pickle.load(open(osp.join(folder, "all_results.pkl"), "rb"))
        train_info[folder] = all_results

# CREATE LEGEND
labels = {
    "run_0": "Baseline",
    "run_1": "Linear (0.5)",
    "run_2": "Linear (0.1)",
    "run_3": "Linear (1.0)",
    "run_4": "Circular (1.0)",
}

# Only plot the runs specified in the labels dictionary
runs = list(labels.keys())

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

# Generate the color palette for the specified runs
colors = generate_color_palette(len(runs))

# Plot 1: Visualize generated samples for each run and dataset
fig, axs = plt.subplots(len(runs), 4, figsize=(16, 4 * len(runs)))
fig.suptitle("Generated Samples", fontsize=16)

for i, run in enumerate(runs):
    for j, dataset in enumerate(datasets):
        images = train_info[run][dataset]["images"]
        axs[i, j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
        axs[i, j].set_title(f"{dataset} - {labels[run]}")
        if run == "run_4":  # Circular guidance
            theta = np.linspace(0, 2*np.pi, 100)
            axs[i, j].plot(np.cos(theta), np.sin(theta), 'r--', label='Circular Path')
        else:
            axs[i, j].plot([-1, 1], [-1, 1], 'r--', label='Linear Path')
        axs[i, j].legend()
        axs[i, j].set_xlim(-2, 2)
        axs[i, j].set_ylim(-2, 2)
    axs[i, 0].set_ylabel(labels[run])

plt.tight_layout()
plt.savefig("generated_samples.png")
plt.close()

# Plot 2: Compare KL divergence across runs and datasets
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.15
index = np.arange(len(datasets))

for i, run in enumerate(runs):
    kl_divergences = [final_results[run][dataset]['means']['kl_divergence'] for dataset in datasets]
    ax.bar(index + i*bar_width, kl_divergences, bar_width, label=labels[run], color=colors[i])

ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence Comparison')
ax.set_xticks(index + bar_width * (len(runs) - 1) / 2)
ax.set_xticklabels(datasets)
ax.legend()

plt.tight_layout()
plt.savefig("kl_divergence_comparison.png")
plt.close()

# Plot 3: Compare path adherence across runs and datasets
fig, ax = plt.subplots(figsize=(12, 6))

for i, run in enumerate(runs):
    if 'path_adherence' in final_results[run][datasets[0]]['means']:
        path_adherences = [final_results[run][dataset]['means']['path_adherence'] for dataset in datasets]
        ax.bar(index + i*bar_width, path_adherences, bar_width, label=labels[run], color=colors[i])

ax.set_ylabel('Path Adherence (MSE)')
ax.set_title('Path Adherence Comparison')
ax.set_xticks(index + bar_width * (len(runs) - 1) / 2)
ax.set_xticklabels(datasets)
ax.legend()

plt.tight_layout()
plt.savefig("path_adherence_comparison.png")
plt.close()

# Plot 4: Trade-off between KL divergence and path adherence
fig, ax = plt.subplots(figsize=(10, 8))

for i, run in enumerate(runs):
    if 'path_adherence' in final_results[run][datasets[0]]['means']:
        kl_divergences = [final_results[run][dataset]['means']['kl_divergence'] for dataset in datasets]
        path_adherences = [final_results[run][dataset]['means']['path_adherence'] for dataset in datasets]
        ax.scatter(kl_divergences, path_adherences, label=labels[run], color=colors[i], s=50)

ax.set_xlabel('KL Divergence')
ax.set_ylabel('Path Adherence (MSE)')
ax.set_title('Trade-off: KL Divergence vs Path Adherence')
ax.legend()

plt.tight_layout()
plt.savefig("kl_path_tradeoff.png")
plt.close()
