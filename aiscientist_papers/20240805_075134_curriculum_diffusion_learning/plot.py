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
    "run_1": "Linear CL",
    "run_2": "Quadratic CL",
    "run_3": "Cosine Beta",
    "run_4": "Increased Capacity",
    "run_5": "Swish Activation"
}

# Only include runs that are in the labels dictionary
runs = [run for run in final_results.keys() if run in labels]


# CREATE PLOTS

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of training loss for each dataset across the runs
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig.suptitle("Training Loss Across Datasets", fontsize=16)

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    for i, run in enumerate(runs):
        mean = train_info[run][dataset]["train_losses"]
        mean = smooth(mean, window_len=25)
        axs[row, col].plot(mean, label=labels[run], color=colors[i])
        axs[row, col].set_title(dataset.capitalize())
        axs[row, col].set_xlabel("Training Step")
        axs[row, col].set_ylabel("Loss")
        axs[row, col].legend()

plt.tight_layout()
plt.savefig("train_loss.png")
plt.close()

# Plot 2: KL Divergence comparison
kl_values = {run: [final_results[run][dataset]['means']['kl_divergence'] for dataset in datasets] for run in runs}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(datasets))
width = 0.15
multiplier = 0

for run, kl_value in kl_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, kl_value, width, label=labels[run])
    ax.bar_label(rects, padding=3, rotation=90, fmt='%.2f')
    multiplier += 1

ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence Comparison Across Datasets and Runs')
ax.set_xticks(x + width, datasets)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, max([max(v) for v in kl_values.values()]) * 1.1)

plt.tight_layout()
plt.savefig("kl_divergence_comparison.png")
plt.close()

# Plot 3: Sample quality during training
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
fig.suptitle("Sample Quality (KL Divergence) During Training", fontsize=16)

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    for i, run in enumerate(runs):
        if "sample_quality_steps" in train_info[run][dataset] and "sample_quality_kl" in train_info[run][dataset]:
            steps = train_info[run][dataset]["sample_quality_steps"]
            kl = train_info[run][dataset]["sample_quality_kl"]
            axs[row, col].plot(steps, kl, label=labels[run], color=colors[i])
    axs[row, col].set_title(dataset.capitalize())
    axs[row, col].set_xlabel("Training Step")
    axs[row, col].set_ylabel("KL Divergence")
    axs[row, col].legend()

plt.tight_layout()
plt.savefig("sample_quality_during_training.png")
plt.close()

# Plot 4: Visualize generated samples
fig, axs = plt.subplots(len(runs), 4, figsize=(16, 4 * len(runs)))
fig.suptitle("Generated Samples Across Datasets and Runs", fontsize=16)

for i, run in enumerate(runs):
    for j, dataset in enumerate(datasets):
        images = train_info[run][dataset]["images"]
        axs[i, j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
        axs[i, j].set_title(f"{dataset.capitalize()} - {labels[run]}")
        axs[i, j].set_xlabel("X")
        axs[i, j].set_ylabel("Y")

plt.tight_layout()
plt.savefig("generated_samples.png")
plt.close()
