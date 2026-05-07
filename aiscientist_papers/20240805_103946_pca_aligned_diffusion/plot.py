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

# CREATE LEGEND -- FILL IN RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
    "run_1": "PCA-Aligned",
    "run_2": "Controlled Gen",
    "run_3": "Cosine Schedule",
    "run_4": "Expanded Weights",
}

# Use the run key as the default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run


# CREATE PLOTS

# Create a programmatic color palette
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')  # You can change 'tab20' to other colormaps like 'Set1', 'Set2', 'Set3', etc.
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


# Get the list of runs and generate the color palette
runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Line plot of training loss for each dataset across the runs with labels
fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    for i, run in enumerate(runs):
        mean = train_info[run][dataset]["train_losses"]
        mean = smooth(mean, window_len=25)
        axs[row, col].plot(mean, label=labels[run], color=colors[i])
        axs[row, col].set_title(dataset)
        axs[row, col].legend()
        axs[row, col].set_xlabel("Training Step")
        axs[row, col].set_ylabel("Loss")

plt.tight_layout()
plt.savefig("train_loss.png")
plt.show()

# Plot 2: Visualize controlled generated samples for the last run
last_run = runs[-1]
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    controlled_samples = train_info[last_run][dataset]["controlled_samples"]
    sample_stats = train_info[last_run][dataset]["sample_stats"]
    
    for k, (samples, stats) in enumerate(zip(controlled_samples, sample_stats)):
        axs[row, col].scatter(samples[:, 0], samples[:, 1], alpha=0.2, label=f'Weights: {stats["weights"]}')
    
    axs[row, col].set_title(dataset)
    axs[row, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[row, col].set_xlabel("X")
    axs[row, col].set_ylabel("Y")

plt.tight_layout()
plt.savefig("controlled_generated_samples.png", bbox_inches='tight')
plt.show()

# Plot 3: Visualize sample statistics
fig, axs = plt.subplots(2, 2, figsize=(20, 16))

for j, dataset in enumerate(datasets):
    row = j // 2
    col = j % 2
    sample_stats = train_info[last_run][dataset]["sample_stats"]
    
    weights = [stats["weights"] for stats in sample_stats]
    means = [stats["mean"] for stats in sample_stats]
    stds = [stats["std"] for stats in sample_stats]
    entropies = [stats["entropy"] for stats in sample_stats]
    correlations = [stats["correlation"] for stats in sample_stats]
    
    x = range(len(weights))
    axs[row, col].plot(x, [m[0] for m in means], 'bo-', label='Mean (1st comp)')
    axs[row, col].plot(x, [m[1] for m in means], 'b^-', label='Mean (2nd comp)')
    axs[row, col].plot(x, [s[0] for s in stds], 'ro-', label='Std Dev (1st comp)')
    axs[row, col].plot(x, [s[1] for s in stds], 'r^-', label='Std Dev (2nd comp)')
    axs[row, col].plot(x, entropies, 'go-', label='Entropy')
    axs[row, col].plot(x, correlations, 'mo-', label='Correlation')
    
    axs[row, col].set_title(dataset)
    axs[row, col].set_xlabel("Weight Configuration")
    axs[row, col].set_ylabel("Statistic Value")
    axs[row, col].legend(loc='upper left', fontsize='small')
    axs[row, col].set_xticks(x)
    axs[row, col].set_xticklabels([f"{w[0]:.1f}, {w[1]:.1f}" for w in weights], rotation=45, ha='right', fontsize='x-small')

plt.tight_layout()
plt.savefig("sample_statistics.png", bbox_inches='tight', dpi=150)
plt.close()

# Plot 4: Compare KL divergence across runs
kl_divergences = {run: [final_results[run][dataset]['kl_divergence'] for dataset in datasets] for run in runs}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(datasets))
width = 0.15
multiplier = 0

for run, kl_values in kl_divergences.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, kl_values, width, label=labels[run])
    ax.bar_label(rects, padding=3, rotation=90)
    multiplier += 1

ax.set_ylabel('KL Divergence')
ax.set_title('KL Divergence Comparison Across Runs')
ax.set_xticks(x + width, datasets)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, max(max(kl_values) for kl_values in kl_divergences.values()) * 1.2)

plt.tight_layout()
plt.savefig("kl_divergence_comparison.png")
plt.show()

# Plot 5: Training and inference time comparison
training_times = {run: [final_results[run][dataset]['training_time'] for dataset in datasets] for run in runs}
inference_times = {run: [final_results[run][dataset]['inference_time'] for dataset in datasets] for run in runs}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

x = np.arange(len(datasets))
width = 0.15
multiplier = 0

for run, times in training_times.items():
    offset = width * multiplier
    rects = ax1.bar(x + offset, times, width, label=labels[run])
    ax1.bar_label(rects, padding=3, rotation=90)
    multiplier += 1

ax1.set_ylabel('Training Time (s)')
ax1.set_title('Training Time Comparison Across Runs')
ax1.set_xticks(x + width, datasets)
ax1.legend(loc='upper left', ncols=3)

multiplier = 0
for run, times in inference_times.items():
    offset = width * multiplier
    rects = ax2.bar(x + offset, times, width, label=labels[run])
    ax2.bar_label(rects, padding=3, rotation=90)
    multiplier += 1

ax2.set_ylabel('Inference Time (s)')
ax2.set_title('Inference Time Comparison Across Runs')
ax2.set_xticks(x + width, datasets)
ax2.legend(loc='upper left', ncols=3)

plt.tight_layout()
plt.savefig("time_comparison.png")
plt.show()
