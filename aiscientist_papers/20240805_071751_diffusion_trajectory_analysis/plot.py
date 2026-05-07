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
labels = {
    "run_0": "Baseline",
    "run_1": "Denoising + Quadrant + Distance",
    "run_2": "Denoising + Quadrant + Distance + Angle",
    "run_3": "Timestep Analysis",
    "run_4": "Visualization",
    "run_5": "Final Analysis"
}

# Use only the runs specified in the labels dictionary
runs = list(labels.keys())


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
        losses = train_info[run][dataset]["train_losses"]
        if isinstance(losses[0], dict):
            total_losses = [loss['loss'] for loss in losses]
        else:
            total_losses = losses
        mean = smooth(total_losses, window_len=25)
        axs[row, col].plot(mean, label=labels[run], color=colors[i])
        axs[row, col].set_title(dataset)
        axs[row, col].legend()
        axs[row, col].set_xlabel("Training Step")
        axs[row, col].set_ylabel("Loss")

plt.tight_layout()
plt.savefig("train_loss.png")
plt.show()

# Plot 2: Visualize generated samples
# If there is more than 1 run, these are added as extra rows.
num_runs = len(runs)
fig, axs = plt.subplots(num_runs, 4, figsize=(14, 3 * num_runs))

for i, run in enumerate(runs):
    for j, dataset in enumerate(datasets):
        images = train_info[run][dataset]["images"]
        if num_runs == 1:
            axs[j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
            axs[j].set_title(dataset)
        else:
            axs[i, j].scatter(images[:, 0], images[:, 1], alpha=0.2, color=colors[i])
            axs[i, j].set_title(dataset)
    if num_runs == 1:
        axs[0].set_ylabel(labels[run])
    else:
        axs[i, 0].set_ylabel(labels[run])

plt.tight_layout()
plt.savefig("generated_images.png")
plt.show()

# Plot 3: Visualize prediction accuracies across timesteps
fig, axs = plt.subplots(4, 3, figsize=(15, 20))
fig.suptitle("Prediction Accuracies Across Timesteps", fontsize=16)

for i, dataset in enumerate(datasets):
    timestep_accuracies = final_results[runs[-1]][dataset]["timestep_accuracies"]
    timesteps = [acc["timestep"] for acc in timestep_accuracies]
    
    quadrant_accuracies = [acc["quadrant_accuracy"] for acc in timestep_accuracies]
    axs[i, 0].plot(timesteps, quadrant_accuracies)
    axs[i, 0].set_title(f"{dataset} - Quadrant Accuracy")
    axs[i, 0].set_xlabel("Timestep")
    axs[i, 0].set_ylabel("Accuracy")
    
    distance_accuracies = [acc["distance_accuracy"] for acc in timestep_accuracies]
    axs[i, 1].plot(timesteps, distance_accuracies)
    axs[i, 1].set_title(f"{dataset} - Distance Accuracy")
    axs[i, 1].set_xlabel("Timestep")
    axs[i, 1].set_ylabel("Accuracy")
    
    angle_accuracies = [acc["angle_accuracy"] for acc in timestep_accuracies]
    axs[i, 2].plot(timesteps, angle_accuracies)
    axs[i, 2].set_title(f"{dataset} - Angle Accuracy")
    axs[i, 2].set_xlabel("Timestep")
    axs[i, 2].set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig("prediction_accuracies.png")
plt.show()
