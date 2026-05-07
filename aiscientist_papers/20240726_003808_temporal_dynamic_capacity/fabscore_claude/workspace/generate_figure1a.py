"""Generate Figure 1(a) equivalent from workspace data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

workspace = os.path.dirname(os.path.abspath(__file__))

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

runs = list(labels.keys())
colors = generate_color_palette(len(runs))

plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    npy_path = os.path.join(workspace, run, 'all_results.npy')
    data = np.load(npy_path, allow_pickle=True).item()

    val_losses = []
    train_losses = []
    iters = None
    for k in data.keys():
        if 'shakespeare_char' in k and 'val_info' in k:
            if iters is None:
                iters = [info['iter'] for info in data[k]]
            val_losses.append([info['val/loss'] for info in data[k]])
            train_losses.append([info['train/loss'] for info in data[k]])

    mean_train = np.mean(train_losses, axis=0)
    sterr_train = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))

    plt.plot(iters, mean_train, label=labels[run], color=colors[i])
    plt.fill_between(iters, mean_train - sterr_train, mean_train + sterr_train, color=colors[i], alpha=0.2)

    print(f"{run}: iters={iters[0]}..{iters[-1]}, final_train_loss={mean_train[-1]:.4f}")

plt.title("Training Loss Across Runs for shakespeare_char Dataset")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.legend(fontsize=7)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
outpath = os.path.join(workspace, 'figure1a_reproduced.png')
plt.savefig(outpath)
print(f"\nSaved figure to: {outpath}")
