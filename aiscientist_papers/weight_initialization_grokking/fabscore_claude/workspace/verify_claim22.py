"""
Verify Claim 22: Figure 2 - Training and Validation Loss for x_minus_y task
Uses freshly generated all_results.npy files from run_0 through run_4.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

workspace = os.path.dirname(os.path.abspath(__file__))
dataset = "x_minus_y"

labels = {
    "run_0": "Baseline",
    "run_1": "Xavier (Glorot)",
    "run_2": "He",
    "run_3": "Orthogonal",
    "run_4": "Kaiming Normal",
}

results_info = {}
for run, label in labels.items():
    npy_path = os.path.join(workspace, run, "all_results.npy")
    if not os.path.exists(npy_path):
        print(f"WARNING: {npy_path} not found, skipping {run}")
        continue
    results_dict = np.load(npy_path, allow_pickle=True).item()
    val_losses, train_losses, steps = [], [], None
    for k in results_dict.keys():
        if dataset in k and "val_info" in k:
            steps = [info["step"] for info in results_dict[k]]
            val_losses.append([info["val_loss"] for info in results_dict[k]])
        if dataset in k and "train_info" in k:
            train_losses.append([info["train_loss"] for info in results_dict[k]])

    if steps is None:
        print(f"WARNING: No data found for {dataset} in {run}")
        continue

    results_info[run] = {
        "step": steps,
        "val_loss": np.mean(val_losses, axis=0),
        "train_loss": np.mean(train_losses, axis=0),
        "val_loss_sterr": np.std(val_losses, axis=0) / np.sqrt(len(val_losses)),
        "train_loss_sterr": np.std(train_losses, axis=0) / np.sqrt(len(train_losses)),
    }
    final_val_loss = results_info[run]["val_loss"][-1]
    final_train_loss = results_info[run]["train_loss"][-1]
    print(f"{run} ({label}): final_val_loss={final_val_loss:.4f}, final_train_loss={final_train_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
for run, info in results_info.items():
    plt.plot(info["step"], info["train_loss"], label=labels[run])
    plt.fill_between(info["step"],
                     info["train_loss"] - info["train_loss_sterr"],
                     info["train_loss"] + info["train_loss_sterr"], alpha=0.2)
plt.title(f"Training Loss Across Runs for {dataset} Dataset")
plt.xlabel("Update Steps")
plt.ylabel("Training Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
out_path = os.path.join(workspace, f"train_loss_{dataset}_claim22.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")

# Plot validation loss
plt.figure(figsize=(10, 6))
for run, info in results_info.items():
    plt.plot(info["step"], info["val_loss"], label=labels[run])
    plt.fill_between(info["step"],
                     info["val_loss"] - info["val_loss_sterr"],
                     info["val_loss"] + info["val_loss_sterr"], alpha=0.2)
plt.title(f"Validation Loss Across Runs for {dataset} Dataset")
plt.xlabel("Update Steps")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
out_path = os.path.join(workspace, f"val_loss_{dataset}_claim22.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")

print("\nDone. Figures generated successfully.")
print(f"PNG files in workspace: {[f for f in os.listdir(workspace) if f.endswith('.png')]}")
