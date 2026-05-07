"""
Verify Claim 23: Figure 3 - Training and Validation Accuracy for x_div_y task
Uses freshly generated all_results.npy files from run_0 through run_4.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

workspace = os.path.dirname(os.path.abspath(__file__))
dataset = "x_div_y"

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
    val_accs, train_accs, steps = [], [], None
    for k in results_dict.keys():
        if dataset in k and "val_info" in k:
            steps = [info["step"] for info in results_dict[k]]
            val_accs.append([info["val_accuracy"] for info in results_dict[k]])
        if dataset in k and "train_info" in k:
            train_accs.append([info["train_accuracy"] for info in results_dict[k]])

    if steps is None:
        print(f"WARNING: No data found for {dataset} in {run}")
        continue

    results_info[run] = {
        "step": steps,
        "val_acc": np.mean(val_accs, axis=0),
        "train_acc": np.mean(train_accs, axis=0),
        "val_acc_sterr": np.std(val_accs, axis=0) / np.sqrt(len(val_accs)),
        "train_acc_sterr": np.std(train_accs, axis=0) / np.sqrt(len(train_accs)),
    }
    final_val_acc = results_info[run]["val_acc"][-1]
    final_train_acc = results_info[run]["train_acc"][-1]
    print(f"{run} ({label}): final_val_acc={final_val_acc:.4f}, final_train_acc={final_train_acc:.4f}")

# Plot training accuracy
plt.figure(figsize=(10, 6))
for run, info in results_info.items():
    plt.plot(info["step"], info["train_acc"], label=labels[run])
    plt.fill_between(info["step"],
                     info["train_acc"] - info["train_acc_sterr"],
                     info["train_acc"] + info["train_acc_sterr"], alpha=0.2)
plt.title(f"Training Accuracy Across Runs for {dataset} Dataset")
plt.xlabel("Update Steps")
plt.ylabel("Training Accuracy")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
out_path = os.path.join(workspace, f"train_acc_{dataset}_claim23.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")

# Plot validation accuracy
plt.figure(figsize=(10, 6))
for run, info in results_info.items():
    plt.plot(info["step"], info["val_acc"], label=labels[run])
    plt.fill_between(info["step"],
                     info["val_acc"] - info["val_acc_sterr"],
                     info["val_acc"] + info["val_acc_sterr"], alpha=0.2)
plt.title(f"Validation Accuracy Across Runs for {dataset} Dataset")
plt.xlabel("Update Steps")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
out_path = os.path.join(workspace, f"val_acc_{dataset}_claim23.png")
plt.savefig(out_path)
plt.close()
print(f"Saved: {out_path}")

print("\nDone. Figures generated successfully.")
print(f"PNG files in workspace: {[f for f in os.listdir(workspace) if f.endswith('.png')]}")
