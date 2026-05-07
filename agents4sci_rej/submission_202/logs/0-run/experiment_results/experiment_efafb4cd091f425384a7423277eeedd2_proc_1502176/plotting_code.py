import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Baseline
act_root = experiment_data.get("activation_fn_tuning", {})
ds = "mnist_claims"
activation_candidates = ["relu", "leakyrelu", "elu", "gelu"]

# 1. Accuracy curves for each activation function
for act in activation_candidates:
    try:
        data = act_root[ds][act]
        epochs = data["epochs"]
        train_acc = data["metrics"]["train_acc"]
        val_acc = data["metrics"]["val_acc"]
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
        plt.plot(epochs, val_acc, marker="s", label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"MNIST+Claims: Accuracy Curves\nVision Activation: {act}")
        plt.legend()
        plt.tight_layout()
        pth = os.path.join(working_dir, f"{ds}_accuracy_curve_{act}.png")
        plt.savefig(pth)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {act}: {e}")
        plt.close()

# 2. Loss curves for each activation function
for act in activation_candidates:
    try:
        data = act_root[ds][act]
        epochs = data["epochs"]
        train_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_loss, marker="o", label="Train Loss")
        plt.plot(epochs, val_loss, marker="s", label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"MNIST+Claims: Loss Curves\nVision Activation: {act}")
        plt.legend()
        plt.tight_layout()
        pth = os.path.join(working_dir, f"{ds}_loss_curve_{act}.png")
        plt.savefig(pth)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {act}: {e}")
        plt.close()

# 3. Bar plot of final validation accuracy for all activations
try:
    plt.figure(figsize=(8, 5))
    final_accs = []
    for act in activation_candidates:
        val_accs = act_root[ds][act]["metrics"]["val_acc"]
        final_accs.append(val_accs[-1] if len(val_accs) > 0 else 0)
    plt.bar(activation_candidates, final_accs, color=["C0", "C1", "C2", "C3"])
    plt.ylabel("Final Validation Accuracy")
    plt.xlabel("Activation Function")
    plt.ylim(0, 1)
    plt.title("MNIST+Claims: Final Validation Accuracy by Vision Activation")
    for i, v in enumerate(final_accs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    bar_path = os.path.join(working_dir, f"{ds}_final_val_acc_barplot.png")
    plt.tight_layout()
    plt.savefig(bar_path)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()
