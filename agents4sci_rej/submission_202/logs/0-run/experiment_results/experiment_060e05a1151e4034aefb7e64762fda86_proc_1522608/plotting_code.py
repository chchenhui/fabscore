import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Try to load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

datasets = ["mnist", "fashion_mnist", "svhn"]

# Plot 1: Validation Accuracy per Epoch for all datasets
try:
    plt.figure(figsize=(8, 6))
    for ds in datasets:
        epochs = experiment_data["shuffle_claim_text"][ds]["epochs"]
        val_acc = experiment_data["shuffle_claim_text"][ds]["metrics"]["val"]
        plt.plot(epochs, val_acc, label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Validation Accuracy Across Epochs\nShuffle Claim-Text Ablation, Datasets: MNIST, FashionMNIST, SVHN"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_acc_shuffle_ablation.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val_acc plot: {e}")
    plt.close()

# Plot 2: Validation Logical Consistency Accuracy per Epoch for all datasets
try:
    plt.figure(figsize=(8, 6))
    for ds in datasets:
        epochs = experiment_data["shuffle_claim_text"][ds]["epochs"]
        val_logic = experiment_data["shuffle_claim_text"][ds]["metrics"]["val_logic"]
        plt.plot(epochs, val_logic, label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Logical Consistency Accuracy")
    plt.title(
        "Logical Consistency Accuracy Across Epochs\nShuffle Claim-Text Ablation, Datasets: MNIST, FashionMNIST, SVHN"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_logic_acc_shuffle_ablation.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val_logic plot: {e}")
    plt.close()

# Plot 3: Per-dataset overlay: Validation Accuracy and Logical Consistency over Epochs
for ds in datasets:
    try:
        plt.figure(figsize=(7, 6))
        epochs = experiment_data["shuffle_claim_text"][ds]["epochs"]
        val_acc = experiment_data["shuffle_claim_text"][ds]["metrics"]["val"]
        val_logic = experiment_data["shuffle_claim_text"][ds]["metrics"]["val_logic"]
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.plot(epochs, val_logic, label="Logical Consistency Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{ds.upper()} - Validation Accuracies per Epoch\nLeft: Val Acc, Right: Logic Acc (Shuffle Claim-Text Ablation)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_acc_shuffle_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating per-dataset plot for {ds}: {e}")
        plt.close()
