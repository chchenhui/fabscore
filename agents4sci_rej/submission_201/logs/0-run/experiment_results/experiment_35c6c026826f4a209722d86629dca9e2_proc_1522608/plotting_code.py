import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

dsnames = ["mnist", "fashion_mnist", "svhn"]

# 1. Combined plot: Validation Accuracy (across datasets)
try:
    plt.figure(figsize=(8, 6))
    for ds in dsnames:
        epochs = experiment_data[ds]["epochs"]
        val_acc = experiment_data[ds]["metrics"]["val"]
        plt.plot(epochs, val_acc, label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy across Datasets (Image-Only Ablation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_acc_across_datasets_image_only.png"))
    plt.close()
except Exception as e:
    print(f"Error creating combined validation accuracy plot: {e}")
    plt.close()

# 2. Combined plot: Validation Logical Consistency Accuracy (across datasets)
try:
    plt.figure(figsize=(8, 6))
    for ds in dsnames:
        epochs = experiment_data[ds]["epochs"]
        val_logic = experiment_data[ds]["metrics"]["val_logic"]
        plt.plot(epochs, val_logic, label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Logical Consistency Accuracy (Validation)")
    plt.title(
        "Validation Logical Consistency Accuracy across Datasets (Image-Only Ablation)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "val_logic_acc_across_datasets_image_only.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating combined logic accuracy plot: {e}")
    plt.close()

# 3. For each dataset: Validation Accuracy & Logic Accuracy + Loss curves
for ds in dsnames:
    # Validation & logic acc
    try:
        plt.figure(figsize=(8, 6))
        epochs = experiment_data[ds]["epochs"]
        val_acc = experiment_data[ds]["metrics"]["val"]
        val_logic = experiment_data[ds]["metrics"]["val_logic"]
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.plot(epochs, val_logic, label="Logical Consistency Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{ds} - Validation/Logical Accuracies per Epoch\nImage-Only Ablation"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_val_and_logic_acc_image_only.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating acc/logic plot for {ds}: {e}")
        plt.close()

    # Training/validation loss
    try:
        plt.figure(figsize=(8, 6))
        train_loss = experiment_data[ds]["losses"]["train"]
        val_loss = experiment_data[ds]["losses"]["val"]
        plt.plot(epochs, train_loss, label="Training Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(f"{ds} - Training/Validation Loss per Epoch\nImage-Only Ablation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_loss_image_only.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

# 4. Print final logical consistency accuracy
for ds in dsnames:
    try:
        logic = experiment_data[ds]["metrics"]["val_logic"][-1]
        print(f"Final Logical Consistency Accuracy (Image Only, {ds}): {logic:.4f}")
    except Exception as e:
        print(f"Could not print final logic accuracy for {ds}: {e}")
