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

# Define datasets and colors
datasets = ["mnist", "fashion_mnist", "svhn"]
ds_labels = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "svhn": "SVHN"}
colors = {"mnist": "b", "fashion_mnist": "r", "svhn": "g"}

# 1. Plot accuracy curves for each dataset
for ds in datasets:
    try:
        plt.figure(figsize=(8, 6))
        epochs = experiment_data[ds]["epochs"]
        plt.plot(
            epochs,
            experiment_data[ds]["metrics"]["train_acc"],
            "--",
            alpha=0.7,
            label="Train Accuracy",
        )
        plt.plot(
            epochs,
            experiment_data[ds]["metrics"]["val_acc"],
            "-",
            label="Validation Accuracy",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_labels[ds]} - Train/Validation Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_train_val_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds}: {e}")
        plt.close()

# 2. Plot loss curves for each dataset
for ds in datasets:
    try:
        plt.figure(figsize=(8, 6))
        epochs = experiment_data[ds]["epochs"]
        plt.plot(
            epochs,
            experiment_data[ds]["losses"]["train"],
            "--",
            alpha=0.7,
            label="Train Loss",
        )
        plt.plot(
            epochs, experiment_data[ds]["losses"]["val"], "-", label="Validation Loss"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{ds_labels[ds]} - Train/Validation Loss")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_train_val_loss.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {ds}: {e}")
        plt.close()

# 3. Comparison plot: validation accuracy across datasets
try:
    plt.figure(figsize=(8, 6))
    for ds in datasets:
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["val_acc"],
            color=colors[ds],
            label=ds_labels[ds],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison\n(MNIST, Fashion-MNIST, SVHN)")
    plt.legend()
    fname = os.path.join(working_dir, "val_acc_compare_all_datasets.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating validation accuracy comparison plot: {e}")
    plt.close()

# 4. Comparison plot: logical consistency accuracy across datasets
try:
    plt.figure(figsize=(8, 6))
    for ds in datasets:
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["val_logic"],
            color=colors[ds],
            label=ds_labels[ds],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Logical Consistency Accuracy")
    plt.title("Logical Consistency Accuracy Comparison\n(MNIST, Fashion-MNIST, SVHN)")
    plt.legend()
    fname = os.path.join(working_dir, "val_logic_acc_compare_all_datasets.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating logical consistency accuracy comparison plot: {e}")
    plt.close()

# 5. For each dataset: Histogram of predictions vs ground truth at final epoch (max 5 datasets, here just 3)
for ds in datasets:
    try:
        preds = experiment_data[ds].get("predictions", None)
        gts = experiment_data[ds].get("ground_truth", None)
        if preds is not None and gts is not None:
            plt.figure(figsize=(7, 4))
            plt.hist(
                [gts, preds], bins=2, alpha=0.7, label=["Ground Truth", "Predictions"]
            )
            plt.xticks([0, 1])
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                f"{ds_labels[ds]} Validation Predictions vs Ground Truth\nLeft: Ground Truth, Right: Model Predictions (Final Epoch)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_val_pred_hist.png")
            plt.savefig(fname)
            plt.close()
    except Exception as e:
        print(f"Error creating val pred histogram for {ds}: {e}")
        plt.close()

# 6. For each dataset: Overlaid validation accuracy & logical consistency across epochs
for ds in datasets:
    try:
        plt.figure(figsize=(8, 6))
        epochs = experiment_data[ds]["epochs"]
        plt.plot(
            epochs,
            experiment_data[ds]["metrics"]["val_acc"],
            label="Validation Accuracy",
        )
        plt.plot(
            epochs,
            experiment_data[ds]["metrics"]["val_logic"],
            label="Logical Consistency Accuracy",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds_labels[ds]} - Val and Logical Consistency Accuracy")
        plt.legend()
        fname = os.path.join(working_dir, f"{ds}_val_vs_logic_accuracy.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating overlaid acc/logic plot for {ds}: {e}")
        plt.close()

# 7. Print final accuracy metrics for report
try:
    print("Final metrics per dataset:")
    for ds in datasets:
        val_acc = experiment_data[ds]["metrics"]["val_acc"][-1]
        val_logic = experiment_data[ds]["metrics"]["val_logic"][-1]
        print(
            f"  {ds_labels[ds]}: Final Val Acc = {val_acc:.4f}, Final Logical Consistency = {val_logic:.4f}"
        )
except Exception as e:
    print(f"Error printing final metrics: {e}")
