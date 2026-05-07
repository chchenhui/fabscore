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

dsnames = ["mnist", "fashion_mnist", "svhn"]
dslabels = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "svhn": "SVHN"}

# 1. Overlay accuracy curves (validation, logic) for all datasets
try:
    plt.figure(figsize=(8, 6))
    for ds in dsnames:
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["val"],
            label=f"{dslabels[ds]} Val Acc",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curves Across Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "compare_val_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison val accuracy plot: {e}")
    plt.close()

try:
    plt.figure(figsize=(8, 6))
    for ds in dsnames:
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["val_logic"],
            label=f"{dslabels[ds]} Logic Acc",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Logical Consistency Accuracy")
    plt.title("Logical Consistency Accuracy Across Datasets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "compare_val_logic_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating comparison val logic acc plot: {e}")
    plt.close()

# 2. Dataset-specific train/val/loss/logic curves
for ds in dsnames:
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["train"],
            label="Train Acc",
        )
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["val"],
            label="Val Acc",
        )
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["train_logic"],
            label="Train Logic Acc",
        )
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["metrics"]["val_logic"],
            label="Val Logic Acc",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"{dslabels[ds]} - Accuracies per Epoch")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_all_accuracies.png"))
        plt.close()
    except Exception as e:
        print(f"Error in plot: {ds} accuracy curves: {e}")
        plt.close()

    try:
        plt.figure(figsize=(8, 6))
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data[ds]["epochs"],
            experiment_data[ds]["losses"]["val"],
            label="Val Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dslabels[ds]} - Losses per Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error in plot: {ds} loss curves: {e}")
        plt.close()

# 3. Prediction vs Ground Truth Histogram (for each dataset, last epoch, max 1 histogram per ds)
for ds in dsnames:
    try:
        preds = experiment_data[ds].get("predictions", None)
        gts = experiment_data[ds].get("ground_truth", None)
        if preds is not None and gts is not None:
            plt.figure(figsize=(7, 4))
            plt.hist(
                [gts, preds],
                bins=[-0.5, 0.5, 1.5],
                alpha=0.7,
                label=["Ground Truth", "Predictions"],
            )
            plt.xticks([0, 1])
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                f"{dslabels[ds]} Validation Prediction Distribution\nLeft: Ground Truth, Right: Predicted (final epoch)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, f"{ds}_val_pred_hist.png"))
            plt.close()
    except Exception as e:
        print(f"Error in plot: {ds} pred hist: {e}")
        plt.close()

# 4. Print summary final val and final logic accuracy by dataset
try:
    print("Final validation and logical consistency accuracy by dataset:")
    for ds in dsnames:
        val = (
            experiment_data[ds]["metrics"]["val"][-1]
            if experiment_data[ds]["metrics"]["val"]
            else -1
        )
        logic = (
            experiment_data[ds]["metrics"]["val_logic"][-1]
            if experiment_data[ds]["metrics"]["val_logic"]
            else -1
        )
        print(f"  {dslabels[ds]}: Val Acc={val:.4f}  | Logic Acc={logic:.4f}")
except Exception as e:
    print(f"Error printing final accuracy summary: {e}")

# 5. Compare final acc/logical acc as grouped bar chart
try:
    final_vals = [
        (
            experiment_data[ds]["metrics"]["val"][-1]
            if experiment_data[ds]["metrics"]["val"]
            else 0
        )
        for ds in dsnames
    ]
    final_logics = [
        (
            experiment_data[ds]["metrics"]["val_logic"][-1]
            if experiment_data[ds]["metrics"]["val_logic"]
            else 0
        )
        for ds in dsnames
    ]
    x = np.arange(len(dsnames))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, final_vals, width, label="Validation Accuracy")
    plt.bar(x + width / 2, final_logics, width, label="Logic Consistency Accuracy")
    plt.xticks(x, [dslabels[ds] for ds in dsnames])
    plt.title("Final Accuracies by Dataset")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "final_acc_across_datasets.png"))
    plt.close()
except Exception as e:
    print(f"Error in grouped bar chart: {e}")
    plt.close()
