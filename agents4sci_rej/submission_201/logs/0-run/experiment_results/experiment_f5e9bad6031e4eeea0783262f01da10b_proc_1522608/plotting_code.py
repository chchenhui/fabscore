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

# Plot 1: Training/validation accuracy and logical accuracy curves per dataset
for ds in dsnames:
    try:
        epochs = experiment_data["late_fusion_only_ablation"][ds]["epochs"]
        metrics = experiment_data["late_fusion_only_ablation"][ds]["metrics"]
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, metrics["train"], label="Train Accuracy")
        plt.plot(epochs, metrics["val"], label="Validation Accuracy")
        plt.plot(epochs, metrics["train_logic"], label="Train Logic Consistency Acc.")
        plt.plot(epochs, metrics["val_logic"], label="Val Logic Consistency Acc.")
        plt.title(
            f"{ds} - Training & Validation Accuracies with Logical Consistency\nLate Fusion Only Ablation"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_all_accuracies_latefusion.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot for {ds}: {e}")
        plt.close()

# Plot 2: Compare predicted vs ground truth at last epoch (scatter or confusion) for each dataset
for ds in dsnames:
    try:
        preds = experiment_data["late_fusion_only_ablation"][ds]["predictions"]
        gt = experiment_data["late_fusion_only_ablation"][ds]["ground_truth"]
        # Only plot at most 100 points for clarity
        n_plot = min(len(preds), 100)
        plt.figure(figsize=(6, 6))
        plt.scatter(
            np.arange(n_plot),
            gt[:n_plot],
            marker="o",
            color="g",
            label="Ground Truth",
            alpha=0.7,
        )
        plt.scatter(
            np.arange(n_plot),
            preds[:n_plot],
            marker="x",
            color="r",
            label="Prediction",
            alpha=0.7,
        )
        plt.ylim(-0.2, 1.2)
        plt.yticks([0, 1])
        plt.xlabel("Sample Index")
        plt.ylabel("Class (0:False, 1:True)")
        plt.title(
            f"{ds} - Left: Ground Truth (o), Right: Predicted (x)\nFinal Validation Samples, Late Fusion Only Ablation"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds}_gt_vs_pred_scatter_latefusion.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating GT vs Prediction plot for {ds}: {e}")
        plt.close()

# Plot 3: Overall comparison of best validation accuracy and logic accuracy for all datasets
try:
    best_acc = [
        max(experiment_data["late_fusion_only_ablation"][ds]["metrics"]["val"])
        for ds in dsnames
    ]
    best_logic = [
        max(experiment_data["late_fusion_only_ablation"][ds]["metrics"]["val_logic"])
        for ds in dsnames
    ]
    x = np.arange(len(dsnames))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, best_acc, width, label="Max Val Accuracy")
    plt.bar(x + width / 2, best_logic, width, label="Max Val Logic Consistency Acc.")
    plt.xticks(x, dsnames)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(
        "Max Validation and Logic Accuracies per Dataset\nLate Fusion Only Ablation"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "latefusion_compare_max_acc_logic.png"))
    plt.close()
except Exception as e:
    print(f"Error creating overall accuracy comparison plot: {e}")
    plt.close()
