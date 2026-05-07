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
    experiment_data = None

datasets = ["mnist", "fashion_mnist", "svhn"]
mainkey = "attention_mask_removal"

for ds in datasets:
    # 1. Plot training and validation accuracy over epochs
    try:
        plt.figure()
        plt.plot(
            experiment_data[mainkey][ds]["epochs"],
            experiment_data[mainkey][ds]["metrics"]["train"],
            label="Train Accuracy",
        )
        plt.plot(
            experiment_data[mainkey][ds]["epochs"],
            experiment_data[mainkey][ds]["metrics"]["val"],
            label="Validation Accuracy",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"[Ablation: No Attn Mask] {ds} - Training/Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_ablation_acc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {ds}: {e}")
        plt.close()

    # 2. Plot training and validation logical consistency accuracy
    try:
        plt.figure()
        plt.plot(
            experiment_data[mainkey][ds]["epochs"],
            experiment_data[mainkey][ds]["metrics"]["train_logic"],
            label="Train Logic Acc",
        )
        plt.plot(
            experiment_data[mainkey][ds]["epochs"],
            experiment_data[mainkey][ds]["metrics"]["val_logic"],
            label="Val Logic Acc",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.title(f"[Ablation: No Attn Mask] {ds} - Logical Consistency Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_ablation_logic_acc_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating logic accuracy curve for {ds}: {e}")
        plt.close()

    # 3. Plot training and validation loss over epochs
    try:
        plt.figure()
        plt.plot(
            experiment_data[mainkey][ds]["epochs"],
            experiment_data[mainkey][ds]["losses"]["train"],
            label="Train Loss",
        )
        plt.plot(
            experiment_data[mainkey][ds]["epochs"],
            experiment_data[mainkey][ds]["losses"]["val"],
            label="Val Loss",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.title(f"[Ablation: No Attn Mask] {ds} - Training/Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_ablation_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {ds}: {e}")
        plt.close()

    # 4. Plot ground truth vs prediction rates at the final epoch as bar charts
    try:
        gts = experiment_data[mainkey][ds]["ground_truth"]
        preds = experiment_data[mainkey][ds]["predictions"]
        if (gts is not None and len(gts) > 0) and (
            preds is not None and len(preds) > 0
        ):
            plt.figure(figsize=(6, 4))
            gt_1 = np.mean(gts)
            pred_1 = np.mean(preds)
            gt_0 = 1 - gt_1
            pred_0 = 1 - pred_1
            x = np.arange(2)
            width = 0.35
            plt.bar(x - width / 2, [gt_0, gt_1], width, label="Ground Truth")
            plt.bar(x + width / 2, [pred_0, pred_1], width, label="Predicted")
            plt.xticks(x, ["Label 0", "Label 1"])
            plt.ylabel("Proportion")
            plt.title(
                f"[Ablation: No Attn Mask]\n{ds.capitalize()} - Final Label Distribution\nLeft: Ground Truth, Right: Predicted"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{ds}_ablation_label_distribution.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating label distribution plot for {ds}: {e}")
        plt.close()

# 5. Comparison Plot Across Datasets: Final Logical Consistency Accuracy (bar graph)
try:
    plt.figure()
    final_logic = []
    for ds in datasets:
        val_logic_list = experiment_data[mainkey][ds]["metrics"]["val_logic"]
        final_logic.append(val_logic_list[-1] if len(val_logic_list) > 0 else 0)
    plt.bar(datasets, final_logic, color=["b", "r", "g"])
    plt.ylim(0, 1)
    plt.ylabel("Logical Consistency Accuracy")
    plt.title("[Ablation: No Attn Mask] Final Logical Consistency Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "ablation_final_logic_accuracy_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating final logic accuracy comparison plot: {e}")
    plt.close()
