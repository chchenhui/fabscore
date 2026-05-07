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

for dsname in ["mnist", "fashion_mnist", "svhn"]:
    exp = experiment_data["no_logic_supervision"][dsname]
    # 1. Accuracy curves
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(exp["epochs"], exp["metrics"]["train"], label="Train Acc")
        plt.plot(exp["epochs"], exp["metrics"]["val"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{dsname.upper()} - Training/Validation Accuracy\n(No Logic Supervision)"
        )
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dsname}_no_logic_acc_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating acc curve for {dsname}: {e}")
        plt.close()
    # 2. Loss curves
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(exp["epochs"], exp["losses"]["train"], label="Train Loss")
        plt.plot(exp["epochs"], exp["losses"]["val"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{dsname.upper()} - Training/Validation Loss\n(No Logic Supervision)"
        )
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(working_dir, f"{dsname}_no_logic_loss_curves.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dsname}: {e}")
        plt.close()
    # 3. Scatter: Ground Truth vs Predictions on Val (final epoch)
    try:
        plt.figure(figsize=(7, 7))
        gt = exp.get("ground_truth")
        preds = exp.get("predictions")
        if (
            gt is not None
            and preds is not None
            and len(gt) == len(preds)
            and len(gt) > 0
        ):
            plt.scatter(gt, preds, alpha=0.2, label="Samples")
            plt.xlabel("Ground Truth")
            plt.ylabel("Model Prediction")
            plt.title(
                f"{dsname.upper()} - Val: Ground Truth vs Prediction\n(No Logic Supervision, Last Epoch)"
            )
            plt.yticks([0, 1])
            plt.xticks([0, 1])
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dsname}_no_logic_val_gt_vs_pred.png")
            )
        plt.close()
    except Exception as e:
        print(f"Error creating GT-vs-Pred scatter for {dsname}: {e}")
        plt.close()
