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

# Plot accuracy curves for each batch size in one figure (also done in experiment, but do for completeness)
try:
    plt.figure(figsize=(8, 5))
    colors = {32: "tab:blue", 64: "tab:orange", 128: "tab:green"}
    for batch_size in [32, 64, 128]:
        subdict = experiment_data["batch_size"][batch_size]
        epochs = subdict["epochs"]
        tr_acc = subdict["metrics"]["train_acc"]
        val_acc = subdict["metrics"]["val_acc"]
        plt.plot(
            epochs,
            val_acc,
            label=f"Val Acc (batch={batch_size})",
            color=colors[batch_size],
            linestyle="-",
        )
        plt.plot(
            epochs,
            tr_acc,
            label=f"Train Acc (batch={batch_size})",
            color=colors[batch_size],
            linestyle="--",
            alpha=0.6,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "MNIST Claims Verification - Train/Validation Accuracy\n(Batch Size Tuning)"
    )
    plt.legend()
    fname = os.path.join(working_dir, "mnist_claims_accuracy_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# Loss curves for each batch (one figure for all, similar as above)
try:
    plt.figure(figsize=(8, 5))
    for batch_size in [32, 64, 128]:
        subdict = experiment_data["batch_size"][batch_size]
        epochs = subdict["epochs"]
        tr_loss = subdict["losses"]["train"]
        val_loss = subdict["losses"]["val"]
        plt.plot(
            epochs,
            val_loss,
            label=f"Val Loss (batch={batch_size})",
            color=colors[batch_size],
            linestyle="-",
        )
        plt.plot(
            epochs,
            tr_loss,
            label=f"Train Loss (batch={batch_size})",
            color=colors[batch_size],
            linestyle="--",
            alpha=0.6,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MNIST Claims Verification - Train/Validation Loss\n(Batch Size Tuning)")
    plt.legend()
    fname = os.path.join(working_dir, "mnist_claims_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Confusion matrix for predictions vs ground truth on final val set (at end of last epoch) for each batch_size
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

for batch_size in [32, 64, 128]:
    try:
        subdict = experiment_data["batch_size"][batch_size]
        preds = subdict.get("predictions", [])
        gts = subdict.get("ground_truth", [])
        if len(preds) > 0 and len(gts) > 0:
            cm = confusion_matrix(gts, preds)
            disp = ConfusionMatrixDisplay(cm, display_labels=["False", "True"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(
                f"Confusion Matrix\nMNIST Claim Verification (Batch={batch_size})"
            )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            fname = os.path.join(
                working_dir, f"mnist_claims_confusion_matrix_batch{batch_size}.png"
            )
            plt.savefig(fname)
            plt.close()
        else:
            print(
                f"No prediction/ground truth data for batch_size={batch_size}, skipping confusion matrix."
            )
    except Exception as e:
        print(f"Error creating confusion matrix for batch_size={batch_size}: {e}")
        plt.close()
