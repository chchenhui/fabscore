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

# 1. Accuracy curves: One plot per optimizer
try:
    for opt_name, record in (
        experiment_data.get("optimizer_type", {}).get("mnist_claims", {}).items()
    ):
        epochs = record["epochs"]
        train_acc = record["metrics"]["train_acc"]
        val_acc = record["metrics"]["val_acc"]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"Train/Validation Accuracy Curve\nMNIST Claims - Optimizer: {opt_name}"
        )
        plt.legend()
        fname = f"mnist_claims_accuracy_curve_{opt_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating accuracy curve plots: {e}")
    plt.close()

# 2. Loss curves: One plot per optimizer
try:
    for opt_name, record in (
        experiment_data.get("optimizer_type", {}).get("mnist_claims", {}).items()
    ):
        epochs = record["epochs"]
        train_loss = record["losses"]["train"]
        val_loss = record["losses"]["val"]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Train/Validation Loss Curve\nMNIST Claims - Optimizer: {opt_name}")
        plt.legend()
        fname = f"mnist_claims_loss_curve_{opt_name}.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
except Exception as e:
    print(f"Error creating loss curve plots: {e}")
    plt.close()

# 3. Validation accuracy overlay: All optimizers
try:
    plt.figure(figsize=(8, 5))
    for opt_name, record in (
        experiment_data.get("optimizer_type", {}).get("mnist_claims", {}).items()
    ):
        plt.plot(record["epochs"], record["metrics"]["val_acc"], label=f"{opt_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curve (All Optimizers)\nMNIST Claims")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "mnist_claims_accuracy_curve_all_optimizers.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating overlay accuracy plot: {e}")
    plt.close()

# 4. Confusion Matrices (using last epoch predictions for each optimizer)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, fname, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["False", "True"])
    plt.figure(figsize=(4, 4))
    disp.plot(cmap="Blues", ax=plt.gca(), colorbar=False)
    plt.title(title)
    plt.savefig(fname)
    plt.close()


try:
    all_opts = list(
        experiment_data.get("optimizer_type", {}).get("mnist_claims", {}).keys()
    )
    max_cm = min(len(all_opts), 5)
    interval = max(1, len(all_opts) // max_cm)
    for idx, opt_name in enumerate(all_opts):
        if idx % interval != 0 and len(all_opts) > 5:
            continue
        record = experiment_data["optimizer_type"]["mnist_claims"][opt_name]
        preds = np.array(record["predictions"]).astype(int)
        gts = np.array(record["ground_truth"]).astype(int)
        fname = os.path.join(
            working_dir, f"mnist_claims_confusion_matrix_{opt_name}.png"
        )
        plot_confusion_matrix(
            gts,
            preds,
            fname,
            f"Confusion Matrix\nMNIST Claims - Optimizer: {opt_name}\n(Left: Ground Truth, Right: Predicted)",
        )
except Exception as e:
    print(f"Error creating confusion matrices: {e}")
    plt.close()
