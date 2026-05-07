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

# Fetch sweep params and data
try:
    cnn_hidden_sizes = [
        int(hid) for hid in experiment_data["cnn_hidden_size"]["mnist_claims"].keys()
    ]
except Exception as e:
    print(f"Could not infer hidden sizes, error: {e}")

# Plot 1: Training/Validation Accuracy vs Epoch for each hidden size
try:
    plt.figure(figsize=(10, 6))
    for hid in cnn_hidden_sizes:
        d = experiment_data["cnn_hidden_size"]["mnist_claims"][str(hid)]
        plt.plot(d["epochs"], d["metrics"]["val_acc"], label=f"Val acc (hid={hid})")
        plt.plot(
            d["epochs"], d["metrics"]["train_acc"], "--", label=f"Train acc (hid={hid})"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Train/Validation Accuracy vs Epoch\nMNIST Claims Dataset (CNN Hidden Size Sweep)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "mnist_claims_cnn_hidden_size_accuracy_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy curve: {e}")
    plt.close()

# Plot 2: Training/Validation Loss vs Epoch for each hidden size
try:
    plt.figure(figsize=(10, 6))
    for hid in cnn_hidden_sizes:
        d = experiment_data["cnn_hidden_size"]["mnist_claims"][str(hid)]
        plt.plot(d["epochs"], d["losses"]["val"], label=f"Val loss (hid={hid})")
        plt.plot(
            d["epochs"], d["losses"]["train"], "--", label=f"Train loss (hid={hid})"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Train/Validation Loss vs Epoch\nMNIST Claims Dataset (CNN Hidden Size Sweep)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "mnist_claims_cnn_hidden_size_loss_curve.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss curve: {e}")
    plt.close()

# Plot 3: Bar plot of final val accuracy for each hidden size
try:
    final_val_accs = []
    for hid in cnn_hidden_sizes:
        acc = experiment_data["cnn_hidden_size"]["mnist_claims"][str(hid)]["metrics"][
            "val_acc"
        ][-1]
        final_val_accs.append(acc)
    plt.figure(figsize=(8, 6))
    plt.bar([str(hid) for hid in cnn_hidden_sizes], final_val_accs, color="skyblue")
    plt.xlabel("CNN Hidden Size")
    plt.ylabel("Final Validation Accuracy")
    plt.title("Final Validation Accuracy by CNN Hidden Size\nMNIST Claims Dataset")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mnist_claims_final_val_acc_barplot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating barplot: {e}")
    plt.close()

# Find best model (highest final val acc)
try:
    best_hid_idx = np.argmax(final_val_accs)
    best_hid = cnn_hidden_sizes[best_hid_idx]
    best_exp = experiment_data["cnn_hidden_size"]["mnist_claims"][str(best_hid)]
except Exception as e:
    print(f"Error identifying best hidden size: {e}")

# Plot 4: Histogram of predictions vs ground truth for best model
try:
    preds = np.array(best_exp.get("predictions", []))
    gts = np.array(best_exp.get("ground_truth", []))
    if preds.size > 0 and gts.size > 0:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(gts, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color="orange")
        plt.title("Ground Truth\n(MNIST Claims, best hid=%d)" % best_hid)
        plt.xticks([0, 1])
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.subplot(1, 2, 2)
        plt.hist(preds, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color="royalblue")
        plt.title("Predicted Labels\n(MNIST Claims, best hid=%d)" % best_hid)
        plt.xticks([0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Count")
        plt.suptitle(
            "Left: Ground Truth, Right: Generated (Predicted)\nHistogram, MNIST Claims Dataset"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        plt.savefig(
            os.path.join(
                working_dir, f"mnist_claims_best_hid{best_hid}_gt_vs_pred_histogram.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating GT/Prediction histogram: {e}")
    plt.close()

# Plot 5: Confusion matrix for best model
try:
    from sklearn.metrics import confusion_matrix

    if preds.size > 0 and gts.size > 0:
        cm = confusion_matrix(gts, preds)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix (Hidden Size={best_hid})\nMNIST Claims Dataset")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["False", "True"])
        plt.yticks(tick_marks, ["False", "True"])
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"mnist_claims_best_hid{best_hid}_confusion_matrix.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
