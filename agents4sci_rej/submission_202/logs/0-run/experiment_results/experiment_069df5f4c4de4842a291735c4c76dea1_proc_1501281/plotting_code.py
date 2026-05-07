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
    experiment_data = None

if experiment_data and "mnist_claims" in experiment_data:
    d = experiment_data["mnist_claims"]
    # Loss curves
    try:
        plt.figure()
        epochs = d.get("epochs")
        train_loss = d.get("losses", {}).get("train")
        val_loss = d.get("losses", {}).get("val")
        if (
            epochs is not None
            and train_loss
            and val_loss
            and len(train_loss) == len(epochs)
        ):
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("MNIST Claims Dataset: Training and Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "mnist_claims_loss_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve: {e}")
        plt.close()

    # Accuracy curves
    try:
        plt.figure()
        train_acc = d.get("metrics", {}).get("train_acc")
        val_acc = d.get("metrics", {}).get("val_acc")
        if (
            epochs is not None
            and train_acc
            and val_acc
            and len(train_acc) == len(epochs)
        ):
            plt.plot(epochs, train_acc, label="Train Accuracy")
            plt.plot(epochs, val_acc, label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("MNIST Claims Dataset: Training and Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(working_dir, "mnist_claims_accuracy_curve.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve: {e}")
        plt.close()

    # Prediction vs ground-truth scatter plot (for last epoch)
    try:
        preds = d.get("predictions")
        gts = d.get("ground_truth")
        if (
            preds is not None
            and gts is not None
            and len(preds) == len(gts)
            and len(preds) > 0
        ):
            plt.figure(figsize=(6, 4))
            plt.scatter(
                np.arange(len(preds)),
                preds,
                label="Prediction",
                alpha=0.6,
                color="b",
                marker="o",
                s=25,
            )
            plt.scatter(
                np.arange(len(gts)),
                gts,
                label="Ground Truth",
                alpha=0.6,
                color="r",
                marker="x",
                s=25,
            )
            plt.xlabel("Sample Index")
            plt.ylabel("Label")
            plt.title(
                "MNIST Claims Dataset: Val Set Predictions vs Ground Truth\n(Left: Ground Truth [red x], Right: Prediction [blue o])"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, "mnist_claims_pred_vs_gt.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating prediction/gt plot: {e}")
        plt.close()
else:
    print("No experiment data for mnist_claims.")
