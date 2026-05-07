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

# Get data by shortcut
try:
    results = experiment_data["num_epochs_tuning"]["mnist_claims"]
    epoch_keys = sorted(
        [k for k in results.keys() if k.startswith("epochs_")],
        key=lambda x: int(x.split("_")[1]),
    )
    epoch_counts = [int(x.split("_")[1]) for x in epoch_keys]
except Exception as e:
    print(f"Error extracting experiment results: {e}")

# 1. Plot accuracy curves for all settings (redundant with original save, but ensure working_dir)
try:
    plt.figure(figsize=(9, 6))
    for idx, ek in enumerate(epoch_keys):
        epochs = results[ek]["epochs"]
        train_acc = results[ek]["metrics"]["train_acc"]
        val_acc = results[ek]["metrics"]["val_acc"]
        plt.plot(
            epochs,
            train_acc,
            linestyle="--",
            alpha=0.6,
            label=f"Train Acc (epochs={epoch_counts[idx]})",
        )
        plt.plot(
            epochs,
            val_acc,
            linestyle="-",
            label=f"Val Acc (epochs={epoch_counts[idx]})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Validation Accuracy Curves\nMNISTClaimDataset (num_epochs tuning)")
    plt.legend()
    save_path = os.path.join(working_dir, "mnist_claims_accuracy_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error creating accuracy curve plot: {e}")
    plt.close()

# 2. Plot loss curves if available
try:
    plt.figure(figsize=(9, 6))
    for idx, ek in enumerate(epoch_keys):
        epochs = results[ek]["epochs"]
        train_loss = results[ek]["losses"]["train"]
        val_loss = results[ek]["losses"]["val"]
        plt.plot(
            epochs,
            train_loss,
            linestyle="--",
            alpha=0.6,
            label=f"Train Loss (epochs={epoch_counts[idx]})",
        )
        plt.plot(
            epochs,
            val_loss,
            linestyle="-",
            label=f"Val Loss (epochs={epoch_counts[idx]})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss Curves\nMNISTClaimDataset (num_epochs tuning)")
    plt.legend()
    save_path = os.path.join(working_dir, "mnist_claims_loss_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 3. Final val prediction vs ground truth histogram for each epoch setting (max 5 plots)
try:
    for ek, epc in zip(epoch_keys, epoch_counts):
        preds = results[ek].get("predictions", None)
        gts = results[ek].get("ground_truth", None)
        if preds is not None and gts is not None:
            plt.figure(figsize=(7, 4))
            plt.hist(
                [gts, preds], bins=2, alpha=0.7, label=["Ground Truth", "Predictions"]
            )
            plt.xticks([0, 1])
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                f"Validation Prediction Distribution (epochs={epc})\nMNISTClaimDataset\n"
                "Left: Ground Truth, Right: Generated Predictions (final epoch)"
            )
            plt.legend()
            save_path = os.path.join(
                working_dir, f"mnist_claims_val_pred_hist_epochs{epc}.png"
            )
            plt.savefig(save_path)
            plt.close()
            print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error creating prediction histogram: {e}")
    plt.close()

# 4. Print summary final validation accuracy for each epoch setting
try:
    print("Final validation accuracies:")
    for ek, epc in zip(epoch_keys, epoch_counts):
        val_accs = results[ek]["metrics"]["val_acc"]
        print(f"  num_epochs={epc}: {val_accs[-1]:.4f}")
except Exception as e:
    print(f"Error printing validation accuracies: {e}")
