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

# Extract learning rate keys
try:
    lr_tuning = experiment_data["learning_rate_tuning"]
    lr_keys = list(lr_tuning.keys())
except Exception as e:
    print(f"Error extracting learning_rate_tuning data: {e}")

# 1. Per-learning-rate accuracy curves
for lr_key in lr_keys:
    try:
        epochs = lr_tuning[lr_key]["epochs"]
        train_acc = lr_tuning[lr_key]["metrics"]["train"]
        val_acc = lr_tuning[lr_key]["metrics"]["val"]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Curves - Learning Rate {lr_key} (Dataset: MNISTClaim)")
        plt.legend()
        plot_path = os.path.join(working_dir, f"mnistclaim_acc_curve_{lr_key}.png")
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {lr_key}: {e}")
        plt.close()

# 2. Per-learning-rate loss curves
for lr_key in lr_keys:
    try:
        epochs = lr_tuning[lr_key]["epochs"]
        train_loss = lr_tuning[lr_key]["losses"]["train"]
        val_loss = lr_tuning[lr_key]["losses"]["val"]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss Curves - Learning Rate {lr_key} (Dataset: MNISTClaim)")
        plt.legend()
        plot_path = os.path.join(working_dir, f"mnistclaim_loss_curve_{lr_key}.png")
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {lr_key}: {e}")
        plt.close()

# 3. Overlay comparison plot for validation accuracy (max 5 curves)
try:
    plt.figure(figsize=(8, 5))
    for lr_key in lr_keys:
        val_acc = lr_tuning[lr_key]["metrics"]["val"]
        epochs = lr_tuning[lr_key]["epochs"]
        plt.plot(epochs, val_acc, label=f"{lr_key.replace('_','=')}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Validation Accuracy vs Epochs\n(Dataset: MNISTClaim, Learning Rate Tuning)"
    )
    plt.legend(title="Learning Rates")
    plot_path = os.path.join(working_dir, "mnistclaim_acc_curve_lr_compare.png")
    plt.savefig(plot_path)
    plt.close()
except Exception as e:
    print(f"Error creating overlay comparison plot: {e}")
    plt.close()

# 4. Final validation accuracy bar chart
try:
    final_accs = []
    for lr_key in lr_keys:
        acc = (
            lr_tuning[lr_key]["metrics"]["val"][-1]
            if lr_tuning[lr_key]["metrics"]["val"]
            else np.nan
        )
        final_accs.append(acc)
    plt.figure(figsize=(7, 5))
    plt.bar(
        range(len(lr_keys)),
        final_accs,
        tick_label=[k.replace("_", "=") for k in lr_keys],
    )
    plt.ylabel("Final Validation Accuracy")
    plt.title("Final Validation Accuracy by Learning Rate\n(Dataset: MNISTClaim)")
    plot_path = os.path.join(working_dir, "mnistclaim_final_valacc_bar.png")
    plt.savefig(plot_path)
    plt.close()
except Exception as e:
    print(f"Error creating final val acc bar chart: {e}")
    plt.close()
