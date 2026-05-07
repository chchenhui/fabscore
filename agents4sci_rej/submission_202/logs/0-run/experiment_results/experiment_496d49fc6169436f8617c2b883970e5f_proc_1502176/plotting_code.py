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

# Plot accuracy and loss curves per kernel size
kernel_names = list(experiment_data["cnn_kernel_size"].keys())
final_val_accs = []
for kname in kernel_names:
    try:
        data = experiment_data["cnn_kernel_size"][kname]
        epochs = data["epochs"]
        train_acc = data["metrics"]["train_acc"]
        val_acc = data["metrics"]["val_acc"]
        train_loss = data["losses"]["train"]
        val_loss = data["losses"]["val"]

        # Accuracy curve
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"Train/Validation Accuracy\nKernel size: {kname.replace('kernel', '')}, Dataset: MNIST Claims"
        )
        plt.legend()
        plt.tight_layout()
        acc_path = os.path.join(working_dir, f"mnist_claims_accuracy_curve_{kname}.png")
        plt.savefig(acc_path)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {kname}: {e}")
        plt.close()

    try:
        # Loss curve
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"Train/Validation Loss\nKernel size: {kname.replace('kernel', '')}, Dataset: MNIST Claims"
        )
        plt.legend()
        plt.tight_layout()
        loss_path = os.path.join(working_dir, f"mnist_claims_loss_curve_{kname}.png")
        plt.savefig(loss_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {kname}: {e}")
        plt.close()

    # For summary bar plot
    if "val_acc" in data["metrics"] and len(data["metrics"]["val_acc"]) > 0:
        final_val_accs.append(data["metrics"]["val_acc"][-1])
    else:
        final_val_accs.append(np.nan)

# Bar plot comparing final validation accuracy
try:
    plt.figure(figsize=(7, 5))
    labels = [k.replace("kernel", "") for k in kernel_names]
    plt.bar(labels, final_val_accs, color=["tab:blue", "tab:orange", "tab:green"])
    plt.xlabel("Kernel Size")
    plt.ylabel("Final Validation Accuracy")
    plt.title("Final Validation Accuracy by CNN Kernel Size\nDataset: MNIST Claims")
    for i, val in enumerate(final_val_accs):
        plt.text(i, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    bar_path = os.path.join(
        working_dir, "mnist_claims_final_val_acc_by_kernel_size.png"
    )
    plt.savefig(bar_path)
    plt.close()
except Exception as e:
    print(f"Error creating bar plot for final validation accuracy: {e}")
    plt.close()
