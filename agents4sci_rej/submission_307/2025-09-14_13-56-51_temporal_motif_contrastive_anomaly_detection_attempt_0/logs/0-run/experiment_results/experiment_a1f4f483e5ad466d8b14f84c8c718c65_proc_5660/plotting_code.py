import matplotlib.pyplot as plt
import numpy as np
import os

# Set up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    # Load experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract data
data_dict = experiment_data["learning_rate_ablation"]["synthetic_dynamic_network"]
train_losses = data_dict["losses"]["train"]
val_metrics = data_dict["metrics"]["val"]
learning_rates = data_dict["learning_rate_settings"]

# Plot training losses and validation F1 scores
try:
    for lr_idx, lr in enumerate(set(learning_rates)):
        plt.figure(figsize=(10, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(
            range(1, 21),
            train_losses[lr_idx * 20 : (lr_idx + 1) * 20],
            label=f"LR={lr}",
        )
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot validation F1 score
        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, 21), val_metrics[lr_idx * 20 : (lr_idx + 1) * 20], label=f"LR={lr}"
        )
        plt.title("Validation F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        # Save plot
        plt.suptitle(f"Learning Rate: {lr}")
        plt.savefig(os.path.join(working_dir, f"plot_lr_{lr}.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plots: {e}")
    plt.close()  # Always close figures even if errors occur
