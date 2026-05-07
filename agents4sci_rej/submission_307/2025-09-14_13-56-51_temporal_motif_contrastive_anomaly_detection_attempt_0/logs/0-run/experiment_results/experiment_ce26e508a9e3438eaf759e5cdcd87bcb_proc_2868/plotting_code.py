import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting
try:
    batch_sizes = experiment_data["batch_size_tuning"].keys()
    for batch_size in batch_sizes:
        # Extract data
        train_losses = experiment_data["batch_size_tuning"][batch_size]["losses"][
            "train"
        ]
        val_f1_scores = experiment_data["batch_size_tuning"][batch_size]["metrics"][
            "val"
        ]
        epochs = range(1, len(train_losses) + 1)

        # Plot training loss
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.title(f"Training Loss for {batch_size}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"train_loss_{batch_size}.png"))
        plt.close()

        # Plot validation F1 score
        plt.figure()
        plt.plot(epochs, val_f1_scores, label="Validation F1 Score")
        plt.title(f"Validation F1 Score for {batch_size}")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"val_f1_{batch_size}.png"))
        plt.close()

except Exception as e:
    print(f"Error creating plots: {e}")
    plt.close()
