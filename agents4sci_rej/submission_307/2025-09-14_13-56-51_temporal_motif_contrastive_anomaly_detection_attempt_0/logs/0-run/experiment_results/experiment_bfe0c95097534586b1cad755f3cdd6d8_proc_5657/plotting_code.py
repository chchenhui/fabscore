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
for connectivity in ["sparse", "dense", "random"]:
    try:
        train_losses = experiment_data[connectivity]["losses"]["train"]
        val_metrics = experiment_data[connectivity]["metrics"]["val"]

        # Plot training losses
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.title(f"Training Loss for {connectivity.capitalize()} Connectivity")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{connectivity}_train_loss.png"))
        plt.close()

        # Plot validation F1 scores
        plt.figure()
        plt.plot(val_metrics, label="Validation F1 Score")
        plt.title(f"Validation F1 Score for {connectivity.capitalize()} Connectivity")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{connectivity}_val_f1.png"))
        plt.close()

    except Exception as e:
        print(f"Error creating plots for {connectivity}: {e}")
        plt.close()
