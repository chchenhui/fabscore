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

# Plot training loss
try:
    train_losses = experiment_data["synthetic_dynamic_network"]["losses"]["train"]
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(
        os.path.join(working_dir, "synthetic_dynamic_network_training_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot validation F1 score
try:
    val_f1_scores = experiment_data["synthetic_dynamic_network"]["metrics"]["val"]
    plt.figure()
    plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, marker="o")
    plt.title("Validation F1 Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.savefig(
        os.path.join(working_dir, "synthetic_dynamic_network_validation_f1.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating validation F1 score plot: {e}")
    plt.close()
