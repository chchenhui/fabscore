import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    # Load experiment data
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data = experiment_data["temporal_motif_contrastive_learning"][
        "synthetic_dynamic_network"
    ]
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Plot training loss
    plt.figure()
    for idx, epochs in enumerate(
        data["epoch_settings"][:5]
    ):  # Plot only first 5 settings
        plt.plot(data["losses"]["train"][:epochs], label=f"Epochs: {epochs}")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dynamic_network_training_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    # Plot validation F1 Score
    plt.figure()
    for idx, epochs in enumerate(
        data["epoch_settings"][:5]
    ):  # Plot only first 5 settings
        plt.plot(data["metrics"]["val"][:epochs], label=f"Epochs: {epochs}")
    plt.title("Validation F1 Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dynamic_network_val_f1.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation f1 score plot: {e}")
    plt.close()
