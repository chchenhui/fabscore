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

hidden_sizes = experiment_data["hidden_layer_size_tuning"]["synthetic_data"][
    "hidden_sizes"
]
train_losses = experiment_data["hidden_layer_size_tuning"]["synthetic_data"]["losses"][
    "train"
]
val_f1_scores = experiment_data["hidden_layer_size_tuning"]["synthetic_data"][
    "metrics"
]["val"]
predictions = experiment_data["hidden_layer_size_tuning"]["synthetic_data"][
    "predictions"
]
ground_truth = experiment_data["hidden_layer_size_tuning"]["synthetic_data"][
    "ground_truth"
]

for i, hidden_size in enumerate(hidden_sizes):
    try:
        plt.figure()
        epochs = range(1, len(train_losses) // len(hidden_sizes) + 1)
        plt.plot(epochs, train_losses[i :: len(hidden_sizes)], label="Training Loss")
        plt.plot(epochs, val_f1_scores[i :: len(hidden_sizes)], label="Validation F1")
        plt.title(f"Training Loss and Validation F1 for Hidden Size {hidden_size}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/F1 Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_f1_hidden_size_{hidden_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss and F1 plot for hidden size {hidden_size}: {e}")
        plt.close()

    try:
        if i % (len(hidden_sizes) // 5) == 0:  # Plot at intervals
            plt.figure()
            plt.plot(range(len(ground_truth)), ground_truth, "o", label="Ground Truth")
            plt.plot(
                range(len(predictions[i])), predictions[i], "x", label="Predictions"
            )
            plt.title(f"Predictions vs Ground Truth for Hidden Size {hidden_size}")
            plt.xlabel("Node Index")
            plt.ylabel("Class")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"predictions_hidden_size_{hidden_size}.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating predictions plot for hidden size {hidden_size}: {e}")
        plt.close()
