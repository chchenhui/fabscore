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
    train_losses = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "losses"
    ]["train"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot validation F1 scores at node count intervals
try:
    val_f1_scores = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "metrics"
    ]["val"]
    node_counts = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "node_count_settings"
    ]
    plt.figure()
    epochs = len(val_f1_scores) // len(node_counts)
    for i, node_count in enumerate(set(node_counts)):
        plt.plot(
            val_f1_scores[i * epochs : (i + 1) * epochs],
            label=f"Node Count {node_count}",
        )
    plt.title("Validation F1 Scores Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_f1_scores.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation F1 scores plot: {e}")
    plt.close()

# Plot Temporal Motif Coverage (TMC)
try:
    tmc_values = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "metrics"
    ]["tmc"]
    plt.figure()
    for i, node_count in enumerate(set(node_counts)):
        plt.plot(
            tmc_values[i * epochs : (i + 1) * epochs], label=f"Node Count {node_count}"
        )
    plt.title("Temporal Motif Coverage Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("TMC")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "tmc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating TMC plot: {e}")
    plt.close()

# Plot predictions vs ground truth
try:
    predictions = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "predictions"
    ]
    ground_truth = experiment_data["node_count_ablation"]["synthetic_dynamic_network"][
        "ground_truth"
    ]
    plt.figure()
    plt.scatter(range(len(ground_truth)), ground_truth, label="Ground Truth", alpha=0.5)
    plt.scatter(range(len(predictions)), predictions, label="Predictions", alpha=0.5)
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "predictions_vs_ground_truth.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
