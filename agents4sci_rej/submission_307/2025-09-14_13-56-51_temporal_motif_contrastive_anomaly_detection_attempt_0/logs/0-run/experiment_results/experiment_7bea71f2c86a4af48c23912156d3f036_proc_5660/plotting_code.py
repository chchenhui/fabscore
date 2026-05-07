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

data = experiment_data["activation_function_ablation"]["synthetic_dynamic_network"]
train_losses = data["losses"]["train"]
val_metrics = data["metrics"]["val"]
predictions = data["predictions"]
ground_truth = data["ground_truth"]
activation_functions = data["activation_functions"]

# Plot training losses
try:
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.title("Training Losses for Different Activation Functions")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dynamic_network_train_losses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot validation F1 scores
try:
    plt.figure()
    plt.plot(val_metrics, label="Validation F1 Score")
    plt.title("Validation F1 Scores for Different Activation Functions")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dynamic_network_val_f1_scores.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating validation F1 score plot: {e}")
    plt.close()

# Plot predictions vs ground truth
try:
    plt.figure()
    plt.scatter(range(len(predictions)), predictions, label="Predictions", alpha=0.5)
    plt.scatter(range(len(ground_truth)), ground_truth, label="Ground Truth", alpha=0.5)
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "synthetic_dynamic_network_predictions_vs_ground_truth.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
