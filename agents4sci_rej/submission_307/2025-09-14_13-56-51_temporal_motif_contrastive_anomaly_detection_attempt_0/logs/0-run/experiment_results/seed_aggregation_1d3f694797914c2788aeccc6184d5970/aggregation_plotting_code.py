import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
experiment_data_path_list = [
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_464c7def0b414a1ebf0e6ef5517a4053_proc_1274/experiment_data.npy",
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_31d8f13ecac7464ab88e0d13c6ac7361_proc_1273/experiment_data.npy",
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_adf352379a1545b38b25de40e7151321_proc_1273/experiment_data.npy",
]

all_train_losses = []
all_val_f1_scores = []

for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        train_losses = experiment_data["synthetic_dynamic_network"]["losses"]["train"]
        val_f1_scores = experiment_data["synthetic_dynamic_network"]["metrics"]["val"]
        all_train_losses.append(train_losses)
        all_val_f1_scores.append(val_f1_scores)
    except Exception as e:
        print(f"Error loading experiment data from {experiment_data_path}: {e}")

# Calculate mean and standard error
mean_train_losses = np.mean(all_train_losses, axis=0)
stderr_train_losses = np.std(all_train_losses, axis=0) / np.sqrt(len(all_train_losses))

mean_val_f1_scores = np.mean(all_val_f1_scores, axis=0)
stderr_val_f1_scores = np.std(all_val_f1_scores, axis=0) / np.sqrt(
    len(all_val_f1_scores)
)

# Plot aggregated training loss
try:
    plt.figure()
    epochs = range(1, len(mean_train_losses) + 1)
    plt.errorbar(
        epochs,
        mean_train_losses,
        yerr=stderr_train_losses,
        fmt="-o",
        label="Mean ± StdErr",
    )
    plt.title("Aggregated Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training loss plot: {e}")
    plt.close()

# Plot aggregated validation F1 score
try:
    plt.figure()
    epochs = range(1, len(mean_val_f1_scores) + 1)
    plt.errorbar(
        epochs,
        mean_val_f1_scores,
        yerr=stderr_val_f1_scores,
        fmt="-o",
        label="Mean ± StdErr",
    )
    plt.title("Aggregated Validation F1 Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "aggregated_validation_f1.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated validation F1 score plot: {e}")
    plt.close()
