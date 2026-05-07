import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load all experiment data
experiment_data_paths = [
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_f58409f0f5ac4bf9be457b5cb8df4339_proc_5660/experiment_data.npy",
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_efef644de81f4f38a63ebe416507de69_proc_5657/experiment_data.npy",
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_af46cdc8800c441e84b3bdaabfb80308_proc_5657/experiment_data.npy",
]

all_losses = []
all_val_f1_scores = []
epochs_list = None

try:
    for path in experiment_data_paths:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), path), allow_pickle=True
        ).item()
        if epochs_list is None:  # Initialize epochs_list if not done
            epochs_list = experiment_data["feature_count_ablation"][
                "synthetic_dynamic_network"
            ]["epoch_settings"]
        all_losses.append(
            experiment_data["feature_count_ablation"]["synthetic_dynamic_network"][
                "losses"
            ]["train"]
        )
        all_val_f1_scores.append(
            experiment_data["feature_count_ablation"]["synthetic_dynamic_network"][
                "metrics"
            ]["val"]
        )
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Calculate means and standard errors
try:
    mean_losses = np.mean(all_losses, axis=0)
    std_err_losses = np.std(all_losses, axis=0) / np.sqrt(len(all_losses))
    mean_val_f1_scores = np.mean(all_val_f1_scores, axis=0)
    std_err_val_f1_scores = np.std(all_val_f1_scores, axis=0) / np.sqrt(
        len(all_val_f1_scores)
    )

    # Plot Training Loss with Error Bars
    plt.figure()
    plt.errorbar(
        epochs_list,
        mean_losses,
        yerr=std_err_losses,
        label="Mean Training Loss",
        fmt="-o",
        capsize=5,
    )
    plt.title("Aggregated Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "aggregated_training_loss.png"))
    plt.close()

    # Plot Validation F1 Score with Error Bars
    plt.figure()
    plt.errorbar(
        epochs_list,
        mean_val_f1_scores,
        yerr=std_err_val_f1_scores,
        label="Mean Validation F1 Score",
        fmt="-o",
        capsize=5,
    )
    plt.title("Aggregated Validation F1 Score Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "aggregated_validation_f1_score.png"))
    plt.close()

except Exception as e:
    print(f"Error creating aggregated plots: {e}")
    plt.close()
