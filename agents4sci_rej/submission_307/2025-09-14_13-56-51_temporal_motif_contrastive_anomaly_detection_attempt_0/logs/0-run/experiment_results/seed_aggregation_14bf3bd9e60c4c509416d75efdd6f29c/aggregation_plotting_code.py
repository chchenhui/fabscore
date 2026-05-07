import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

experiment_data_path_list = [
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_3584be22a4804a749991509b8ebac97d_proc_4370/experiment_data.npy",
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_fc983fd5bff04d1cb8d0237a1153b70b_proc_4371/experiment_data.npy",
    "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_6fdebb88173d411e993d5b75c3cd4553_proc_4370/experiment_data.npy",
]

try:
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    # Aggregate training losses
    train_losses = [
        data["hyperparam_tuning_epochs"]["synthetic_dynamic_network"]["losses"]["train"]
        for data in all_experiment_data
    ]
    max_epochs = max(len(loss) for loss in train_losses)
    aggregated_train_losses = np.array(
        [
            np.pad(
                loss, (0, max_epochs - len(loss)), "constant", constant_values=np.nan
            )
            for loss in train_losses
        ]
    )
    mean_train_loss = np.nanmean(aggregated_train_losses, axis=0)
    std_err_train_loss = np.nanstd(aggregated_train_losses, axis=0) / np.sqrt(
        len(train_losses)
    )

    plt.figure()
    epochs = range(max_epochs)
    plt.plot(epochs, mean_train_loss, label="Mean Training Loss")
    plt.fill_between(
        epochs,
        mean_train_loss - std_err_train_loss,
        mean_train_loss + std_err_train_loss,
        alpha=0.3,
    )
    plt.title("Mean Training Loss with Standard Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "synthetic_dynamic_network_mean_training_loss.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating mean training loss plot: {e}")
    plt.close()

try:
    # Aggregate validation F1 scores
    val_f1_scores = [
        data["hyperparam_tuning_epochs"]["synthetic_dynamic_network"]["metrics"]["val"]
        for data in all_experiment_data
    ]
    max_epochs = max(len(f1_score) for f1_score in val_f1_scores)
    aggregated_val_f1_scores = np.array(
        [
            np.pad(
                f1_score,
                (0, max_epochs - len(f1_score)),
                "constant",
                constant_values=np.nan,
            )
            for f1_score in val_f1_scores
        ]
    )
    mean_val_f1_score = np.nanmean(aggregated_val_f1_scores, axis=0)
    std_err_val_f1_score = np.nanstd(aggregated_val_f1_scores, axis=0) / np.sqrt(
        len(val_f1_scores)
    )

    plt.figure()
    plt.plot(epochs, mean_val_f1_score, label="Mean Validation F1 Score")
    plt.fill_between(
        epochs,
        mean_val_f1_score - std_err_val_f1_score,
        mean_val_f1_score + std_err_val_f1_score,
        alpha=0.3,
    )
    plt.title("Mean Validation F1 Score with Standard Error")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dynamic_network_mean_val_f1.png"))
    plt.close()
except Exception as e:
    print(f"Error creating mean validation F1 score plot: {e}")
    plt.close()
