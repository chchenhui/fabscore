import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data_path_list = [
        "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_180ae9aa612f4149a933552c01758270_proc_2848/experiment_data.npy",
        "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_b9d73619bcd841b1b4d546dae07eb8f6_proc_2868/experiment_data.npy",
        "experiments/2025-09-14_13-56-51_temporal_motif_contrastive_anomaly_detection_attempt_0/logs/0-run/experiment_results/experiment_a337b829145446c08f7a14d28f3206ee_proc_2848/experiment_data.npy",
    ]
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
    all_train_losses = [
        exp_data["hyperparam_tuning_epochs"]["synthetic_dynamic_network"]["losses"][
            "train"
        ]
        for exp_data in all_experiment_data
    ]
    max_epochs = min([len(losses) for losses in all_train_losses])
    train_losses_mean = np.mean(
        [losses[:max_epochs] for losses in all_train_losses], axis=0
    )
    train_losses_std = np.std(
        [losses[:max_epochs] for losses in all_train_losses], axis=0
    )

    # Plot aggregated training losses
    plt.figure()
    plt.plot(train_losses_mean, label="Mean Training Loss")
    plt.fill_between(
        range(max_epochs),
        train_losses_mean - train_losses_std,
        train_losses_mean + train_losses_std,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Aggregated Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            working_dir, "aggregated_synthetic_dynamic_network_training_loss.png"
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated training loss plot: {e}")
    plt.close()

try:
    # Aggregate validation F1 scores
    all_val_scores = [
        exp_data["hyperparam_tuning_epochs"]["synthetic_dynamic_network"]["metrics"][
            "val"
        ]
        for exp_data in all_experiment_data
    ]
    max_epochs = min([len(scores) for scores in all_val_scores])
    val_scores_mean = np.mean(
        [scores[:max_epochs] for scores in all_val_scores], axis=0
    )
    val_scores_std = np.std([scores[:max_epochs] for scores in all_val_scores], axis=0)

    # Plot aggregated validation F1 scores
    plt.figure()
    plt.plot(val_scores_mean, label="Mean Validation F1 Score")
    plt.fill_between(
        range(max_epochs),
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.2,
        label="Standard Error",
    )
    plt.title("Aggregated Validation F1 Score over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(
        os.path.join(working_dir, "aggregated_synthetic_dynamic_network_val_f1.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated validation F1 score plot: {e}")
    plt.close()
