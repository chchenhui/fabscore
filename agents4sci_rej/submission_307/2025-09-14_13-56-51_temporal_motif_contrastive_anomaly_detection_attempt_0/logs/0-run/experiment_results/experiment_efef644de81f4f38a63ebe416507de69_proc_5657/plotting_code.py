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

try:
    feature_counts = experiment_data["feature_count_ablation"][
        "synthetic_dynamic_network"
    ]["feature_counts"]
    epochs_list = experiment_data["feature_count_ablation"][
        "synthetic_dynamic_network"
    ]["epoch_settings"]
    losses = experiment_data["feature_count_ablation"]["synthetic_dynamic_network"][
        "losses"
    ]["train"]
    val_f1_scores = experiment_data["feature_count_ablation"][
        "synthetic_dynamic_network"
    ]["metrics"]["val"]

    for i, feature_count in enumerate(set(feature_counts)):
        plt.figure()
        epoch_indices = [
            idx for idx, fc in enumerate(feature_counts) if fc == feature_count
        ]
        interval = max(
            1, len(epoch_indices) // 5
        )  # Plot at most 5 figures per feature count
        selected_epochs = epoch_indices[::interval]

        plt.subplot(1, 2, 1)
        plt.plot(
            [epochs_list[idx] for idx in selected_epochs],
            [losses[idx] for idx in selected_epochs],
        )
        plt.title(f"Training Loss for Feature Count {feature_count}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(
            [epochs_list[idx] for idx in selected_epochs],
            [val_f1_scores[idx] for idx in selected_epochs],
        )
        plt.title(f"Validation F1 Score for Feature Count {feature_count}")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir,
                f"synthetic_dynamic_network_feature_{feature_count}_plot.png",
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating plots: {e}")
    plt.close()
