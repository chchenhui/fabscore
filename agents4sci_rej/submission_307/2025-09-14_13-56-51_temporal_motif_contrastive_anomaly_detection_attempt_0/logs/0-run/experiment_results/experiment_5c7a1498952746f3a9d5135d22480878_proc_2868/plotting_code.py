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

learning_rates = experiment_data["learning_rate_tuning"]["synthetic_dynamic_network"][
    "metrics"
].keys()

for lr in learning_rates:
    try:
        train_losses = experiment_data["learning_rate_tuning"][
            "synthetic_dynamic_network"
        ]["losses"][lr]["train"]
        val_f1_scores = experiment_data["learning_rate_tuning"][
            "synthetic_dynamic_network"
        ]["metrics"][lr]["val"]
        epochs = range(1, len(train_losses) + 1)

        # Plot training loss
        plt.figure()
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.title(f"Training Loss for Learning Rate {lr}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            os.path.join(
                working_dir, f"synthetic_dynamic_network_train_loss_lr_{lr}.png"
            )
        )
        plt.close()

        # Plot validation F1 score
        plt.figure()
        plt.plot(epochs, val_f1_scores, label="Validation F1 Score")
        plt.title(f"Validation F1 Score for Learning Rate {lr}")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.savefig(
            os.path.join(working_dir, f"synthetic_dynamic_network_val_f1_lr_{lr}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plots for learning rate {lr}: {e}")
        plt.close()
