import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Extract top-level relevant dicts
try:
    freeze_unfreeze_results = experiment_data["freeze_unfreeze_bert_encoder"]
    dataset_name = "mnist_claims"
    config_names = list(freeze_unfreeze_results.keys())
except Exception as e:
    print(f"Error accessing freeze/unfreeze experiment data: {e}")
    config_names = []

# 1. Plot Accuracy curves for each config
for config in config_names:
    try:
        m = freeze_unfreeze_results[config][dataset_name]
        plt.figure()
        plt.plot(m["epochs"], m["metrics"]["train_acc"], label="Train Accuracy")
        plt.plot(m["epochs"], m["metrics"]["val_acc"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_name} Train/Validation Accuracy\nBERT config: {config}")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(
            working_dir, f"{dataset_name}_{config}_accuracy_curve.png"
        )
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {config}: {e}")
        plt.close()

# 2. Plot Loss curves for each config
for config in config_names:
    try:
        m = freeze_unfreeze_results[config][dataset_name]
        plt.figure()
        plt.plot(m["epochs"], m["losses"]["train"], label="Train Loss")
        plt.plot(m["epochs"], m["losses"]["val"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset_name} Train/Validation Loss\nBERT config: {config}")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(working_dir, f"{dataset_name}_{config}_loss_curve.png")
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {config}: {e}")
        plt.close()

# 3. Bar chart: Final validation accuracy for each config
try:
    plt.figure()
    bar_vals = []
    for config in config_names:
        m = freeze_unfreeze_results[config][dataset_name]
        bar_vals.append(m["metrics"]["val_acc"][-1] if m["metrics"]["val_acc"] else 0.0)
    plt.bar(config_names, bar_vals)
    plt.ylabel("Final Validation Accuracy")
    plt.title(
        f"{dataset_name}: Final Validation Accuracy\nacross BERT Freezing Strategies"
    )
    plt.xticks(rotation=25)
    plt.tight_layout()
    save_path = os.path.join(working_dir, f"{dataset_name}_final_val_accuracy_bar.png")
    plt.savefig(save_path)
    plt.close()
except Exception as e:
    print(f"Error creating final validation accuracy bar chart: {e}")
    plt.close()
