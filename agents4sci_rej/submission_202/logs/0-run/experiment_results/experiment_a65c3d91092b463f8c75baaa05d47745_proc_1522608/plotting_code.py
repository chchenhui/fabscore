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

dslist = ["mnist", "fashion_mnist", "svhn"]

# 1. Training/validation accuracy curves for each dataset
for ds in dslist:
    try:
        plt.figure()
        epochs = experiment_data["claim_diversity_ablation"][ds]["epochs"]
        tr = experiment_data["claim_diversity_ablation"][ds]["metrics"]["train"]
        va = experiment_data["claim_diversity_ablation"][ds]["metrics"]["val"]
        plt.plot(epochs, tr, label="Train Accuracy")
        plt.plot(epochs, va, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{ds.capitalize()} (Claim Diversity Ablation)\nTraining vs Validation Accuracy"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_train_val_acc_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating train/val acc plot for {ds}: {e}")
        plt.close()

# 2. Training/validation logical consistency accuracy curves for each dataset
for ds in dslist:
    try:
        plt.figure()
        epochs = experiment_data["claim_diversity_ablation"][ds]["epochs"]
        trl = experiment_data["claim_diversity_ablation"][ds]["metrics"]["train_logic"]
        val = experiment_data["claim_diversity_ablation"][ds]["metrics"]["val_logic"]
        plt.plot(epochs, trl, label="Train Logic Acc")
        plt.plot(epochs, val, label="Val Logic Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.title(f"{ds.capitalize()} (Claim Diversity Ablation)\nLogic Accuracies")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_logic_acc_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating logic acc plot for {ds}: {e}")
        plt.close()

# 3. Training/validation loss curves for each dataset
for ds in dslist:
    try:
        plt.figure()
        epochs = experiment_data["claim_diversity_ablation"][ds]["epochs"]
        trl = experiment_data["claim_diversity_ablation"][ds]["losses"]["train"]
        val = experiment_data["claim_diversity_ablation"][ds]["losses"]["val"]
        plt.plot(epochs, trl, label="Train Loss")
        plt.plot(epochs, val, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(
            f"{ds.capitalize()} (Claim Diversity Ablation)\nTraining vs Validation Loss"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_train_val_loss_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating train/val loss plot for {ds}: {e}")
        plt.close()

# 4. Compare validation accuracy across datasets
try:
    plt.figure()
    for ds in dslist:
        epochs = experiment_data["claim_diversity_ablation"][ds]["epochs"]
        va = experiment_data["claim_diversity_ablation"][ds]["metrics"]["val"]
        plt.plot(epochs, va, label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy (Claim Diversity Ablation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_acc_compare_ablation.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation accuracy comparison plot: {e}")
    plt.close()

# 5. Compare validation logic accuracy across datasets
try:
    plt.figure()
    for ds in dslist:
        epochs = experiment_data["claim_diversity_ablation"][ds]["epochs"]
        va = experiment_data["claim_diversity_ablation"][ds]["metrics"]["val_logic"]
        plt.plot(epochs, va, label=ds)
    plt.xlabel("Epoch")
    plt.ylabel("Logical Consistency Accuracy")
    plt.title("Logical Consistency Accuracy (Claim Diversity Ablation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_logic_acc_compare_ablation.png"))
    plt.close()
except Exception as e:
    print(f"Error creating logical accuracy comparison plot: {e}")
    plt.close()

# 6. Overlay val acc and val logical acc for each dataset (already required in prompt)
for ds in dslist:
    try:
        plt.figure()
        epochs = experiment_data["claim_diversity_ablation"][ds]["epochs"]
        va = experiment_data["claim_diversity_ablation"][ds]["metrics"]["val"]
        logva = experiment_data["claim_diversity_ablation"][ds]["metrics"]["val_logic"]
        plt.plot(epochs, va, label="Val Acc")
        plt.plot(epochs, logva, label="Logic Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{ds.capitalize()} (Ablation) - Accuracies per Epoch\nLeft: Validation Accuracy, Right: Logical Consistency Accuracy"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{ds}_acc_ablation.png"))
        plt.close()
    except Exception as e:
        print(f"Error overlaying logic/val acc for {ds}: {e}")
        plt.close()
