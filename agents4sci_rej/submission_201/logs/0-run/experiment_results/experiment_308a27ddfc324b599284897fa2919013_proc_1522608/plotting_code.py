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
    experiment_data = {}

ablation_key = "random_claim_text"
datasets = ["mnist", "fashion_mnist", "svhn"]

# 1. Training/Validation Accuracy/Loss for each dataset
for ds in datasets:
    try:
        d = experiment_data[ablation_key][ds]
        epochs = d["epochs"]
        train_acc = d["metrics"]["train"]
        val_acc = d["metrics"]["val"]
        train_loss = d["losses"]["train"]
        val_loss = d["losses"]["val"]
        logic_acc = d["metrics"]["val_logic"]

        # Accuracy Curve
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{ds.upper()} (Random Claim Text)\nTrain/Val Accuracy vs. Epoch")
        plt.legend()
        plt.tight_layout()
        fname = f"{ablation_key}_{ds}_accuracy_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {ds}: {e}")
        plt.close()

    try:
        # Loss Curve
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("BCE Loss")
        plt.title(f"{ds.upper()} (Random Claim Text)\nTrain/Val Loss vs. Epoch")
        plt.legend()
        plt.tight_layout()
        fname = f"{ablation_key}_{ds}_loss_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds}: {e}")
        plt.close()

    try:
        # Logical Consistency Curve
        plt.figure()
        plt.plot(epochs, logic_acc, label="Val Logical Consistency Accuracy", color="g")
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.title(f"{ds.upper()} (Random Claim Text)\nLogical Consistency vs. Epoch")
        plt.legend()
        plt.tight_layout()
        fname = f"{ablation_key}_{ds}_logic_consistency_curve.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating logic consistency plot for {ds}: {e}")
        plt.close()

# 2. Comparison plot: Final Validation Accuracy and Logic Consistency Accuracy across datasets
try:
    val_accs = []
    logic_accs = []
    for ds in datasets:
        val_accs.append(experiment_data[ablation_key][ds]["metrics"]["val"][-1])
        logic_accs.append(experiment_data[ablation_key][ds]["metrics"]["val_logic"][-1])
    x = np.arange(len(datasets))
    width = 0.35
    plt.figure()
    plt.bar(x - width / 2, val_accs, width, label="Val Accuracy")
    plt.bar(x + width / 2, logic_accs, width, label="Logic Consistency")
    plt.ylabel("Accuracy")
    plt.xticks(x, [d.upper() for d in datasets])
    plt.title(
        "Final Validation Accuracy and Logical Consistency\n(Random Claim Text Ablation)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{ablation_key}_final_acc_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy comparison plot: {e}")
    plt.close()

# 3. Overlayed plot: Val accuracy and Logic acc per dataset
for ds in datasets:
    try:
        d = experiment_data[ablation_key][ds]
        epochs = d["epochs"]
        val_acc = d["metrics"]["val"]
        logic_acc = d["metrics"]["val_logic"]
        plt.figure()
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.plot(epochs, logic_acc, label="Logical Consistency Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{ds.upper()} (Random Claim Text)\nValidation vs. Logic Consistency Accuracy"
        )
        plt.legend()
        plt.tight_layout()
        fname = f"{ablation_key}_{ds}_val_vs_logic_overlay.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating overlay acc/logic plot for {ds}: {e}")
        plt.close()
