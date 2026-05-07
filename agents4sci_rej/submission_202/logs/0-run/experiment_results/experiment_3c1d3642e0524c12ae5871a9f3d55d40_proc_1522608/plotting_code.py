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

CLAIM_TYPES = [
    "sum_even",
    "all_lt_5",
    "exactly_two_odd",
    "at_least_one_is_7",
    "all_unique",
]
DATASETS = ["mnist", "fashion_mnist", "svhn"]
ablation_name = "one_claim_training_only"

# Plot: Validation Accuracy (all claims per dataset, overlaid)
try:
    plt.figure(figsize=(10, 7))
    colors = ["b", "r", "g", "c", "m"]
    for i, dsname in enumerate(DATASETS):
        for j, claim_type in enumerate(CLAIM_TYPES):
            key = f"{dsname}_{claim_type}"
            if key not in experiment_data.get(ablation_name, {}):
                continue
            epochs = experiment_data[ablation_name][key]["epochs"]
            val_acc = experiment_data[ablation_name][key]["metrics"]["val"]
            plt.plot(
                epochs,
                val_acc,
                color=colors[j % len(colors)],
                linestyle=["-", "--", ":", "-.", (0, (3, 2, 1, 2))][j % 5],
                label=f"{dsname} ({claim_type})",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy (Per-Claim, No Joint Training)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "all_datasets_val_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot: val_acc: {e}")
    plt.close()

# Plot: Logical Consistency (all claims per dataset, overlaid)
try:
    plt.figure(figsize=(10, 7))
    colors = ["b", "r", "g", "c", "m"]
    for i, dsname in enumerate(DATASETS):
        for j, claim_type in enumerate(CLAIM_TYPES):
            key = f"{dsname}_{claim_type}"
            if key not in experiment_data.get(ablation_name, {}):
                continue
            epochs = experiment_data[ablation_name][key]["epochs"]
            logic_acc = experiment_data[ablation_name][key]["metrics"]["val_logic"]
            plt.plot(
                epochs,
                logic_acc,
                color=colors[j % len(colors)],
                linestyle=["-", "--", ":", "-.", (0, (3, 2, 1, 2))][j % 5],
                label=f"{dsname} ({claim_type})",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Logical Consistency")
    plt.title("Logical Consistency (Per-Claim, No Joint Training)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "all_datasets_val_logic_acc.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot: val_logic_acc: {e}")
    plt.close()

# For each (dataset, claim): Validation vs Logic accuracy curves
for dsname in DATASETS:
    for claim_type in CLAIM_TYPES:
        try:
            key = f"{dsname}_{claim_type}"
            if (
                ablation_name not in experiment_data
                or key not in experiment_data[ablation_name]
            ):
                continue
            epochs = experiment_data[ablation_name][key]["epochs"]
            val_acc = experiment_data[ablation_name][key]["metrics"]["val"]
            logic_acc = experiment_data[ablation_name][key]["metrics"]["val_logic"]
            plt.figure()
            plt.plot(epochs, val_acc, label="Val Accuracy")
            plt.plot(epochs, logic_acc, label="Logical Consistency")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{dsname.capitalize()} ({claim_type}) - Accuracies per Epoch",
                fontsize=11,
            )
            plt.legend()
            fname = f"{dsname}_{claim_type}_perclaim_acc.png"
            plt.tight_layout()
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating plot {dsname}, {claim_type}: {e}")
            plt.close()

# Print final logic accuracy for each (dataset, claim)
for dsname in DATASETS:
    for claim_type in CLAIM_TYPES:
        try:
            key = f"{dsname}_{claim_type}"
            if (
                ablation_name not in experiment_data
                or key not in experiment_data[ablation_name]
            ):
                continue
            logic = experiment_data[ablation_name][key]["metrics"]["val_logic"][-1]
            print(f"Final Logic Acc ({dsname}, {claim_type}): {logic:.4f}")
        except Exception as e:
            print(f"Error printing logic acc for {dsname}, {claim_type}: {e}")
