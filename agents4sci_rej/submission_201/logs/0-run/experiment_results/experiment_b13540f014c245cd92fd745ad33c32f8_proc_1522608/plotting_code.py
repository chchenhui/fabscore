import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# 1: Training and validation loss curves per dataset
for dsname in ["mnist", "fashion_mnist", "svhn"]:
    try:
        epochs = experiment_data["permute_order_ablation"][dsname]["epochs"]
        train_loss = experiment_data["permute_order_ablation"][dsname]["losses"][
            "train"
        ]
        val_loss = experiment_data["permute_order_ablation"][dsname]["losses"]["val"]
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"{dsname.upper()} - Loss Curves (Permuted Order Ablation)")
        plt.tight_layout()
        fname = f"{dsname}_loss_curves_permuted_ablation.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve plot for {dsname}: {e}")
        plt.close()

# 2: Training and validation accuracy per dataset
for dsname in ["mnist", "fashion_mnist", "svhn"]:
    try:
        epochs = experiment_data["permute_order_ablation"][dsname]["epochs"]
        train_acc = experiment_data["permute_order_ablation"][dsname]["metrics"][
            "train"
        ]
        val_acc = experiment_data["permute_order_ablation"][dsname]["metrics"]["val"]
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"{dsname.upper()} - Accuracy Curves (Permuted Order Ablation)")
        plt.tight_layout()
        fname = f"{dsname}_accuracy_curves_permuted_ablation.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve plot for {dsname}: {e}")
        plt.close()

# 3: Logical consistency accuracy per dataset
for dsname in ["mnist", "fashion_mnist", "svhn"]:
    try:
        epochs = experiment_data["permute_order_ablation"][dsname]["epochs"]
        logic_acc_train = experiment_data["permute_order_ablation"][dsname]["metrics"][
            "train_logic"
        ]
        logic_acc_val = experiment_data["permute_order_ablation"][dsname]["metrics"][
            "val_logic"
        ]
        plt.figure()
        plt.plot(epochs, logic_acc_train, label="Train Logical Consistency Accuracy")
        plt.plot(epochs, logic_acc_val, label="Validation Logical Consistency Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.legend()
        plt.title(f"{dsname.upper()} - Logical Consistency (Permuted Order Ablation)")
        plt.tight_layout()
        fname = f"{dsname}_logic_acc_curves_permuted_ablation.png"
        plt.savefig(os.path.join(working_dir, fname))
        plt.close()
    except Exception as e:
        print(f"Error creating logic consistency plot for {dsname}: {e}")
        plt.close()

# 4: Comparison plots across datasets (Validation Accuracy and Logical Consistency Accuracy)
try:
    colors = {"mnist": "b", "fashion_mnist": "g", "svhn": "r"}
    plt.figure()
    for dsname in ["mnist", "fashion_mnist", "svhn"]:
        epochs = experiment_data["permute_order_ablation"][dsname]["epochs"]
        val_acc = experiment_data["permute_order_ablation"][dsname]["metrics"]["val"]
        plt.plot(epochs, val_acc, label=f"{dsname}", color=colors[dsname])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.title("Validation Accuracy Comparison (Permuted Order Ablation)")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_acc_compare_permuted_ablation.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation accuracy comparison plot: {e}")
    plt.close()

try:
    plt.figure()
    for dsname in ["mnist", "fashion_mnist", "svhn"]:
        epochs = experiment_data["permute_order_ablation"][dsname]["epochs"]
        val_logic = experiment_data["permute_order_ablation"][dsname]["metrics"][
            "val_logic"
        ]
        plt.plot(epochs, val_logic, label=f"{dsname}", color=colors[dsname])
    plt.xlabel("Epoch")
    plt.ylabel("Validation Logical Consistency Accuracy")
    plt.legend()
    plt.title("Logical Consistency Accuracy Comparison (Permuted Order Ablation)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "val_logic_acc_compare_permuted_ablation.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating logic consistency comparison plot: {e}")
    plt.close()
