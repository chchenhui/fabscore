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

datasets = ["mnist", "fashion_mnist", "svhn"]
abl_key = "no_token_type_embeddings"

# 1. Accuracy curves (training and validation) per dataset
for dsname in datasets:
    try:
        epochs = experiment_data[abl_key][dsname]["epochs"]
        train_acc = experiment_data[abl_key][dsname]["metrics"]["train"]
        val_acc = experiment_data[abl_key][dsname]["metrics"]["val"]
        plt.figure()
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{dsname.upper()} (No TokenType Embeddings): Accuracy Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dsname}_acc_curve_no_token_type.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy curve for {dsname}: {e}")
        plt.close()

# 2. Loss curves (training and validation) per dataset
for dsname in datasets:
    try:
        epochs = experiment_data[abl_key][dsname]["epochs"]
        train_loss = experiment_data[abl_key][dsname]["losses"]["train"]
        val_loss = experiment_data[abl_key][dsname]["losses"]["val"]
        plt.figure()
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dsname.upper()} (No TokenType Embeddings): Loss Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dsname}_loss_curve_no_token_type.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curve for {dsname}: {e}")
        plt.close()

# 3. Logical consistency accuracy (training and validation) per dataset
for dsname in datasets:
    try:
        epochs = experiment_data[abl_key][dsname]["epochs"]
        train_logic = experiment_data[abl_key][dsname]["metrics"]["train_logic"]
        val_logic = experiment_data[abl_key][dsname]["metrics"]["val_logic"]
        plt.figure()
        plt.plot(epochs, train_logic, label="Train Logic Acc")
        plt.plot(epochs, val_logic, label="Val Logic Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.title(f"{dsname.upper()} (No TokenType Embeddings): Logic Accuracy Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dsname}_logic_acc_curve_no_token_type.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating logic acc curve for {dsname}: {e}")
        plt.close()

# 4. Combined summary plots (validation accuracy and logic acc) across datasets
try:
    plt.figure()
    for dsname in datasets:
        epochs = experiment_data[abl_key][dsname]["epochs"]
        val_acc = experiment_data[abl_key][dsname]["metrics"]["val"]
        plt.plot(epochs, val_acc, label=f"{dsname}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison (No TokenType Embeddings)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_acc_compare_no_token_type.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val acc compare plot: {e}")
    plt.close()

try:
    plt.figure()
    for dsname in datasets:
        epochs = experiment_data[abl_key][dsname]["epochs"]
        val_logic = experiment_data[abl_key][dsname]["metrics"]["val_logic"]
        plt.plot(epochs, val_logic, label=f"{dsname}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Logical Consistency Accuracy")
    plt.title("Validation Logic Consistency Comparison (No TokenType Embeddings)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "val_logic_acc_compare_no_token_type.png"))
    plt.close()
except Exception as e:
    print(f"Error creating val logic acc compare plot: {e}")
    plt.close()
