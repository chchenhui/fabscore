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

for dsname in ["mnist", "fashion_mnist", "svhn"]:
    # Plot 1: Train/Val Loss
    try:
        plt.figure(figsize=(7, 5))
        epochs = experiment_data["freeze_vision"][dsname]["epochs"]
        train_loss = experiment_data["freeze_vision"][dsname]["losses"]["train"]
        val_loss = experiment_data["freeze_vision"][dsname]["losses"]["val"]
        plt.plot(epochs, train_loss, label="Train Loss", color="b")
        plt.plot(epochs, val_loss, label="Val Loss", color="r")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"FreezeVision Loss Curves ({dsname})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, f"{dsname}_loss_freezevision.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dsname}: {e}")
        plt.close()
    # Plot 2: Train/Val Accuracy
    try:
        plt.figure(figsize=(7, 5))
        epochs = experiment_data["freeze_vision"][dsname]["epochs"]
        train_acc = experiment_data["freeze_vision"][dsname]["metrics"]["train"]
        val_acc = experiment_data["freeze_vision"][dsname]["metrics"]["val"]
        plt.plot(epochs, train_acc, label="Train Acc", color="b")
        plt.plot(epochs, val_acc, label="Val Acc", color="r")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"FreezeVision Accuracy Curves ({dsname})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dsname}_acc_freezevision_trainval.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for {dsname}: {e}")
        plt.close()
    # Plot 3: Train/Val Logic Consistency Accuracy
    try:
        plt.figure(figsize=(7, 5))
        epochs = experiment_data["freeze_vision"][dsname]["epochs"]
        train_logic = experiment_data["freeze_vision"][dsname]["metrics"]["train_logic"]
        val_logic = experiment_data["freeze_vision"][dsname]["metrics"]["val_logic"]
        plt.plot(epochs, train_logic, label="Train Logical Consistency", color="b")
        plt.plot(epochs, val_logic, label="Val Logical Consistency", color="r")
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.title(f"FreezeVision Logical Consistency Accuracy ({dsname})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{dsname}_logicacc_freezevision_trainval.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating logic accuracy plot for {dsname}: {e}")
        plt.close()
