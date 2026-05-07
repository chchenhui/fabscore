import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

# Only proceed if data loaded
if experiment_data is not None:
    datasets = ["mnist", "fashion_mnist", "svhn"]
    metrics = ["train", "val", "train_logic", "val_logic"]
    losses = ["train", "val"]

    # 1. For each dataset: plot training and validation accuracy curves
    for ds in datasets:
        try:
            plt.figure()
            epochs = experiment_data["no_tokenization"][ds]["epochs"]
            for m, label in [
                ("train", "Train Accuracy"),
                ("val", "Validation Accuracy"),
            ]:
                vals = experiment_data["no_tokenization"][ds]["metrics"][m]
                if len(vals):
                    plt.plot(epochs, vals, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds.upper()} - Training/Validation Accuracy\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_train_val_acc_ablation.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating accuracy plot for {ds}: {e}")
            plt.close()

    # 2. For each dataset: plot logic consistency curves (train and val)
    for ds in datasets:
        try:
            plt.figure()
            epochs = experiment_data["no_tokenization"][ds]["epochs"]
            for m, label in [
                ("train_logic", "Train Logic Consistency"),
                ("val_logic", "Validation Logic Consistency"),
            ]:
                vals = experiment_data["no_tokenization"][ds]["metrics"][m]
                if len(vals):
                    plt.plot(epochs, vals, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Logic Consistency Accuracy")
            plt.title(
                f"{ds.upper()} - Logic Consistency Accuracies\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_logic_acc_ablation.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating logic consistency plot for {ds}: {e}")
            plt.close()

    # 3. For each dataset: plot training and validation loss curves
    for ds in datasets:
        try:
            plt.figure()
            epochs = experiment_data["no_tokenization"][ds]["epochs"]
            for lkey, label in [("train", "Train Loss"), ("val", "Validation Loss")]:
                vals = experiment_data["no_tokenization"][ds]["losses"][lkey]
                if len(vals):
                    plt.plot(epochs, vals, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{ds.upper()} - Training/Validation Loss\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_loss_curve_ablation.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating loss curve plot for {ds}: {e}")
            plt.close()

    # 4. Overlay plot for each dataset: compare val acc and logic acc
    for ds in datasets:
        try:
            plt.figure()
            epochs = experiment_data["no_tokenization"][ds]["epochs"]
            val_acc = experiment_data["no_tokenization"][ds]["metrics"]["val"]
            val_logic = experiment_data["no_tokenization"][ds]["metrics"]["val_logic"]
            if len(val_acc) and len(val_logic):
                plt.plot(epochs, val_acc, label="Validation Accuracy")
                plt.plot(epochs, val_logic, label="Validation Logic Consistency")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds.upper()} - Validation vs Logical-Consistency Accuracy\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_val_and_logic_acc_overlay_ablation.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error creating overlay acc/logic plot for {ds}: {e}")
            plt.close()
