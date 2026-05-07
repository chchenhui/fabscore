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

# 1. Plot training/validation loss curves for all num_conv_layers configs
try:
    plt.figure(figsize=(10, 6))
    num_layer_options = list(experiment_data["num_conv_layers"].keys())
    for exp_key in num_layer_options:
        d = experiment_data["num_conv_layers"][exp_key]
        epochs = d["epochs"]
        train_loss = d["losses"]["train"]
        val_loss = d["losses"]["val"]
        n_layers = d["n_layers"]
        plt.plot(epochs, train_loss, label=f"Train ({n_layers} conv)")
        plt.plot(epochs, val_loss, linestyle="--", label=f"Val ({n_layers} conv)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "MNIST Claim Verification: Loss Curves\nTrain and Validation Loss Per Number of CNN Layers"
    )
    plt.legend()
    fname = os.path.join(working_dir, "mnist_claims_num_conv_layers_loss_curve.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# 2. Summary bar plot: final validation accuracy for each n_layers
try:
    plt.figure(figsize=(7, 5))
    layers = []
    accuracies = []
    for exp_key in experiment_data["num_conv_layers"]:
        d = experiment_data["num_conv_layers"][exp_key]
        n_layers = d["n_layers"]
        layers.append(str(n_layers))
        if len(d["metrics"]["val"]) > 0:
            acc = d["metrics"]["val"][-1]
            accuracies.append(acc)
        else:
            accuracies.append(0)
    plt.bar(layers, accuracies, color="skyblue")
    plt.xlabel("Number of CNN Conv Layers")
    plt.ylabel("Final Validation Accuracy")
    plt.title("Final Validation Accuracy vs CNN Depth\nDataset: MNIST Claims")
    fname = os.path.join(working_dir, "mnist_claims_num_conv_layers_final_val_acc.png")
    plt.savefig(fname)
    plt.close()
except Exception as e:
    print(f"Error creating final val accuracy bar plot: {e}")
    plt.close()

# 3. For each configuration with predictions and ground truth stored, plot confusion matrix for val set
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    for exp_key in experiment_data["num_conv_layers"]:
        d = experiment_data["num_conv_layers"][exp_key]
        y_pred = d.get("predictions", [])
        y_true = d.get("ground_truth", [])
        n_layers = d["n_layers"]
        if len(y_pred) and len(y_true):
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(values_format="d", cmap="Blues")
            plt.title(
                f"Confusion Matrix: MNIST Claims\n{n_layers} Conv Layers (Val Set)"
            )
            fname = os.path.join(
                working_dir, f"mnist_claims_confusion_matrix_{n_layers}_conv.png"
            )
            plt.savefig(fname)
            plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()
