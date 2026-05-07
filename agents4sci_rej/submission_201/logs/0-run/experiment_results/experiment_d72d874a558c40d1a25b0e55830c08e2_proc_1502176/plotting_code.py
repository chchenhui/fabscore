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
    experiment_data = None

max_length_list = [16, 32, 64]
setting_names = [f"maxlen_{ml}" for ml in max_length_list]

# (1) Plot accuracy curves
try:
    plt.figure(figsize=(9, 6))
    for max_length in max_length_list:
        setting = f"maxlen_{max_length}"
        epochs = experiment_data["bert_max_length"][setting]["epochs"]
        train_acc = experiment_data["bert_max_length"][setting]["metrics"]["train_acc"]
        val_acc = experiment_data["bert_max_length"][setting]["metrics"]["val_acc"]
        plt.plot(epochs, train_acc, label=f"Train Acc (maxlen={max_length})")
        plt.plot(epochs, val_acc, "--", label=f"Val Acc (maxlen={max_length})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Val Accuracy (MNIST+Claim, BERT max_length Sweep)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(working_dir, "mnist_claims_maxlen_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# (2) Plot loss curves
try:
    plt.figure(figsize=(9, 6))
    for max_length in max_length_list:
        setting = f"maxlen_{max_length}"
        epochs = experiment_data["bert_max_length"][setting]["epochs"]
        train_loss = experiment_data["bert_max_length"][setting]["losses"]["train"]
        val_loss = experiment_data["bert_max_length"][setting]["losses"]["val"]
        plt.plot(epochs, train_loss, label=f"Train Loss (maxlen={max_length})")
        plt.plot(epochs, val_loss, "--", label=f"Val Loss (maxlen={max_length})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss (MNIST+Claim, BERT max_length Sweep)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(working_dir, "mnist_claims_maxlen_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# (3) Bar plot: final validation accuracy for each setting
try:
    plt.figure(figsize=(7, 5))
    final_accs = []
    for max_length in max_length_list:
        setting = f"maxlen_{max_length}"
        val_acc = experiment_data["bert_max_length"][setting]["metrics"]["val_acc"][-1]
        final_accs.append(val_acc)
    plt.bar([str(ml) for ml in max_length_list], final_accs, color="skyblue")
    for idx, acc in enumerate(final_accs):
        plt.text(idx, acc + 0.01, f"{acc:.3f}", ha="center", size=10)
    plt.ylim(0, 1)
    plt.xlabel("BERT max_length")
    plt.ylabel("Final Validation Accuracy")
    plt.title("Final Val Accuracy by max_length (MNIST+Claim)")
    plt.savefig(os.path.join(working_dir, "mnist_claims_maxlen_final_val_acc_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()
