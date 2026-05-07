import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from collections import defaultdict

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

# Load .npy data
try:
    experiment_data_path = "experiment_results/experiment_726b2721d45c4800b9381c5d265fefcb_proc_332087/experiment_data.npy"
    experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Define a function to load confusion matrices
def load_confusion_matrix(file_path):
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"Error loading confusion matrix from {file_path}: {e}")
        return None

# Load confusion matrices
cm_BPI2012 = load_confusion_matrix("experiment_results/experiment_726b2721d45c4800b9381c5d265fefcb_proc_332087/cm_BPI2012.npy")
cm_ROAD = load_confusion_matrix("experiment_results/experiment_726b2721d45c4800b9381c5d265fefcb_proc_332087/cm_ROAD.npy")
cm_BPI2017 = load_confusion_matrix("experiment_results/experiment_726b2721d45c4800b9381c5d265fefcb_proc_332087/cm_BPI2017.npy")

# Plot loss curves for each dataset
def plot_loss_curves(experiment_data, dataset_name):
    try:
        train_loss = [y for (_, y) in experiment_data.get("losses", {}).get("train", [])]
        val_loss = [y for (_, y) in experiment_data.get("losses", {}).get("val", [])]
        plt.figure(dpi=300)
        if train_loss:
            plt.plot(train_loss, label="Train Loss")
        if val_loss:
            plt.plot(val_loss, label="Validation Loss")
        plt.title(f"Loss Curves - {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/{dataset_name}_loss_curves.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting loss curves for {dataset_name}: {e}")

# Plot confusion matrix for each dataset
def plot_confusion_matrix(cm, dataset_name):
    try:
        plt.figure(dpi=300, figsize=(6, 5))
        plt.imshow(cm, aspect="auto", cmap="Blues")
        plt.colorbar()
        plt.title(f"Confusion Matrix - {dataset_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"figures/{dataset_name}_confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix for {dataset_name}: {e}")

# Plot Top-3 accuracy vs prefix length for each dataset
def plot_top3_vs_prefix_length(experiment_data, dataset_name):
    try:
        pref_lens = experiment_data.get("prefix_lens", [])
        top3_flags = experiment_data.get("top3_flags", [])
        if pref_lens and top3_flags:
            d = defaultdict(list)
            for L, flag in zip(pref_lens, top3_flags):
                d[int(L)].append(int(flag))
            xs = sorted(d.keys())
            ys = [np.mean(d[k]) for k in xs]
            plt.figure(dpi=300)
            plt.plot(xs, ys, marker="o")
            plt.title(f"Top-3 Accuracy vs Prefix Length - {dataset_name}")
            plt.xlabel("Prefix Length")
            plt.ylabel("Top-3 Accuracy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"figures/{dataset_name}_top3_vs_prefixlen.png")
            plt.close()
    except Exception as e:
        print(f"Error plotting Top-3 vs prefix length for {dataset_name}: {e}")

# Plot macro precision-recall curve for each dataset
def plot_macro_pr(experiment_data, dataset_name):
    try:
        probs = np.array(experiment_data.get("probs", []))
        y_true = experiment_data.get("ground_truth", [])
        if probs.size > 0 and y_true:
            classes = sorted(set(y_true))
            Y = label_binarize(np.array(y_true), classes=range(probs.shape[1]))
            precisions = []
            aps = []
            grid = np.linspace(0, 1, 101)
            for c in classes:
                p, r, _ = precision_recall_curve(Y[:, c], probs[:, c])
                precisions.append(np.interp(grid, r[::-1], p[::-1]))
                aps.append(average_precision_score(Y[:, c], probs[:, c]))
            macro_p = np.mean(np.stack(precisions, 0), 0)
            plt.figure(dpi=300)
            plt.plot(grid, macro_p, label=f"mAP={np.mean(aps):.3f}")
            plt.title(f"Macro Precision-Recall (Test) - {dataset_name}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"figures/{dataset_name}_macro_pr.png")
            plt.close()
    except Exception as e:
        print(f"Error plotting macro precision-recall for {dataset_name}: {e}")

# Additional plots for detailed analysis
def plot_additional_analysis(experiment_data, dataset_name):
    # Example: Histogram of prefix lengths
    try:
        prefix_lengths = experiment_data.get("prefix_lens", [])
        plt.figure(dpi=300)
        plt.hist(prefix_lengths, bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Prefix Length Distribution - {dataset_name}")
        plt.xlabel("Prefix Length")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/{dataset_name}_prefix_length_distribution.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting prefix length distribution for {dataset_name}: {e}")

# Main plotting function
def main():
    for dataset_name, data in experiment_data.items():
        plot_loss_curves(data, dataset_name)
        if dataset_name == "BPI2012" and cm_BPI2012 is not None:
            plot_confusion_matrix(cm_BPI2012, dataset_name)
        elif dataset_name == "ROAD" and cm_ROAD is not None:
            plot_confusion_matrix(cm_ROAD, dataset_name)
        elif dataset_name == "BPI2017" and cm_BPI2017 is not None:
            plot_confusion_matrix(cm_BPI2017, dataset_name)
        plot_top3_vs_prefix_length(data, dataset_name)
        plot_macro_pr(data, dataset_name)
        plot_additional_analysis(data, dataset_name)

if __name__ == "__main__":
    main()