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

# Names and quick validation
try:
    aug_tuning = experiment_data["augmentation_tuning"]
    aug_names = list(aug_tuning.keys())
except Exception as e:
    print("Error extracting augmentation_tuning:", e)

# (1) Validation curves for all augmentations (already in original code, but plot again with full explicit subtitle)
try:
    plt.figure(figsize=(10, 6))
    for aug_name in aug_names:
        ep = aug_tuning[aug_name]["epochs"]
        val_acc = aug_tuning[aug_name]["metrics"]["val"]
        plt.plot(ep, val_acc, label=aug_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Validation Accuracy for Different Augmentation Schemes\nDataset: MNISTClaim"
    )
    plt.legend()
    plt.grid()
    plot_path = os.path.join(
        working_dir, "mnistclaim_augmentation_val_accuracy_all_schemes.png"
    )
    plt.savefig(plot_path)
    plt.close()
except Exception as e:
    print(f"Error creating val acc summary plot: {e}")
    plt.close()

# (2) Training vs Validation accuracy - only for top 3 best-performing augmentations (by final val acc)
try:
    # Find top 3
    final_acc = [(aug, aug_tuning[aug]["metrics"]["val"][-1]) for aug in aug_names]
    final_acc_sorted = sorted(final_acc, key=lambda x: x[1], reverse=True)
    for i, (aug_name, val_acc) in enumerate(final_acc_sorted[:3]):
        ep = aug_tuning[aug_name]["epochs"]
        tr_acc = aug_tuning[aug_name]["metrics"]["train"]
        val_accs = aug_tuning[aug_name]["metrics"]["val"]
        plt.figure()
        plt.plot(ep, tr_acc, "o-", label="Train")
        plt.plot(ep, val_accs, "s-", label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{aug_name}: Train vs Val Accuracy Curves\nDataset: MNISTClaim")
        plt.legend()
        plt.grid()
        outname = os.path.join(
            working_dir, f"mnistclaim_train_val_curve_{aug_name}.png"
        )
        plt.savefig(outname)
        plt.close()
except Exception as e:
    print(f"Error creating train/val curves: {e}")
    plt.close()

# (3) Final accuracy bar plot for all augmentations
try:
    plt.figure(figsize=(10, 5))
    final_val_accs = [aug_tuning[aug]["metrics"]["val"][-1] for aug in aug_names]
    plt.bar(aug_names, final_val_accs, color="skyblue")
    plt.ylabel("Final Validation Accuracy")
    plt.xlabel("Augmentation Setting")
    plt.title("Final Validation Accuracy by Augmentation Scheme\nDataset: MNISTClaim")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(os.path.join(working_dir, "mnistclaim_final_val_acc_bar.png"))
    plt.close()
except Exception as e:
    print(f"Error creating final accuracy bar plot: {e}")
    plt.close()

# (4) If available, plot confusion matrix for the best augmentation setting (using predictions and ground_truth)
try:
    # Find best setting
    best_aug = max(aug_names, key=lambda k: aug_tuning[k]["metrics"]["val"][-1])
    preds = aug_tuning[best_aug].get("predictions", None)
    gts = aug_tuning[best_aug].get("ground_truth", None)
    if preds is not None and gts is not None and len(preds) == len(gts):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        cm = confusion_matrix(gts, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=["False", "True"])
        disp.plot(cmap="Blues")
        plt.title(
            f"Confusion Matrix - Best Augmentation ({best_aug})\nDataset: MNISTClaim"
        )
        plt.savefig(
            os.path.join(working_dir, f"mnistclaim_confusion_matrix_{best_aug}.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()
