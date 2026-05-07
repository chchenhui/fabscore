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
    if not isinstance(experiment_data, dict):
        raise ValueError("experiment_data.npy did not contain a dictionary.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

# Print metrics and plot
from sklearn.metrics import confusion_matrix
import itertools

for ds_name, res in experiment_data.items():
    print(f"\n=== {ds_name} Metrics ===")
    try:
        test_metrics = dict(res.get("metrics", {}).get("test", [("final", {})])[0][1])
        print("Test metrics:")
        for k, v in test_metrics.items():
            print(f" - {k}: {v:.4f}")
    except Exception as e:
        print(f"Could not print metrics for {ds_name}: {e}")

    # 1) Train vs Val loss
    try:
        train_loss = np.array(res.get("losses", {}).get("train", []))
        val_loss = np.array(res.get("losses", {}).get("val", []))
        if train_loss.size > 0 or val_loss.size > 0:
            plt.figure()
            if train_loss.size > 0:
                plt.plot(train_loss[:, 0], train_loss[:, 1], label="Train Loss")
            if val_loss.size > 0:
                plt.plot(val_loss[:, 0], val_loss[:, 1], label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{ds_name} Training and Validation Loss\nSubtitle: Time-based split, Next-Activity task"
            )
            plt.legend()
            out = os.path.join(working_dir, f"{ds_name}_train_val_loss.png")
            plt.savefig(out)
            plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {ds_name}: {e}")
        try:
            plt.close()
        except:
            pass

    # 2) Validation metric curves (acc, f1, top3, ece)
    try:
        val_curves = res.get("val_curves", {})
        have_any = any(
            len(val_curves.get(k, [])) > 0 for k in ["acc", "f1", "top3", "ece"]
        )
        if have_any:
            plt.figure()
            for key, label in [
                ("acc", "Val Accuracy"),
                ("f1", "Val Macro-F1"),
                ("top3", "Val Top-3"),
                ("ece", "Val ECE"),
            ]:
                arr = np.array(val_curves.get(key, []))
                if arr.size > 0:
                    plt.plot(arr[:, 0], arr[:, 1], label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Metric value")
            plt.title(
                f"{ds_name} Validation Curves\nSubtitle: Acc/F1/Top-3/ECE (Next-Activity)"
            )
            plt.legend()
            out = os.path.join(working_dir, f"{ds_name}_val_curves.png")
            plt.savefig(out)
            plt.close()
    except Exception as e:
        print(f"Error creating validation curves for {ds_name}: {e}")
        try:
            plt.close()
        except:
            pass

    # 3) Confusion matrix (test)
    try:
        cm_path = os.path.join(working_dir, f"{ds_name}_confusion_matrix.npy")
        cm = None
        if os.path.exists(cm_path):
            cm = np.load(cm_path)
        else:
            y_true = np.array(res.get("ground_truth", []))
            y_pred = np.array(res.get("predictions", []))
            if y_true.size > 0 and y_pred.size > 0:
                cm = confusion_matrix(y_true, y_pred)
        if cm is not None and cm.size > 0:
            plt.figure()
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(
                f"{ds_name} Confusion Matrix (Test)\nSubtitle: Next-Activity classification"
            )
            plt.colorbar()
            tick_marks = np.arange(min(20, cm.shape[0]))
            # For readability, limit ticks if too many classes
            if cm.shape[0] <= 20:
                plt.xticks(np.arange(cm.shape[1]))
                plt.yticks(np.arange(cm.shape[0]))
            else:
                plt.xticks(tick_marks)
                plt.yticks(tick_marks)
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            out = os.path.join(working_dir, f"{ds_name}_confusion_matrix.png")
            plt.savefig(out)
            plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix for {ds_name}: {e}")
        try:
            plt.close()
        except:
            pass

    # 4) Validation Top-3 zoomed (if available) to satisfy task-specific plot
    try:
        top3 = np.array(res.get("val_curves", {}).get("top3", []))
        if top3.size > 0:
            plt.figure()
            plt.plot(top3[:, 0], top3[:, 1], label="Val Top-3 Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Top-3 Accuracy")
            plt.title(
                f"{ds_name} Val Top-3 Accuracy over Epochs\nSubtitle: Next-Activity task"
            )
            plt.legend()
            out = os.path.join(working_dir, f"{ds_name}_val_top3.png")
            plt.savefig(out)
            plt.close()
    except Exception as e:
        print(f"Error creating top-3 plot for {ds_name}: {e}")
        try:
            plt.close()
        except:
            pass

# Also try to include any pre-saved per-dataset arrays if present (generic loader)
for ds_name in experiment_data.keys():
    try:
        # Already printed; nothing extra beyond constraints
        pass
    except Exception as e:
        print(f"Extra plotting error for {ds_name}: {e}")

# Print a brief summary table to stdout
for ds_name, res in experiment_data.items():
    try:
        tm = dict(res.get("metrics", {}).get("test", [("final", {})])[0][1])
        print(
            f"[Summary] {ds_name}: loss={tm.get('loss', np.nan):.4f}, acc={tm.get('acc', np.nan):.4f}, macro_f1={tm.get('macro_f1', np.nan):.4f}, top3={tm.get('top3', np.nan):.4f}, ece={tm.get('ece', np.nan):.4f}"
        )
    except Exception as e:
        print(f"Error printing summary for {ds_name}: {e}")
