import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Experiment data paths -- replace None with actual npy files if more available
experiment_data_path_list = [
    "experiments/2025-07-28_23-01-58_scientific_claim_verification_mnist_attempt_0/logs/0-run/experiment_results/experiment_272875af61684c03898c49779a68b795_proc_1514333/experiment_data.npy",
]
all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data at {experiment_data_path}: {e}")

if len(all_experiment_data) == 0:
    print("No experiment data successfully loaded. Exiting.")
else:
    # Setup
    datasets = ["mnist", "fashion_mnist", "svhn"]
    ds_labels = {"mnist": "MNIST", "fashion_mnist": "Fashion-MNIST", "svhn": "SVHN"}
    colors = {"mnist": "b", "fashion_mnist": "r", "svhn": "g"}

    # Helper to aggregate metric across different runs with variable epoch lengths
    def extract_metric_over_runs(metric_path, ds):
        """metric_path is a list ['metrics','val_acc'] etc."""
        all_arrs = []
        for exp in all_experiment_data:
            try:
                arr = exp[ds]
                for key in metric_path:
                    arr = arr[key]
                all_arrs.append(np.array(arr))
            except Exception:
                continue
        # Truncate to shortest available run for safe stacking
        if len(all_arrs) == 0:
            return None
        min_len = min([len(a) for a in all_arrs])
        all_arrs = [a[:min_len] for a in all_arrs]
        stacked = np.stack(all_arrs, axis=0)  # shape = (num_runs, num_epochs)
        return stacked

    def plot_with_error_bands(
        x, metrics_arr, label_mean, label_err, color, linestyle="-"
    ):
        mean = np.mean(metrics_arr, axis=0)
        stderr = (
            np.std(metrics_arr, axis=0, ddof=1) / np.sqrt(metrics_arr.shape[0])
            if metrics_arr.shape[0] > 1
            else np.zeros_like(mean)
        )
        plt.plot(x, mean, linestyle, color=color, label=label_mean)
        plt.fill_between(
            x,
            mean - stderr,
            mean + stderr,
            color=color,
            alpha=0.18,
            label=label_err,
        )
        return mean, stderr

    # For each dataset: accuracy and loss with mean/stderr
    for ds in datasets:
        try:
            metric_dict = {
                "Accuracy": (["metrics", "train_acc"], ["metrics", "val_acc"]),
                "Loss": (["losses", "train"], ["losses", "val"]),
            }
            for metric_name, (train_key, val_key) in metric_dict.items():
                metrics_train = extract_metric_over_runs(train_key, ds)
                metrics_val = extract_metric_over_runs(val_key, ds)
                # Use epochs from first successful run
                epochs = None
                for exp in all_experiment_data:
                    if ds in exp and "epochs" in exp[ds]:
                        epochs = np.array(exp[ds]["epochs"])
                        break
                if epochs is None or metrics_train is None or metrics_val is None:
                    continue
                min_len = min(len(epochs), metrics_train.shape[1], metrics_val.shape[1])
                epochs_plot = epochs[:min_len]
                metrics_train = metrics_train[:, :min_len]
                metrics_val = metrics_val[:, :min_len]
                plt.figure(figsize=(8, 6))
                train_mean, train_se = plot_with_error_bands(
                    epochs_plot,
                    metrics_train,
                    label_mean="Train Mean",
                    label_err="Train Std. Error",
                    color=colors[ds],
                    linestyle="--",
                )
                val_mean, val_se = plot_with_error_bands(
                    epochs_plot,
                    metrics_val,
                    label_mean="Val Mean",
                    label_err="Val Std. Error",
                    color=colors[ds],
                    linestyle="-",
                )
                plt.xlabel("Epoch")
                plt.ylabel(metric_name)
                plt.title(
                    f"{ds_labels[ds]} - Train/Validation {metric_name}\n(Mean ± Std. Error across {metrics_train.shape[0]} runs)"
                )
                plt.legend()
                fname = os.path.join(
                    working_dir, f"{ds}_mean_stderr_train_val_{metric_name.lower()}.png"
                )
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated {metric_name} plot for {ds}: {e}")
            plt.close()

    # Comparison plot: validation accuracy across datasets, with error bands
    try:
        plt.figure(figsize=(8, 6))
        for ds in datasets:
            metrics_val = extract_metric_over_runs(["metrics", "val_acc"], ds)
            if metrics_val is None:
                continue
            # Use epochs from first successful run
            for exp in all_experiment_data:
                if ds in exp and "epochs" in exp[ds]:
                    epochs = np.array(exp[ds]["epochs"])
                    break
            min_len = min(len(epochs), metrics_val.shape[1])
            epochs_plot = epochs[:min_len]
            metrics_val = metrics_val[:, :min_len]
            val_mean, val_se = plot_with_error_bands(
                epochs_plot,
                metrics_val,
                label_mean=f"{ds_labels[ds]} Mean",
                label_err=f"{ds_labels[ds]} Std. Error",
                color=colors[ds],
            )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title(
            "Validation Accuracy Comparison Across Datasets\nMean ± Std. Error over runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, "val_acc_compare_all_datasets_avg.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated validation accuracy comparison plot: {e}")
        plt.close()

    # Comparison plot: logical consistency accuracy across datasets, with error bands
    try:
        plt.figure(figsize=(8, 6))
        for ds in datasets:
            metrics_val = extract_metric_over_runs(["metrics", "val_logic"], ds)
            if metrics_val is None:
                continue
            for exp in all_experiment_data:
                if ds in exp and "epochs" in exp[ds]:
                    epochs = np.array(exp[ds]["epochs"])
                    break
            min_len = min(len(epochs), metrics_val.shape[1])
            epochs_plot = epochs[:min_len]
            metrics_val = metrics_val[:, :min_len]
            val_mean, val_se = plot_with_error_bands(
                epochs_plot,
                metrics_val,
                label_mean=f"{ds_labels[ds]} Mean",
                label_err=f"{ds_labels[ds]} Std. Error",
                color=colors[ds],
            )
        plt.xlabel("Epoch")
        plt.ylabel("Logical Consistency Accuracy")
        plt.title(
            "Logical Consistency Accuracy Comparison\nMean ± Std. Error over runs"
        )
        plt.legend()
        fname = os.path.join(working_dir, "val_logic_acc_compare_all_datasets_avg.png")
        plt.savefig(fname)
        plt.close()
    except Exception as e:
        print(
            f"Error creating aggregated logical consistency accuracy comparison plot: {e}"
        )
        plt.close()

    # Overlaid validation acc & logic for each dataset, mean/se
    for ds in datasets:
        try:
            val_acc = extract_metric_over_runs(["metrics", "val_acc"], ds)
            val_logic = extract_metric_over_runs(["metrics", "val_logic"], ds)
            for exp in all_experiment_data:
                if ds in exp and "epochs" in exp[ds]:
                    epochs = np.array(exp[ds]["epochs"])
                    break
            if val_acc is None or val_logic is None:
                continue
            min_len = min(len(epochs), val_acc.shape[1], val_logic.shape[1])
            epochs_plot = epochs[:min_len]
            val_acc = val_acc[:, :min_len]
            val_logic = val_logic[:, :min_len]
            plt.figure(figsize=(8, 6))
            acc_mean, acc_se = plot_with_error_bands(
                epochs_plot,
                val_acc,
                "Val Acc Mean",
                "Val Acc Std. Err",
                color="b",
                linestyle="-",
            )
            logic_mean, logic_se = plot_with_error_bands(
                epochs_plot,
                val_logic,
                "Logic Acc Mean",
                "Logic Acc Std. Err",
                color="r",
                linestyle="--",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds_labels[ds]}: Validation vs Logical Consistency (Mean ± Std. Err)"
            )
            plt.legend()
            fname = os.path.join(working_dir, f"{ds}_val_vs_logic_accuracy_avg.png")
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Error creating overlaid acc/logic (mean/stderr) plot for {ds}: {e}")
            plt.close()

    # Only aggregate-level histograms if ground truth and predictions exist (not mean/stderr)
    for ds in datasets:
        try:
            preds_list, gts_list = [], []
            for exp in all_experiment_data:
                preds = exp[ds].get("predictions", None) if ds in exp else None
                gts = exp[ds].get("ground_truth", None) if ds in exp else None
                if preds is not None and gts is not None:
                    preds_list.append(preds)
                    gts_list.append(gts)
            if preds_list and gts_list:
                # Use first run's preds/gts for histogram
                preds, gts = preds_list[0], gts_list[0]
                plt.figure(figsize=(7, 4))
                plt.hist(
                    [gts, preds],
                    bins=2,
                    alpha=0.7,
                    label=["Ground Truth", "Predictions"],
                )
                plt.xticks([0, 1])
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title(
                    f"{ds_labels[ds]} Validation Predictions vs Ground Truth\nLeft: Ground Truth, Right: Model Predictions (Final Epoch)"
                )
                plt.legend()
                fname = os.path.join(working_dir, f"{ds}_val_pred_hist.png")
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Error creating val pred histogram for {ds}: {e}")
            plt.close()

    # Print final mean ± std. error for val acc and logic
    try:
        print("Final mean ± std. error per dataset (last epoch, across runs):")
        for ds in datasets:
            r_mean, r_se = "--", "--"
            l_mean, l_se = "--", "--"
            val_acc = extract_metric_over_runs(["metrics", "val_acc"], ds)
            val_logic = extract_metric_over_runs(["metrics", "val_logic"], ds)
            if val_acc is not None and val_acc.shape[1] > 0:
                mean = np.mean(val_acc[:, -1])
                se = (
                    np.std(val_acc[:, -1], ddof=1) / np.sqrt(val_acc.shape[0])
                    if val_acc.shape[0] > 1
                    else 0
                )
                r_mean, r_se = mean, se
            if val_logic is not None and val_logic.shape[1] > 0:
                mean = np.mean(val_logic[:, -1])
                se = (
                    np.std(val_logic[:, -1], ddof=1) / np.sqrt(val_logic.shape[0])
                    if val_logic.shape[0] > 1
                    else 0
                )
                l_mean, l_se = mean, se
            print(
                f"  {ds_labels[ds]}: Val Acc = {r_mean:.4f} ± {r_se:.4f}  |"
                f" Logical Consistency = {l_mean:.4f} ± {l_se:.4f}"
            )
    except Exception as e:
        print(f"Error printing final aggregate metrics: {e}")
