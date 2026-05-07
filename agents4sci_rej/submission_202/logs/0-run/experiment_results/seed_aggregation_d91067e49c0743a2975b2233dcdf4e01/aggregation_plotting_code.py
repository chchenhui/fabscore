import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data_path_list = [
    "experiments/2025-07-28_23-01-58_scientific_claim_verification_mnist_attempt_0/logs/0-run/experiment_results/experiment_8d1c9cccde634b2592d985e98793f7fe_proc_1502176/experiment_data.npy"
    # Add paths here if more are available
]
all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        edata = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(edata)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Gather all runs under "num_epochs_tuning" > dataset_name = "mnist_claims"
try:
    results_by_run = []
    for edata in all_experiment_data:
        if (
            "num_epochs_tuning" in edata
            and "mnist_claims" in edata["num_epochs_tuning"]
        ):
            results_by_run.append(edata["num_epochs_tuning"]["mnist_claims"])
    if not results_by_run:
        raise ValueError("No valid mnist_claims results found")
except Exception as e:
    print(f"Error extracting experiment results: {e}")

# Collect all epoch configs
try:
    all_epoch_keys = set()
    for run in results_by_run:
        keys = [k for k in run if k.startswith("epochs_")]
        all_epoch_keys.update(keys)
    epoch_keys = sorted(list(all_epoch_keys), key=lambda x: int(x.split("_")[1]))
    epoch_counts = [int(x.split("_")[1]) for x in epoch_keys]
except Exception as e:
    print(f"Error collecting epoch configs: {e}")

# Stack metrics for each epoch config (shape: [num_runs, T])
from collections import defaultdict

metric_arrays = defaultdict(lambda: defaultdict(list))
epoch_lists = defaultdict(list)  # Store time axes to check consistency
try:
    for ek in epoch_keys:
        for run in results_by_run:
            if ek in run:
                entry = run[ek]
                epochs = entry["epochs"]
                epoch_lists[ek].append(epochs)
                # Metrics and losses
                train_acc = entry["metrics"]["train_acc"]
                val_acc = entry["metrics"]["val_acc"]
                train_loss = entry["losses"]["train"]
                val_loss = entry["losses"]["val"]
                metric_arrays[ek]["train_acc"].append(train_acc)
                metric_arrays[ek]["val_acc"].append(val_acc)
                metric_arrays[ek]["train_loss"].append(train_loss)
                metric_arrays[ek]["val_loss"].append(val_loss)
except Exception as e:
    print(f"Error aggregating metrics across runs: {e}")

# Helper: consistent epoch axis? Use first found
epoch_axis_by_key = {
    ek: epoch_lists[ek][0] if epoch_lists[ek] else None for ek in epoch_keys
}

# Plot aggregated train/val accuracy curves with SEM
try:
    plt.figure(figsize=(10, 7))
    for i, ek in enumerate(epoch_keys):
        if len(metric_arrays[ek]["train_acc"]) == 0:
            continue
        epochs = np.array(epoch_axis_by_key[ek])
        arr_train = np.stack(metric_arrays[ek]["train_acc"])
        arr_val = np.stack(metric_arrays[ek]["val_acc"])
        mean_train = arr_train.mean(axis=0)
        mean_val = arr_val.mean(axis=0)
        sem_train = arr_train.std(axis=0, ddof=1) / np.sqrt(arr_train.shape[0])
        sem_val = arr_val.std(axis=0, ddof=1) / np.sqrt(arr_val.shape[0])
        plt.plot(
            epochs,
            mean_train,
            "--",
            alpha=0.7,
            label=f"Train, epochs={epoch_counts[i]}",
        )
        plt.fill_between(
            epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.15
        )
        plt.plot(epochs, mean_val, "-", label=f"Val, epochs={epoch_counts[i]}")
        plt.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.15)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        "Train/Validation Accuracy (Mean ± SEM)\nMNISTClaimDataset (num_epochs tuning)"
    )
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Mean curves ± SEM")
    save_path = os.path.join(working_dir, "mnist_claims_accuracy_curve_agg.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# Plot aggregated train/val loss curves with SEM
try:
    plt.figure(figsize=(10, 7))
    for i, ek in enumerate(epoch_keys):
        if len(metric_arrays[ek]["train_loss"]) == 0:
            continue
        epochs = np.array(epoch_axis_by_key[ek])
        arr_train = np.stack(metric_arrays[ek]["train_loss"])
        arr_val = np.stack(metric_arrays[ek]["val_loss"])
        mean_train = arr_train.mean(axis=0)
        mean_val = arr_val.mean(axis=0)
        sem_train = arr_train.std(axis=0, ddof=1) / np.sqrt(arr_train.shape[0])
        sem_val = arr_val.std(axis=0, ddof=1) / np.sqrt(arr_val.shape[0])
        plt.plot(
            epochs,
            mean_train,
            "--",
            alpha=0.7,
            label=f"Train, epochs={epoch_counts[i]}",
        )
        plt.fill_between(
            epochs, mean_train - sem_train, mean_train + sem_train, alpha=0.15
        )
        plt.plot(epochs, mean_val, "-", label=f"Val, epochs={epoch_counts[i]}")
        plt.fill_between(epochs, mean_val - sem_val, mean_val + sem_val, alpha=0.15)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        "Train/Validation Loss (Mean ± SEM)\nMNISTClaimDataset (num_epochs tuning)"
    )
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Mean curves ± SEM")
    save_path = os.path.join(working_dir, "mnist_claims_loss_curve_agg.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# Final Val Accuracy Bar Plot (mean ± sem for each epoch config)
try:
    means = []
    sems = []
    valid_epoch_counts = []
    for i, ek in enumerate(epoch_keys):
        vals = []
        for arr in metric_arrays[ek]["val_acc"]:
            if len(arr) > 0:
                vals.append(arr[-1])
        if len(vals) > 0:
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / (np.sqrt(len(vals))))
            valid_epoch_counts.append(epoch_counts[i])
    if means:
        plt.figure(figsize=(9, 5))
        plt.bar(valid_epoch_counts, means, yerr=sems, capsize=5, alpha=0.85)
        plt.xlabel("Number of Training Epochs")
        plt.ylabel("Final Validation Accuracy")
        plt.title(
            "Final Validation Accuracy (Mean ± SEM)\nMNISTClaimDataset (num_epochs tuning)"
        )
        plt.tight_layout()
        save_path = os.path.join(working_dir, "mnist_claims_agg_final_val_acc_bar.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error creating final val accuracy bar plot: {e}")
    plt.close()

# Optionally, plot aggregated prediction/GT histogram for at most 5 settings, if available (pick evenly spread across configs)
try:
    chosen = []
    if len(epoch_keys) > 0:
        # Choose max 5 configs, spread out
        step = max(1, len(epoch_keys) // 5)
        chosen = [epoch_keys[i] for i in range(0, len(epoch_keys), step)][:5]
    for ek in chosen:
        # Stack all preds/gts for last epoch from all runs for this config
        all_preds = []
        all_gts = []
        for run in results_by_run:
            if ek in run:
                preds = run[ek].get("predictions", None)
                gts = run[ek].get("ground_truth", None)
                if preds is not None and gts is not None:
                    all_preds.append(np.array(preds))
                    all_gts.append(np.array(gts))
        if all_preds and all_gts:
            preds_flat = np.concatenate(all_preds)
            gts_flat = np.concatenate(all_gts)
            plt.figure(figsize=(7, 4))
            plt.hist(
                [gts_flat, preds_flat],
                bins=2,
                alpha=0.7,
                label=["Ground Truth", "Predictions"],
            )
            plt.xticks([0, 1])
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.title(
                f"Aggregated Validation Prediction Distribution (epochs={ek.split('_')[1]})\nMNISTClaimDataset\n"
                "Left: Ground Truth, Right: Generated Predictions (final epoch, all runs)"
            )
            plt.legend()
            save_path = os.path.join(
                working_dir,
                f"mnist_claims_agg_val_pred_hist_epochs{ek.split('_')[1]}.png",
            )
            plt.savefig(save_path)
            plt.close()
            print(f"Saved: {save_path}")
except Exception as e:
    print(f"Error creating aggregated prediction histogram: {e}")
    plt.close()

# Print out mean ± sem of final validation accuracy for each config
try:
    print("Final validation accuracy (across all runs):")
    for c, m, s in zip(valid_epoch_counts, means, sems):
        print(f"  num_epochs={c}: {m:.4f} ± {s:.4f}")
except Exception as e:
    print(f"Error printing summary final accuracies: {e}")
