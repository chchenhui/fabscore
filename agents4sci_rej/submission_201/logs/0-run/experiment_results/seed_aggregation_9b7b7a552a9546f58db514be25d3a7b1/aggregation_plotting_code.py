import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Gather the list of available experiment_data.npy files
experiment_data_path_list = [
    "experiments/2025-07-28_23-01-58_scientific_claim_verification_mnist_attempt_0/logs/0-run/experiment_results/experiment_04ec95ecb3d443e4b9b2727a0a0b4c7a_proc_1522608/experiment_data.npy"
    # Add more paths as appropriate if more runs exist (including ablation or other seeds)
]

all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data from {experiment_data_path}: {e}")

# Proceed only if we loaded at least one experiment_data dict
if len(all_experiment_data) == 0:
    print("No experiment data loaded; skipping plotting.")
else:
    datasets = ["mnist", "fashion_mnist", "svhn"]
    sets = ["train", "val", "train_logic", "val_logic"]
    loss_sets = ["train", "val"]

    variant = "no_tokenization"

    for ds in datasets:
        # ========== ACCURACY / LOGIC CONSISTENCY MEANS + ERRORBARS ==========
        try:
            # Collect NUM_RUNS x NUM_EPOCHS arrays for each metric
            epochs_all = []
            acc_dict = {m: [] for m in sets}
            loss_dict = {l: [] for l in loss_sets}

            for exp_data in all_experiment_data:
                try:
                    epochs_stub = exp_data[variant][ds]["epochs"]
                    # Save epochs for checking
                    epochs_all.append(epochs_stub)
                    # Metrics
                    metrics = exp_data[variant][ds]["metrics"]
                    for m in sets:
                        acc_dict[m].append(np.asarray(metrics.get(m, [])))
                    # Loss
                    losses = exp_data[variant][ds]["losses"]
                    for l in loss_sets:
                        loss_dict[l].append(np.asarray(losses.get(l, [])))
                except Exception as e:
                    print(f"Error extracting {ds} data from one run: {e}")

            # Consistency: all epochs arrays should be the same; take the first
            epochs = epochs_all[0] if len(epochs_all) else None

            # ---- Training/Validation Accuracy mean and std error ----
            plt.figure()
            for m, label in [
                ("train", "Train Accuracy"),
                ("val", "Validation Accuracy"),
            ]:
                vals_list = [arr for arr in acc_dict[m] if arr.size > 0]
                if len(vals_list) == 0:
                    continue
                values = np.stack(vals_list)
                mean = np.mean(values, axis=0)
                sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                plt.plot(epochs, mean, label=f"{label} (Mean)", linewidth=2)
                plt.fill_between(
                    epochs, mean - sem, mean + sem, alpha=0.2, label=f"{label} (±SE)"
                )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds.upper()} - Train & Val Accuracy with Std. Error\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_train_val_acc_ablation_agg.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()

            # ---- Logic Consistency mean and std error ----
            plt.figure()
            for m, label in [
                ("train_logic", "Train Logic Consistency"),
                ("val_logic", "Validation Logic Consistency"),
            ]:
                vals_list = [arr for arr in acc_dict[m] if arr.size > 0]
                if len(vals_list) == 0:
                    continue
                values = np.stack(vals_list)
                mean = np.mean(values, axis=0)
                sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                plt.plot(epochs, mean, label=f"{label} (Mean)", linewidth=2)
                plt.fill_between(
                    epochs, mean - sem, mean + sem, alpha=0.2, label=f"{label} (±SE)"
                )
            plt.xlabel("Epoch")
            plt.ylabel("Logic Consistency Accuracy")
            plt.title(
                f"{ds.upper()} - Logic Consistency Accuracies with Std. Error\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_logic_acc_ablation_agg.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting accuracy/logic with error bars for {ds}: {e}")
            plt.close()

        # ========== LOSS MEANS + ERRORBARS ==========
        try:
            plt.figure()
            for lkey, label in [("train", "Train Loss"), ("val", "Validation Loss")]:
                vals_list = [arr for arr in loss_dict[lkey] if arr.size > 0]
                if len(vals_list) == 0:
                    continue
                values = np.stack(vals_list)
                mean = np.mean(values, axis=0)
                sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                plt.plot(epochs, mean, label=f"{label} (Mean)", linewidth=2)
                plt.fill_between(
                    epochs, mean - sem, mean + sem, alpha=0.2, label=f"{label} (±SE)"
                )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"{ds.upper()} - Training/Validation Loss with Std. Error\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_loss_curve_ablation_agg.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting losses with error bars for {ds}: {e}")
            plt.close()

        # ========== Overlaid VAL/LOGIC ACC with ERRORBARS ==========
        try:
            plt.figure()
            for m, label in [
                ("val", "Validation Accuracy"),
                ("val_logic", "Validation Logic Consistency"),
            ]:
                vals_list = [arr for arr in acc_dict[m] if arr.size > 0]
                if len(vals_list) == 0:
                    continue
                values = np.stack(vals_list)
                mean = np.mean(values, axis=0)
                sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
                plt.plot(epochs, mean, label=f"{label} (Mean)", linewidth=2)
                plt.fill_between(
                    epochs, mean - sem, mean + sem, alpha=0.2, label=f"{label} (±SE)"
                )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                f"{ds.upper()} - Validation vs Logic Accuracy (Mean ± SE)\n(No Tokenization Ablation)"
            )
            plt.legend()
            fname = f"{ds}_val_and_logic_acc_overlay_ablation_agg.png"
            plt.savefig(os.path.join(working_dir, fname))
            plt.close()
        except Exception as e:
            print(f"Error plotting overlay val/logic with error bars for {ds}: {e}")
            plt.close()
