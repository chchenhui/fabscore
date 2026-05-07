import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# List of experiment data paths (filtered, non-None)
experiment_data_path_list = [
    "experiments/2025-07-28_23-01-58_scientific_claim_verification_mnist_attempt_0/logs/0-run/experiment_results/experiment_069df5f4c4de4842a291735c4c76dea1_proc_1501281/experiment_data.npy"
]

all_experiment_data = []
# Load experiment data
for experiment_data_path in experiment_data_path_list:
    try:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
    except Exception as e:
        print(f"Error loading experiment data from {experiment_data_path}: {e}")

# Aggregate only if at least 1 experiment loaded and mnist_claims exists
mnist_runs = []
for ed in all_experiment_data:
    if ed and "mnist_claims" in ed:
        mnist_runs.append(ed["mnist_claims"])

if len(mnist_runs) == 0:
    print("No valid mnist_claims experiment data to aggregate.")
else:
    # Helper to safely extract arrays
    def extract_metric(run, metric_expr):
        try:
            x = metric_expr(run)
            return np.array(x) if x is not None else None
        except Exception:
            return None

    # Gather epochs
    epochs_list = [extract_metric(r, lambda d: d.get("epochs")) for r in mnist_runs]
    common_epochs = None
    for e in epochs_list:
        if e is not None:
            common_epochs = np.array(e)
            break
    if common_epochs is None:
        print("No epochs found for mnist_claims runs.")

    # Loss curves aggregation
    try:
        all_train_loss = []
        all_val_loss = []
        for r in mnist_runs:
            train = extract_metric(r, lambda d: d.get("losses", {}).get("train"))
            val = extract_metric(r, lambda d: d.get("losses", {}).get("val"))
            # Only keep if shapes match epochs
            if (
                train is not None
                and val is not None
                and len(train) == len(common_epochs)
            ):
                all_train_loss.append(np.array(train))
                all_val_loss.append(np.array(val))
        if len(all_train_loss) >= 1:
            all_train_loss = np.stack(all_train_loss, axis=0)
            all_val_loss = np.stack(all_val_loss, axis=0)
            mean_train = np.mean(all_train_loss, axis=0)
            se_train = np.std(all_train_loss, axis=0, ddof=1) / np.sqrt(
                all_train_loss.shape[0]
            )
            mean_val = np.mean(all_val_loss, axis=0)
            se_val = np.std(all_val_loss, axis=0, ddof=1) / np.sqrt(
                all_val_loss.shape[0]
            )
            plt.figure()
            plt.plot(common_epochs, mean_train, label="Mean Train Loss", color="C0")
            plt.fill_between(
                common_epochs,
                mean_train - se_train,
                mean_train + se_train,
                alpha=0.25,
                color="C0",
                label="Train Loss StdErr",
            )
            plt.plot(common_epochs, mean_val, label="Mean Validation Loss", color="C1")
            plt.fill_between(
                common_epochs,
                mean_val - se_val,
                mean_val + se_val,
                alpha=0.25,
                color="C1",
                label="Val Loss StdErr",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                "MNIST Claims Dataset: Aggregated Training/Validation Loss\n(Mean ± StdErr across runs)"
            )
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, "mnist_claims_loss_curve_aggregated.png")
            )
            plt.close()
            print(
                f"Final epoch loss mean±se train: {mean_train[-1]:.4f} ± {se_train[-1]:.4f}, val: {mean_val[-1]:.4f} ± {se_val[-1]:.4f}"
            )
        else:
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss curve: {e}")
        plt.close()

    # Accuracy curves aggregation
    try:
        all_train_acc = []
        all_val_acc = []
        for r in mnist_runs:
            train = extract_metric(r, lambda d: d.get("metrics", {}).get("train_acc"))
            val = extract_metric(r, lambda d: d.get("metrics", {}).get("val_acc"))
            if (
                train is not None
                and val is not None
                and len(train) == len(common_epochs)
            ):
                all_train_acc.append(np.array(train))
                all_val_acc.append(np.array(val))
        if len(all_train_acc) >= 1:
            all_train_acc = np.stack(all_train_acc, axis=0)
            all_val_acc = np.stack(all_val_acc, axis=0)
            mean_train = np.mean(all_train_acc, axis=0)
            se_train = np.std(all_train_acc, axis=0, ddof=1) / np.sqrt(
                all_train_acc.shape[0]
            )
            mean_val = np.mean(all_val_acc, axis=0)
            se_val = np.std(all_val_acc, axis=0, ddof=1) / np.sqrt(all_val_acc.shape[0])
            plt.figure()
            plt.plot(common_epochs, mean_train, label="Mean Train Accuracy", color="C0")
            plt.fill_between(
                common_epochs,
                mean_train - se_train,
                mean_train + se_train,
                alpha=0.25,
                color="C0",
                label="Train Acc StdErr",
            )
            plt.plot(
                common_epochs, mean_val, label="Mean Validation Accuracy", color="C1"
            )
            plt.fill_between(
                common_epochs,
                mean_val - se_val,
                mean_val + se_val,
                alpha=0.25,
                color="C1",
                label="Val Acc StdErr",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(
                "MNIST Claims Dataset: Aggregated Training/Validation Accuracy\n(Mean ± StdErr across runs)"
            )
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, "mnist_claims_accuracy_curve_aggregated.png")
            )
            plt.close()
            print(
                f"Final epoch accuracy mean±se train: {mean_train[-1]:.4f} ± {se_train[-1]:.4f}, val: {mean_val[-1]:.4f} ± {se_val[-1]:.4f}"
            )
        else:
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated accuracy curve: {e}")
        plt.close()

    # At most 5 prediction-vs-ground-truth scatter plots, sampled evenly across available runs
    try:
        num_to_plot = min(len(mnist_runs), 5)
        idxs = np.linspace(0, len(mnist_runs) - 1, num_to_plot, dtype=int)
        for i, idx in enumerate(idxs):
            d = mnist_runs[idx]
            preds = extract_metric(d, lambda x: x.get("predictions"))
            gts = extract_metric(d, lambda x: x.get("ground_truth"))
            if (
                preds is not None
                and gts is not None
                and len(preds) == len(gts)
                and len(preds) > 0
            ):
                plt.figure(figsize=(6, 4))
                plt.scatter(
                    np.arange(len(preds)),
                    preds,
                    label="Prediction",
                    alpha=0.6,
                    color="b",
                    marker="o",
                    s=25,
                )
                plt.scatter(
                    np.arange(len(gts)),
                    gts,
                    label="Ground Truth",
                    alpha=0.6,
                    color="r",
                    marker="x",
                    s=25,
                )
                plt.xlabel("Sample Index")
                plt.ylabel("Label")
                plt.title(
                    f"MNIST Claims Dataset: Val Set Predictions vs Ground Truth\nRun {idx+1} of {len(mnist_runs)} (Left: GT [red x], Right: Pred [blue o])"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"mnist_claims_pred_vs_gt_run{idx+1}.png")
                )
                plt.close()
    except Exception as e:
        print(f"Error creating aggregated prediction/gt plots: {e}")
        plt.close()
