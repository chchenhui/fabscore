import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Aggregate results across multiple experiment_data.npy files and plot means with SEM
def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def compute_sem(arr, axis=0):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(arr, axis=axis))
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(n)


def main():
    # Load multiple experiment_data dicts
    try:
        experiment_data_path_list = [
            "experiments/2025-09-13_11-32-42_resource_centric_ppm_agents_attempt_0/logs/0-run/experiment_results/experiment_9a2ff2eda58e4ea8966e9f8f6f6ba8c5_proc_332087/experiment_data.npy",
            "None/experiment_data.npy",
            "experiments/2025-09-13_11-32-42_resource_centric_ppm_agents_attempt_0/logs/0-run/experiment_results/experiment_74c75ff464094554b5c0e6b72036923c_proc_332088/experiment_data.npy",
        ]
        all_experiment_data = []
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        for rel_path in experiment_data_path_list:
            try:
                full_path = os.path.join(root, rel_path) if root else rel_path
                experiment_data = np.load(full_path, allow_pickle=True).item()
                if isinstance(experiment_data, dict):
                    all_experiment_data.append(experiment_data)
                    print(f"Loaded experiment data from: {full_path}")
                else:
                    print(f"Unexpected data format in: {full_path}")
            except Exception as e:
                print(f"Error loading experiment data: {e}")
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        all_experiment_data = []

    # Merge by dataset name across runs
    datasets = {}
    for run_idx, exp in enumerate(all_experiment_data):
        for ds_name, ed in exp.items():
            if ds_name not in datasets:
                datasets[ds_name] = {
                    "train_losses": [],
                    "val_losses": [],
                    "epochs": [],
                    "prefix_lens": [],
                    "top3_flags": [],
                    "test_metrics": [],  # dicts with loss, acc, macro_f1, top3
                }
            # losses
            tr = ed.get("losses", {}).get("train", [])
            va = ed.get("losses", {}).get("val", [])
            # Extract only the y values preserving order by epoch index
            tr_vals = [y for (_, y) in tr] if len(tr) > 0 else []
            va_vals = [y for (_, y) in va] if len(va) > 0 else []
            epochs = ed.get(
                "epochs", list(range(1, min(len(tr_vals), len(va_vals)) + 1))
            )
            # store if any present
            if len(tr_vals) > 0:
                datasets[ds_name]["train_losses"].append(np.array(tr_vals, dtype=float))
            if len(va_vals) > 0:
                datasets[ds_name]["val_losses"].append(np.array(va_vals, dtype=float))
            if len(epochs) > 0:
                datasets[ds_name]["epochs"].append(np.array(epochs, dtype=int))

            # top3 vs prefix length
            pref = ed.get("prefix_lens", [])
            flags = ed.get("top3_flags", [])
            if len(pref) > 0 and len(flags) > 0:
                datasets[ds_name]["prefix_lens"].append(np.array(pref, dtype=int))
                datasets[ds_name]["top3_flags"].append(np.array(flags, dtype=int))

            # test metrics
            test_m = ed.get("metrics", {}).get("test", [])
            if (
                isinstance(test_m, list)
                and len(test_m) > 0
                and isinstance(test_m[0], (list, tuple))
                and isinstance(test_m[0][1], dict)
            ):
                datasets[ds_name]["test_metrics"].append(dict(test_m[0][1]))

    # For each dataset, create aggregated plots
    for name, bundle in datasets.items():
        # Aggregated loss curves (mean ± SEM) using min common length across runs
        try:
            tr_list = bundle["train_losses"]
            va_list = bundle["val_losses"]

            # Determine aligned lengths
            if len(tr_list) > 0:
                min_tr_len = min(len(x) for x in tr_list)
                tr_aligned = (
                    np.stack([x[:min_tr_len] for x in tr_list], axis=0)
                    if min_tr_len > 0
                    else None
                )
            else:
                tr_aligned = None
            if len(va_list) > 0:
                min_va_len = min(len(x) for x in va_list)
                va_aligned = (
                    np.stack([x[:min_va_len] for x in va_list], axis=0)
                    if min_va_len > 0
                    else None
                )
            else:
                va_aligned = None

            if (tr_aligned is not None and tr_aligned.size > 0) or (
                va_aligned is not None and va_aligned.size > 0
            ):
                plt.figure()
                legend_handles = []
                if tr_aligned is not None and tr_aligned.size > 0:
                    x_tr = np.arange(1, tr_aligned.shape[1] + 1)
                    tr_mean = np.mean(tr_aligned, axis=0)
                    tr_sem = compute_sem(tr_aligned, axis=0)
                    plt.plot(x_tr, tr_mean, color="tab:blue", label="Train mean")
                    plt.fill_between(
                        x_tr,
                        tr_mean - tr_sem,
                        tr_mean + tr_sem,
                        color="tab:blue",
                        alpha=0.2,
                        label="Train ± SEM",
                    )
                if va_aligned is not None and va_aligned.size > 0:
                    x_va = np.arange(1, va_aligned.shape[1] + 1)
                    va_mean = np.mean(va_aligned, axis=0)
                    va_sem = compute_sem(va_aligned, axis=0)
                    plt.plot(x_va, va_mean, color="tab:orange", label="Val mean")
                    plt.fill_between(
                        x_va,
                        va_mean - va_sem,
                        va_mean + va_sem,
                        color="tab:orange",
                        alpha=0.2,
                        label="Val ± SEM",
                    )
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(
                    f"Aggregated Loss Curves - {name}\nLeft: Ground Truth, Right: Generated Samples | Dataset: {name}"
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{name}_aggregated_loss_curves.png")
                )
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {name}: {e}")
            plt.close()

        # Aggregated Top-3 vs Prefix Length (mean ± SEM across runs)
        try:
            pref_runs = bundle["prefix_lens"]
            flag_runs = bundle["top3_flags"]
            if len(pref_runs) > 0 and len(flag_runs) > 0:
                # For each run, compute per-length mean
                per_len_run_means = {}
                for pr, fr in zip(pref_runs, flag_runs):
                    d = {}
                    # group flags by prefix length for this run
                    for L, f in zip(pr, fr):
                        d.setdefault(int(L), []).append(int(f))
                    # convert to mean per L for this run
                    for L, lst in d.items():
                        per_len_run_means.setdefault(L, []).append(float(np.mean(lst)))
                # Aggregate across runs
                xs = sorted(per_len_run_means.keys())
                vals = [per_len_run_means[L] for L in xs]
                means = np.array([np.mean(v) for v in vals], dtype=float)
                sems = np.array(
                    [compute_sem(np.array(v)) if len(v) > 1 else 0.0 for v in vals],
                    dtype=float,
                )
                plt.figure()
                plt.errorbar(
                    xs, means, yerr=sems, fmt="-o", capsize=3, label="Top-3 mean ± SEM"
                )
                plt.xlabel("Prefix Length")
                plt.ylabel("Top-3 Accuracy")
                plt.title(
                    f"Aggregated Top-3 Accuracy vs Prefix Length - {name}\nLeft: Ground Truth, Right: Generated Samples | Dataset: {name}"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{name}_aggregated_top3_vs_prefixlen.png"
                    )
                )
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated Top-3 vs prefix length for {name}: {e}")
            plt.close()

        # Aggregated final test metrics (bar-like error plot)
        try:
            tms = bundle["test_metrics"]
            if len(tms) > 0:
                # Collect arrays
                keys = ["acc", "macro_f1", "top3", "loss"]
                data = {k: [] for k in keys}
                for m in tms:
                    for k in keys:
                        if k in m:
                            data[k].append(float(m[k]))
                # Only include metrics that exist
                present_keys = [k for k in keys if len(data[k]) > 0]
                if len(present_keys) > 0:
                    means = [np.mean(data[k]) for k in present_keys]
                    sems = [
                        compute_sem(np.array(data[k])) if len(data[k]) > 1 else 0.0
                        for k in present_keys
                    ]
                    x = np.arange(len(present_keys))
                    plt.figure()
                    # simple point with error bars to avoid simulating bar heights
                    plt.errorbar(
                        x,
                        means,
                        yerr=sems,
                        fmt="o",
                        capsize=5,
                        linestyle="None",
                        label="Mean ± SEM",
                    )
                    plt.xticks(x, present_keys)
                    plt.ylabel("Value")
                    plt.title(
                        f"Aggregated Test Metrics - {name}\nLeft: Ground Truth, Right: Generated Samples | Dataset: {name}"
                    )
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(working_dir, f"{name}_aggregated_test_metrics.png")
                    )
                    plt.close()
        except Exception as e:
            print(f"Error creating aggregated test metrics for {name}: {e}")
            plt.close()

        # Print aggregated evaluation metrics
        try:
            tms = bundle["test_metrics"]
            if len(tms) > 0:
                accs = [m["acc"] for m in tms if "acc" in m]
                f1s = [m["macro_f1"] for m in tms if "macro_f1" in m]
                top3s = [m["top3"] for m in tms if "top3" in m]
                losses = [m["loss"] for m in tms if "loss" in m]

                def stat_str(arr, name_):
                    if len(arr) == 0:
                        return f"{name_}=N/A"
                    mean = np.mean(arr)
                    sem = compute_sem(np.array(arr)) if len(arr) > 1 else 0.0
                    return f"{name_} mean={mean:.4f} ± {sem:.4f} (SEM), n={len(arr)}"

                print(
                    f"{name} | "
                    + " | ".join(
                        [
                            stat_str(accs, "acc"),
                            stat_str(f1s, "macro_f1"),
                            stat_str(top3s, "top3"),
                            stat_str(losses, "loss"),
                        ]
                    )
                )
        except Exception as e:
            print(f"Error printing metrics for {name}: {e}")


if __name__ == "__main__":
    main()
