import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

from collections import defaultdict
from sklearn.metrics import confusion_matrix


def safe_load_experiment(path_str):
    try:
        # Try environment-rooted path
        root = os.getenv("AI_SCIENTIST_ROOT")
        if root is not None and len(root) > 0:
            p = os.path.join(root, path_str)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True).item()
        # Fallback: treat as relative/absolute path
        if os.path.isfile(path_str):
            return np.load(path_str, allow_pickle=True).item()
        # Also try under working dir if provided as None/experiment_data.npy pattern
        p2 = os.path.join(working_dir, os.path.basename(path_str))
        if os.path.isfile(p2):
            return np.load(p2, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading experiment data from {path_str}: {e}")
    return None


def sem(a, axis=0):
    a = np.array(a, dtype=float)
    if a.size == 0:
        return np.array([])
    # count non-nan along axis
    n = np.sum(~np.isnan(a), axis=axis)
    std = np.nanstd(a, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = std / np.sqrt(np.maximum(n, 1))
    return se


def downsample_xs(xs, max_points=5):
    xs_sorted = np.array(sorted(xs))
    if len(xs_sorted) <= max_points:
        return xs_sorted.tolist()
    # pick approx quantile positions
    qs = np.linspace(0, 1, num=max_points)
    idx = np.unique(
        np.clip((qs * (len(xs_sorted) - 1)).round().astype(int), 0, len(xs_sorted) - 1)
    )
    return xs_sorted[idx].tolist()


def main():
    # Collect all experiment data dicts
    try:
        experiment_data_path_list = [
            "None/experiment_data.npy",
            "experiments/2025-09-13_11-32-42_resource_centric_ppm_agents_attempt_0/logs/0-run/experiment_results/experiment_57ccd5de0de34674be3985e9f94220ad_proc_361839/experiment_data.npy",
            "None/experiment_data.npy",
        ]
        all_experiment_data = []
        for experiment_data_path in experiment_data_path_list:
            data = safe_load_experiment(experiment_data_path)
            if data is not None and isinstance(data, dict) and len(data) > 0:
                all_experiment_data.append(data)
            else:
                print(
                    f"Skipped loading from {experiment_data_path} (missing or empty)."
                )
        if len(all_experiment_data) == 0:
            print("No experiment_data loaded; nothing to plot.")
            return
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        all_experiment_data = []
        return

    # Build dataset -> list of run dicts
    datasets_runs = defaultdict(list)
    for run_idx, exp_dict in enumerate(all_experiment_data):
        for ds_name, ds_payload in exp_dict.items():
            datasets_runs[ds_name].append(ds_payload)

    # Aggregate and plot per dataset
    for ds_name, runs in datasets_runs.items():
        # Aggregate losses (train/val) by aligning epochs to min length
        try:
            # Collect lists of loss sequences
            train_losses = []
            val_losses = []
            for r in runs:
                tr = r.get("losses", {}).get("train", [])
                vl = r.get("losses", {}).get("val", [])
                # entries are list of (epoch, value) or just values; normalize to values
                tr_vals = (
                    [y for (_, y) in tr]
                    if len(tr) > 0 and isinstance(tr[0], (list, tuple))
                    else list(tr)
                )
                vl_vals = (
                    [y for (_, y) in vl]
                    if len(vl) > 0 and isinstance(vl[0], (list, tuple))
                    else list(vl)
                )
                if len(tr_vals) > 0:
                    train_losses.append(np.array(tr_vals, dtype=float))
                if len(vl_vals) > 0:
                    val_losses.append(np.array(vl_vals, dtype=float))
            # Plot if we have at least one train or val
            if len(train_losses) > 0 or len(val_losses) > 0:
                plt.figure()
                subtitle = "Aggregated across runs | Next-activity"
                if len(train_losses) > 0:
                    L = min([len(a) for a in train_losses])
                    TL = np.stack([a[:L] for a in train_losses], axis=0)
                    mean_tr = np.nanmean(TL, axis=0)
                    sem_tr = sem(TL, axis=0)
                    xs = np.arange(1, L + 1)
                    plt.plot(xs, mean_tr, label="Train (Mean)", color="tab:blue")
                    plt.fill_between(
                        xs,
                        mean_tr - sem_tr,
                        mean_tr + sem_tr,
                        color="tab:blue",
                        alpha=0.2,
                        label="Train (SEM)",
                    )
                if len(val_losses) > 0:
                    L = min([len(a) for a in val_losses])
                    VL = np.stack([a[:L] for a in val_losses], axis=0)
                    mean_va = np.nanmean(VL, axis=0)
                    sem_va = sem(VL, axis=0)
                    xs = np.arange(1, L + 1)
                    plt.plot(xs, mean_va, label="Val (Mean)", color="tab:orange")
                    plt.fill_between(
                        xs,
                        mean_va - sem_va,
                        mean_va + sem_va,
                        color="tab:orange",
                        alpha=0.2,
                        label="Val (SEM)",
                    )
                plt.legend()
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"{ds_name} - Loss Curves\n{subtitle}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{ds_name}_aggregated_loss_curves.png")
                )
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss curves for {ds_name}: {e}")
            plt.close()

        # Aggregate test metrics across runs and bar plot with error bars
        try:
            metrics_list = []
            for r in runs:
                test_entries = r.get("metrics", {}).get("test", [])
                if isinstance(test_entries, list) and len(test_entries) > 0:
                    # take first entry's dict
                    m = (
                        dict(test_entries[0][1])
                        if isinstance(test_entries[0], (list, tuple))
                        else dict(test_entries[0])
                    )
                    # Only keep numeric keys
                    usable = {
                        k: float(v)
                        for k, v in m.items()
                        if isinstance(v, (int, float, np.floating))
                    }
                    if len(usable) > 0:
                        metrics_list.append(usable)
            if len(metrics_list) > 0:
                keys = ["acc", "macro_f1", "top3", "loss"]
                vals = []
                for k in keys:
                    arr = [d.get(k, np.nan) for d in metrics_list]
                    vals.append(arr)
                means = [np.nanmean(a) for a in vals]
                errors = [sem(a) for a in vals]
                xs = np.arange(len(keys))
                plt.figure()
                plt.bar(
                    xs,
                    means,
                    yerr=errors,
                    capsize=4,
                    color=["tab:green", "tab:purple", "tab:red", "tab:gray"],
                    alpha=0.8,
                    label="Mean ± SEM",
                )
                plt.xticks(xs, keys)
                plt.ylabel("Value")
                plt.title(
                    f"{ds_name} - Aggregated Test Metrics\nAggregated across runs | Next-activity"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(working_dir, f"{ds_name}_aggregated_test_metrics.png")
                )
                plt.close()
                # Print metrics
                print(
                    f"{ds_name} | Test metrics (mean ± SEM): "
                    + ", ".join(
                        [f"{k}={m:.4f}±{e:.4f}" for k, m, e in zip(keys, means, errors)]
                    )
                )
        except Exception as e:
            print(f"Error creating aggregated test metrics for {ds_name}: {e}")
            plt.close()

        # Aggregate Top-3 vs Prefix Length across runs (mean ± SEM), downsample xs to at most 5
        try:
            per_run_maps = []
            all_L = set()
            for r in runs:
                pref = r.get("prefix_lens", [])
                flags = r.get("top3_flags", [])
                if len(pref) > 0 and len(flags) > 0:
                    d = defaultdict(list)
                    for L, f in zip(pref, flags):
                        d[int(L)].append(int(f))
                    # per-run mean per L
                    run_map = {L: float(np.mean(v)) for L, v in d.items() if len(v) > 0}
                    if len(run_map) > 0:
                        per_run_maps.append(run_map)
                        all_L.update(run_map.keys())
            if len(per_run_maps) > 0:
                xs_all = sorted(list(all_L))
                xs_plot = downsample_xs(xs_all, max_points=5)
                # Build matrix R x X with NaNs for missing
                R = len(per_run_maps)
                X = len(xs_plot)
                M = np.full((R, X), np.nan, dtype=float)
                for i, run_map in enumerate(per_run_maps):
                    for j, L in enumerate(xs_plot):
                        if L in run_map:
                            M[i, j] = run_map[L]
                mean_y = np.nanmean(M, axis=0)
                err_y = sem(M, axis=0)
                plt.figure()
                plt.errorbar(
                    xs_plot, mean_y, yerr=err_y, fmt="-o", capsize=3, label="Mean ± SEM"
                )
                plt.ylim(0.0, 1.0)
                plt.xlabel("Prefix Length")
                plt.ylabel("Top-3 Accuracy")
                plt.title(
                    f"{ds_name} - Top-3 Accuracy vs Prefix Length\nAggregated across runs | Next-activity"
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{ds_name}_aggregated_top3_vs_prefixlen.png"
                    )
                )
                plt.close()
        except Exception as e:
            print(
                f"Error creating aggregated Top-3 vs Prefix Length for {ds_name}: {e}"
            )
            plt.close()

        # Aggregated confusion matrix (sum over runs)
        try:
            cm_sum = None
            for r in runs:
                y_true = r.get("ground_truth", [])
                y_pred = r.get("predictions", [])
                if len(y_true) > 0 and len(y_pred) > 0:
                    # compute confusion for the labels present in this run
                    labels = sorted(set(y_true) | set(y_pred))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    if cm_sum is None:
                        cm_sum = cm
                        cm_labels = labels
                    else:
                        # align labels
                        all_labels = sorted(set(cm_labels) | set(labels))
                        # expand cm_sum
                        new_sum = np.zeros(
                            (len(all_labels), len(all_labels)), dtype=int
                        )
                        # map old indices
                        idx_old = {lab: i for i, lab in enumerate(cm_labels)}
                        idx_new = {lab: i for i, lab in enumerate(all_labels)}
                        for lab_i in cm_labels:
                            for lab_j in cm_labels:
                                new_sum[idx_new[lab_i], idx_new[lab_j]] += cm_sum[
                                    idx_old[lab_i], idx_old[lab_j]
                                ]
                        # add current cm
                        idx_cur = {lab: i for i, lab in enumerate(labels)}
                        for lab_i in labels:
                            for lab_j in labels:
                                new_sum[idx_new[lab_i], idx_new[lab_j]] += cm[
                                    idx_cur[lab_i], idx_cur[lab_j]
                                ]
                        cm_sum = new_sum
                        cm_labels = all_labels
            if cm_sum is not None:
                plt.figure(figsize=(6, 5))
                plt.imshow(cm_sum, aspect="auto", cmap="Blues")
                plt.colorbar()
                plt.title(
                    f"{ds_name} - Aggregated Confusion Matrix (Test)\nAggregated across runs | Next-activity"
                )
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{ds_name}_aggregated_confusion_matrix.png"
                    )
                )
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated confusion matrix for {ds_name}: {e}")
            plt.close()


if __name__ == "__main__":
    main()
