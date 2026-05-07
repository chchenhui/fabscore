import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load multiple experiment_data.npy files
try:
    experiment_data_path_list = [
        "None/experiment_data.npy",
        "None/experiment_data.npy",
        "experiments/2025-09-13_11-32-42_resource_centric_ppm_agents_attempt_0/logs/0-run/experiment_results/experiment_8fe8449b53944b169ac6bb9b756455fa_proc_436257/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        try:
            root = os.getenv("AI_SCIENTIST_ROOT", "")
            full_path = (
                os.path.join(root, experiment_data_path)
                if root
                else experiment_data_path
            )
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"Not found: {full_path}")
            experiment_data = np.load(full_path, allow_pickle=True).item()
            all_experiment_data.append(experiment_data)
            print(f"Loaded experiment_data from: {full_path}")
        except Exception as e:
            print(f"Error loading experiment data: {e}")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# Aggregate across runs per dataset
def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    n = np.sum(~np.isnan(a), axis=axis)
    s = np.nanstd(a, axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return s / np.sqrt(np.maximum(n, 1))


# Collect per-dataset containers
datasets = {}
for run_idx, edict in enumerate(all_experiment_data):
    if not isinstance(edict, dict):
        continue
    for ds_name, ed in edict.items():
        d = datasets.setdefault(
            ds_name,
            {
                "train_losses": [],  # list of arrays
                "val_losses": [],  # list of arrays
                "epochs": [],  # list of arrays
                "test_metrics": [],  # list of dicts
                "prefix_len_flags": [],  # list of tuples (prefix_lens, flags)
            },
        )
        # Loss curves
        losses = ed.get("losses", {})
        tr = [y for (_, y) in losses.get("train", [])]
        va = [y for (_, y) in losses.get("val", [])]
        ep = (
            [x for (x, _) in losses.get("train", [])]
            if len(losses.get("train", [])) > 0
            else list(range(1, len(tr) + 1))
        )
        if len(tr) > 0:
            d["train_losses"].append(np.array(tr, dtype=float))
            d["epochs"].append(np.array(ep, dtype=int))
        if len(va) > 0:
            d["val_losses"].append(np.array(va, dtype=float))
        # Test metrics
        try:
            test_items = ed.get("metrics", {}).get("test", [])
            if len(test_items) > 0 and isinstance(test_items[0], (list, tuple)):
                tm = dict(test_items[0][1])
                d["test_metrics"].append(tm)
        except Exception as e:
            print(f"Warning parsing test metrics for {ds_name}: {e}")
        # Prefix-len top3 flags
        pref = ed.get("prefix_lens", [])
        flags = ed.get("top3_flags", [])
        if len(pref) > 0 and len(flags) > 0:
            d["prefix_len_flags"].append(
                (np.array(pref, dtype=int), np.array(flags, dtype=int))
            )


# Helper: pad sequences with NaN to same length
def pad_to_same_length(arr_list):
    if len(arr_list) == 0:
        return np.array([])
    max_len = max(len(a) for a in arr_list)
    out = np.full((len(arr_list), max_len), np.nan, dtype=float)
    for i, a in enumerate(arr_list):
        L = len(a)
        out[i, :L] = a
    return out


# Aggregate and plot per dataset
agg_summary = {}  # for printing aggregated test metrics
for ds_name, d in datasets.items():
    # Aggregate loss curves
    try:
        # Train loss: mean and SEM over runs (epoch-aligned by index)
        if len(d["train_losses"]) > 0:
            TL = pad_to_same_length(d["train_losses"])
            tl_mean = np.nanmean(TL, axis=0)
            tl_sem = sem(TL, axis=0)
            # Validation loss (may be missing in some runs)
            VL = (
                pad_to_same_length(d["val_losses"])
                if len(d["val_losses"]) > 0
                else np.array([])
            )
            if VL.size > 0:
                vl_mean = np.nanmean(VL, axis=0)
                vl_sem = sem(VL, axis=0)
            # Epochs (use longest)
            max_len = len(tl_mean)
            epochs = np.arange(1, max_len + 1)
            plt.figure()
            plt.plot(epochs, tl_mean, color="tab:blue", label="Train mean")
            plt.fill_between(
                epochs,
                tl_mean - tl_sem,
                tl_mean + tl_sem,
                color="tab:blue",
                alpha=0.2,
                label="Train SEM",
            )
            if VL.size > 0:
                e_v = np.arange(1, len(vl_mean) + 1)
                plt.plot(e_v, vl_mean, color="tab:orange", label="Val mean")
                plt.fill_between(
                    e_v,
                    vl_mean - vl_sem,
                    vl_mean + vl_sem,
                    color="tab:orange",
                    alpha=0.2,
                    label="Val SEM",
                )
            plt.legend()
            plt.title(
                f"Aggregated Loss Curves - {ds_name}\nDataset: {ds_name}; Mean ± SEM across runs"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.tight_layout()
            save_path = os.path.join(
                working_dir, f"{ds_name}_aggregated_loss_curves.png"
            )
            plt.savefig(save_path)
            plt.close()
        else:
            # nothing to plot
            pass
    except Exception as e:
        print(f"Error creating aggregated loss plot for {ds_name}: {e}")
        plt.close()

    # Aggregate Top-3 vs Prefix Length with SEM
    try:
        if len(d["prefix_len_flags"]) > 0:
            # Build dictionary: prefix_len -> list of run means
            # First, for each run, compute per-length mean flag
            per_run_dicts = []
            for pref, flags in d["prefix_len_flags"]:
                rdict = {}
                # group by length
                unique_L = np.unique(pref)
                for L in unique_L:
                    m = np.mean(flags[pref == L].astype(float))
                    rdict[int(L)] = m
                per_run_dicts.append(rdict)
            # Collect union of lengths and form matrix
            all_L = sorted(set().union(*[set(r.keys()) for r in per_run_dicts]))
            M = np.full((len(per_run_dicts), len(all_L)), np.nan, dtype=float)
            for i, r in enumerate(per_run_dicts):
                for j, L in enumerate(all_L):
                    if L in r:
                        M[i, j] = r[L]
            m_mean = np.nanmean(M, axis=0)
            m_sem = sem(M, axis=0)
            plt.figure()
            plt.plot(all_L, m_mean, color="tab:green", marker="o", label="Top-3 mean")
            plt.fill_between(
                all_L,
                m_mean - m_sem,
                m_mean + m_sem,
                color="tab:green",
                alpha=0.2,
                label="Top-3 SEM",
            )
            plt.ylim(0.0, 1.0)
            plt.xlabel("Prefix Length")
            plt.ylabel("Top-3 Accuracy")
            plt.title(
                f"Top-3 Accuracy vs Prefix Length (Aggregated) - {ds_name}\nDataset: {ds_name}; Mean ± SEM across runs"
            )
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(
                working_dir, f"{ds_name}_aggregated_top3_vs_prefixlen.png"
            )
            plt.savefig(save_path)
            plt.close()
    except Exception as e:
        print(f"Error creating aggregated Top-3 vs Prefix Length for {ds_name}: {e}")
        plt.close()

    # Aggregate test metrics bar chart per dataset
    try:
        if len(d["test_metrics"]) > 0:
            keys = ["acc", "macro_f1", "top3", "loss"]
            vals = {k: [] for k in keys}
            for tm in d["test_metrics"]:
                for k in keys:
                    if k in tm:
                        vals[k].append(float(tm[k]))
            means = []
            sems = []
            labels = []
            for k in keys:
                arr = np.array(vals[k], dtype=float)
                if arr.size == 0:
                    continue
                labels.append(k)
                means.append(np.mean(arr))
                # use sample sem if n>1 else 0
                if arr.size > 1:
                    sems.append(np.std(arr, ddof=1) / np.sqrt(arr.size))
                else:
                    sems.append(0.0)
            if len(labels) > 0:
                x = np.arange(len(labels))
                plt.figure()
                plt.bar(
                    x,
                    means,
                    yerr=sems,
                    capsize=4,
                    color="tab:purple",
                    alpha=0.8,
                    label="Mean ± SEM",
                )
                plt.xticks(x, labels)
                plt.ylabel("Metric value")
                plt.title(
                    f"Aggregated Test Metrics - {ds_name}\nDataset: {ds_name}; Mean ± SEM across runs"
                )
                plt.legend()
                plt.tight_layout()
                save_path = os.path.join(
                    working_dir, f"{ds_name}_aggregated_test_metrics.png"
                )
                plt.savefig(save_path)
                plt.close()
            # Save summary for printing
            agg_summary[ds_name] = {
                labels[i]: (means[i], sems[i]) for i in range(len(labels))
            }
    except Exception as e:
        print(f"Error creating aggregated test metrics for {ds_name}: {e}")
        plt.close()

# Optional: Cross-dataset comparison plots (e.g., accuracy across datasets)
try:
    if len(agg_summary) > 0:
        # Build accuracy comparison across datasets
        labels = []
        means = []
        errs = []
        for ds_name, dsum in agg_summary.items():
            if "acc" in dsum:
                labels.append(ds_name)
                means.append(dsum["acc"][0])
                errs.append(dsum["acc"][1])
        if len(labels) > 0:
            x = np.arange(len(labels))
            plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
            plt.bar(
                x,
                means,
                yerr=errs,
                capsize=4,
                color="tab:blue",
                alpha=0.8,
                label="Accuracy Mean ± SEM",
            )
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Accuracy")
            plt.title(
                "Cross-Dataset Test Accuracy (Aggregated)\nDatasets: multiple; Mean ± SEM across runs"
            )
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(
                working_dir, "cross_dataset_aggregated_accuracy.png"
            )
            plt.savefig(save_path)
            plt.close()
except Exception as e:
    print(f"Error creating cross-dataset comparison: {e}")
    plt.close()

# Print aggregated metrics
if len(agg_summary) > 0:
    for ds_name, metrics in agg_summary.items():
        parts = []
        for k, (m, e) in metrics.items():
            parts.append(f"{k}= {m:.4f} ± {e:.4f}")
        print(f"{ds_name} | " + ", ".join(parts))
else:
    print("No aggregated metrics available to print.")
