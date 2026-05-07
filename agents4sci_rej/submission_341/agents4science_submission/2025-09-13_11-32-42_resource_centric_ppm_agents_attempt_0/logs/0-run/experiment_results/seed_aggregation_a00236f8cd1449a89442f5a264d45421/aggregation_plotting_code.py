import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

from collections import defaultdict, OrderedDict


def se(a, axis=0):
    a = np.asarray(a, dtype=float)
    n = np.sum(~np.isnan(a), axis=axis)
    std = np.nanstd(a, axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return std / np.sqrt(np.maximum(n, 1))


def intersect_epoch_series(list_of_series):
    # series are lists of y-values per epoch (1..T). We align to min length
    if len(list_of_series) == 0:
        return []
    min_len = min(len(s) for s in list_of_series if len(s) > 0)
    if min_len == 0:
        return []
    arr = np.stack(
        [np.asarray(s[:min_len], dtype=float) for s in list_of_series], axis=0
    )
    mean = np.nanmean(arr, axis=0)
    stderr = se(arr, axis=0)
    epochs = np.arange(1, min_len + 1)
    return epochs, mean, stderr


def aggregate_prefix_len(flags_list, lens_list):
    # flags_list: list of arrays of 0/1; lens_list: list of arrays of lengths, matched per run
    # Return dict: L -> (mean, se, n_runs_contributed)
    bucket = defaultdict(list)
    for flags, lens in zip(flags_list, lens_list):
        if flags is None or lens is None:
            continue
        if len(flags) == 0 or len(lens) == 0:
            continue
        # build per-L mean for this run first (avoid per-sample unequal weights across runs)
        d = defaultdict(list)
        for L, f in zip(lens, flags):
            try:
                d[int(L)].append(int(f))
            except:
                continue
        for L, vals in d.items():
            if len(vals) > 0:
                bucket[L].append(np.mean(vals))
    if not bucket:
        return {}
    out = {}
    for L in sorted(bucket.keys()):
        vals = np.array(bucket[L], dtype=float)
        m = float(np.mean(vals))
        s = float(se(vals, axis=0)) if vals.size > 1 else 0.0
        out[L] = (m, s, len(vals))
    return out


def safe_get_losses(ed, split):
    # returns list of y-values from ed['losses'][split] which is list of (epoch, val)
    try:
        items = ed.get("losses", {}).get(split, [])
        ys = [y for (_, y) in items]
        return ys
    except Exception:
        return []


def main():
    # Load multiple experiment_data.npy files
    try:
        experiment_data_path_list = [
            "experiments/2025-09-13_11-32-42_resource_centric_ppm_agents_attempt_0/logs/0-run/experiment_results/experiment_582151be1232410a9f4163cd33e1b808_proc_404246/experiment_data.npy",
            "experiments/2025-09-13_11-32-42_resource_centric_ppm_agents_attempt_0/logs/0-run/experiment_results/experiment_a3b6f5d7af4646e98fe21372820c42d9_proc_404247/experiment_data.npy",
            "None/experiment_data.npy",
        ]
        all_experiment_data = []
        for experiment_data_path in experiment_data_path_list:
            try:
                exp = np.load(
                    os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
                    allow_pickle=True,
                ).item()
                all_experiment_data.append(exp)
            except Exception as e:
                print(f"Error loading experiment data: {e}")
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        all_experiment_data = []

    # Index by dataset name across runs
    datasets_union = set()
    for exp in all_experiment_data:
        datasets_union.update(exp.keys())

    # Collect and print aggregated test metrics
    aggregated_report = {}
    for ds in sorted(datasets_union):
        test_metrics_runs = []
        train_losses_runs = []
        val_losses_runs = []
        pref_flags_runs = []
        pref_lens_runs = []
        for exp in all_experiment_data:
            if ds not in exp:
                continue
            ed = exp[ds]
            # test metrics
            try:
                test_list = ed.get("metrics", {}).get("test", [])
                if (
                    len(test_list) > 0
                    and isinstance(test_list[0], (list, tuple))
                    and isinstance(test_list[0][1], dict)
                ):
                    tm = test_list[0][1]
                    # ensure required keys
                    keys = ["loss", "acc", "macro_f1", "top3"]
                    if all(k in tm for k in keys):
                        test_metrics_runs.append(
                            [tm["loss"], tm["acc"], tm["macro_f1"], tm["top3"]]
                        )
            except Exception:
                pass
            # losses
            tl = safe_get_losses(ed, "train")
            vl = safe_get_losses(ed, "val")
            if len(tl) > 0:
                train_losses_runs.append(tl)
            if len(vl) > 0:
                val_losses_runs.append(vl)
            # prefix lens and flags
            try:
                flags = ed.get("top3_flags", [])
                lens = ed.get("prefix_lens", [])
                if len(flags) > 0 and len(lens) > 0:
                    pref_flags_runs.append(np.array(flags))
                    pref_lens_runs.append(np.array(lens))
            except Exception:
                pass

        # Aggregate test metrics
        if len(test_metrics_runs) > 0:
            arr = np.asarray(test_metrics_runs, dtype=float)  # shape (R, 4)
            mean = np.nanmean(arr, axis=0)
            stderr_vals = se(arr, axis=0)
            aggregated_report[ds] = {"mean": mean, "se": stderr_vals, "n": arr.shape[0]}

        # Plot aggregated loss curves (mean ± SE)
        try:
            if len(train_losses_runs) > 0:
                ep, mu, se_vals = intersect_epoch_series(train_losses_runs)
                if len(mu) > 0:
                    plt.figure()
                    plt.plot(ep, mu, label="Train mean", color="tab:blue")
                    plt.fill_between(
                        ep,
                        mu - se_vals,
                        mu + se_vals,
                        color="tab:blue",
                        alpha=0.2,
                        label="Train SE",
                    )
                    # if val available
                    if len(val_losses_runs) > 0:
                        epv, muv, sev = intersect_epoch_series(val_losses_runs)
                        if len(muv) > 0:
                            plt.plot(epv, muv, label="Val mean", color="tab:orange")
                            plt.fill_between(
                                epv,
                                muv - sev,
                                muv + sev,
                                color="tab:orange",
                                alpha=0.2,
                                label="Val SE",
                            )
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.title(f"Aggregated Loss Curves - {ds}\nMean±SE; Next-activity")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            working_dir, f"{ds}_aggregated_loss_curves_mean_se.png"
                        )
                    )
                    plt.close()
                else:
                    plt.close()
        except Exception as e:
            print(f"Error creating aggregated loss plot for {ds}: {e}")
            plt.close()

        # Plot aggregated Top-3 vs prefix length with error bars
        try:
            if len(pref_flags_runs) > 0 and len(pref_lens_runs) > 0:
                agg = aggregate_prefix_len(pref_flags_runs, pref_lens_runs)
                if len(agg) > 0:
                    xs = sorted(agg.keys())
                    means = [agg[L][0] for L in xs]
                    ses = [agg[L][1] for L in xs]
                    plt.figure()
                    plt.errorbar(
                        xs,
                        means,
                        yerr=ses,
                        fmt="-o",
                        capsize=3,
                        label="Mean Top-3 ± SE",
                    )
                    plt.xlabel("Prefix Length")
                    plt.ylabel("Top-3 Accuracy")
                    plt.title(
                        f"Aggregated Top-3 Accuracy vs Prefix Length - {ds}\nMean±SE; Next-activity"
                    )
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            working_dir,
                            f"{ds}_aggregated_top3_vs_prefixlen_mean_se.png",
                        )
                    )
                    plt.close()
                else:
                    plt.close()
        except Exception as e:
            print(f"Error creating aggregated Top-3 vs prefix length for {ds}: {e}")
            plt.close()

        # Bar chart of aggregated test metrics (loss, acc, macro_f1, top3) with error bars
        try:
            if ds in aggregated_report:
                m = aggregated_report[ds]["mean"]
                s = aggregated_report[ds]["se"]
                labels = ["loss", "acc", "macro_f1", "top3"]
                x = np.arange(len(labels))
                plt.figure()
                plt.bar(x, m, yerr=s, capsize=4)
                plt.xticks(x, labels)
                plt.ylabel("Metric value")
                plt.title(f"Aggregated Test Metrics - {ds}\nMean±SE; Next-activity")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        working_dir, f"{ds}_aggregated_test_metrics_bar_mean_se.png"
                    )
                )
                plt.close()
        except Exception as e:
            print(f"Error creating aggregated test metrics bar for {ds}: {e}")
            plt.close()

    # Print aggregated metrics
    for ds, rep in aggregated_report.items():
        m = rep["mean"]
        s = rep["se"]
        n = rep["n"]
        print(
            f"{ds} | runs={n} | Test mean±SE: loss={m[0]:.4f}±{s[0]:.4f}, acc={m[1]:.4f}±{s[1]:.4f}, macro_f1={m[2]:.4f}±{s[2]:.4f}, top3={m[3]:.4f}±{s[3]:.4f}"
        )


if __name__ == "__main__":
    main()
