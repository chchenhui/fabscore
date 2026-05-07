"""
Collect optimized RRCS results across r_cap values and seeds.
Produces per-rcap summary JSONs and a combined comparison JSON.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

SEEDS = [42, 123, 456]
R_CAPS = [2.0, 5.0, 8.0]
BASE = Path(__file__).resolve().parents[1] / "examples" / "nanogpt"
RESULTS_DIR = BASE / "results"
LOGS_DIR = RESULTS_DIR / "logs"


def r_cap_tag(r_cap):
    return str(r_cap).replace(".", "p")


def load_diagnostics(r_cap, seed):
    tag = r_cap_tag(r_cap)
    csv_path = LOGS_DIR / f"rrcs_opt_rcap{tag}_seed{seed}" / "diagnostics.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_summary(r_cap, seed):
    tag = r_cap_tag(r_cap)
    path = LOGS_DIR / f"rrcs_opt_rcap{tag}_seed{seed}" / "summary.json"
    with open(path) as f:
        return json.load(f)


def compute_param_drift(r_cap, seed):
    tag = r_cap_tag(r_cap)
    run_dir = LOGS_DIR / f"rrcs_opt_rcap{tag}_seed{seed}"
    init_path = run_dir / "h_res_logits_init.pt"
    ckpt_path = run_dir / "ckpt.pt"

    init_snap = torch.load(init_path, map_location="cpu", weights_only=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    total_drift_sq = 0.0
    count = 0
    for name, init_val in init_snap.items():
        ckpt_key = name + ".H_res_logits"
        if ckpt_key in ckpt["model"]:
            final_val = ckpt["model"][ckpt_key]
            diff = (final_val.float() - init_val.float())
            total_drift_sq += diff.pow(2).sum().item()
            count += 1

    return total_drift_sq ** 0.5, count


def compute_grad_spike_ratio(diag_rows):
    grad_norms = [(int(r["iter"]), r["grad_norm_global"]) for r in diag_rows]
    grad_norms = [(it, gn) for it, gn in grad_norms if it > 200]
    if len(grad_norms) < 101:
        return 0.0
    gn_vals = [gn for _, gn in grad_norms]
    max_ratio = 0.0
    for i in range(100, len(gn_vals)):
        window = gn_vals[max(0, i - 100):i]
        median_val = float(np.median(window))
        if median_val > 1e-10:
            ratio = gn_vals[i] / median_val
            max_ratio = max(max_ratio, ratio)
    return max_ratio


def collect_one_rcap(r_cap):
    results = {"condition": f"rrcs_opt_rcap{r_cap}", "r_cap": r_cap, "seeds": SEEDS, "per_seed": {}}

    val_losses = []
    best_val_losses = []
    grad_medians = []
    drifts = []

    for seed in SEEDS:
        summary = load_summary(r_cap, seed)
        diag_rows = load_diagnostics(r_cap, seed)

        best_val = summary["best_val_loss"]
        final_val = summary["last_eval"]["val"]
        val_losses.append(final_val)
        best_val_losses.append(best_val)

        post_warmup = [r for r in diag_rows if int(r["iter"]) > 200]
        all_h_res_grads = [r["h_res_grad_median"] for r in post_warmup]
        grad_median = float(np.median(all_h_res_grads)) if all_h_res_grads else 0.0
        grad_mean = float(np.mean(all_h_res_grads)) if all_h_res_grads else 0.0
        grad_medians.append(grad_median)

        drift_fro, drift_count = compute_param_drift(r_cap, seed)
        drifts.append(drift_fro)

        final_row = diag_rows[-1]
        spike_ratio = compute_grad_spike_ratio(diag_rows)

        sinkhorn_ranges = [r["sinkhorn_range_mean"] for r in post_warmup]
        s_means = [r["rrcs_s_mean"] for r in post_warmup]
        s_mins = [r["rrcs_s_min"] for r in post_warmup]
        s_fracs = [r["rrcs_s_frac_active"] for r in post_warmup]

        seed_result = {
            "best_val_loss": best_val,
            "final_val_loss": final_val,
            "final_train_loss": summary["last_eval"]["train"],
            "h_res_grad_median_post_warmup": grad_median,
            "h_res_grad_mean_post_warmup": grad_mean,
            "h_res_param_drift_fro": drift_fro,
            "h_res_param_drift_layers": drift_count,
            "ds_row_error_mean_final": final_row["ds_row_error_mean"],
            "ds_row_error_max_final": final_row["ds_row_error_max"],
            "ds_col_error_mean_final": final_row["ds_col_error_mean"],
            "ds_col_error_max_final": final_row["ds_col_error_max"],
            "entropy_mean_final": final_row["entropy_mean"],
            "grad_norm_spike_ratio": spike_ratio,
            "sinkhorn_range_mean": float(np.mean(sinkhorn_ranges)) if sinkhorn_ranges else 0.0,
            "sinkhorn_range_max": float(np.max(sinkhorn_ranges)) if sinkhorn_ranges else 0.0,
            "sinkhorn_range_p50": float(np.median(sinkhorn_ranges)) if sinkhorn_ranges else 0.0,
            "elapsed_s": summary["elapsed_s"],
            "rrcs_s_mean_over_steps": float(np.mean(s_means)) if s_means else 0.0,
            "rrcs_s_min_over_steps": float(np.min(s_mins)) if s_mins else 0.0,
            "rrcs_s_frac_active_mean": float(np.mean(s_fracs)) if s_fracs else 0.0,
            "rrcs_s_mean_std_over_steps": float(np.std(s_means)) if s_means else 0.0,
        }
        results["per_seed"][str(seed)] = seed_result

    val_arr = np.array(val_losses)
    best_val_arr = np.array(best_val_losses)
    grad_arr = np.array(grad_medians)
    drift_arr = np.array(drifts)

    results["aggregate"] = {
        "val_loss_mean": float(val_arr.mean()),
        "val_loss_std": float(val_arr.std()),
        "val_loss_min": float(val_arr.min()),
        "val_loss_max": float(val_arr.max()),
        "best_val_loss_mean": float(best_val_arr.mean()),
        "best_val_loss_std": float(best_val_arr.std()),
        "h_res_grad_median_mean": float(grad_arr.mean()),
        "h_res_grad_median_std": float(grad_arr.std()),
        "h_res_param_drift_fro_mean": float(drift_arr.mean()),
        "h_res_param_drift_fro_std": float(drift_arr.std()),
    }
    return results


def main():
    all_results = {}
    for r_cap in R_CAPS:
        tag = r_cap_tag(r_cap)
        print(f"\n=== r_cap={r_cap} ===")
        try:
            results = collect_one_rcap(r_cap)
            all_results[str(r_cap)] = results

            out_path = RESULTS_DIR / f"rrcs_opt_rcap{tag}_summary.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Written to {out_path}")

            agg = results["aggregate"]
            print(f"  Val loss: {agg['best_val_loss_mean']:.4f} +/- {agg['best_val_loss_std']:.4f}")
            print(f"  H_res grad median: {agg['h_res_grad_median_mean']:.4e}")
            print(f"  H_res param drift: {agg['h_res_param_drift_fro_mean']:.4e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    comparison_path = RESULTS_DIR / "rrcs_opt_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison written to {comparison_path}")

    print("\n=== Summary Comparison ===")
    print(f"{'r_cap':>6} | {'Val Loss':>14} | {'Grad Median':>14} | {'Param Drift':>14}")
    print("-" * 60)
    baseline_grad = 1.9e-15
    baseline_drift = 2.05e-7
    for r_cap_str, res in all_results.items():
        agg = res["aggregate"]
        grad_ratio = agg["h_res_grad_median_mean"] / baseline_grad if baseline_grad > 0 else 0
        drift_ratio = agg["h_res_param_drift_fro_mean"] / baseline_drift if baseline_drift > 0 else 0
        print(f"{r_cap_str:>6} | {agg['best_val_loss_mean']:.4f}+/-{agg['best_val_loss_std']:.4f} | {agg['h_res_grad_median_mean']:.4e} ({grad_ratio:.0f}x) | {agg['h_res_param_drift_fro_mean']:.4e} ({drift_ratio:.0f}x)")

    print(f"\n{'orig30':>6} | 4.7714+/-0.0092 | 1.9323e-15 (1x) | 2.0512e-07 (1x)")
    print(f"{'mhc_def':>6} | 4.7718+/-0.0114 | 0.0000e+00 (0x) | 0.0000e+00 (0x)")


if __name__ == "__main__":
    main()
