"""
Collect mHC baseline results from 3-seed training runs.

Reads summary.json, diagnostics.csv, and h_res_logits checkpoints
to produce results/mhc_default_summary.json.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

SEEDS = [42, 123, 456]
BASE = Path(__file__).resolve().parents[1] / "examples" / "nanogpt"
RESULTS_DIR = BASE / "results"
LOGS_DIR = RESULTS_DIR / "logs"


def load_diagnostics(seed):
    csv_path = LOGS_DIR / f"mhc_default_seed{seed}" / "diagnostics.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_summary(seed):
    path = LOGS_DIR / f"mhc_default_seed{seed}" / "summary.json"
    with open(path) as f:
        return json.load(f)


def load_checkpoint_h_res(seed):
    ckpt_path = LOGS_DIR / f"mhc_default_seed{seed}" / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h_res_params = {}
    for k, v in ckpt["model"].items():
        if "H_res_logits" in k:
            h_res_params[k] = v
    return h_res_params


def load_init_snapshot(seed):
    path = LOGS_DIR / f"mhc_default_seed{seed}" / "h_res_logits_init.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_param_drift(seed):
    init_snap = load_init_snapshot(seed)
    final_params = load_checkpoint_h_res(seed)

    total_drift_sq = 0.0
    count = 0
    for name, init_val in init_snap.items():
        matched = None
        for k, v in final_params.items():
            if name in k or k.endswith(name.split(".")[-1]):
                if v.shape == init_val.shape:
                    matched = v
                    break
        if matched is None:
            for k, v in final_params.items():
                if "H_res_logits" in k and v.shape == init_val.shape:
                    key_parts = name.split(".")
                    k_parts = k.split(".")
                    if any(p == kp for p, kp in zip(key_parts, k_parts)):
                        matched = v
                        break

        if matched is not None:
            diff = (matched.float() - init_val.float())
            total_drift_sq += diff.pow(2).sum().item()
            count += 1

    if count == 0:
        init_keys = sorted(init_snap.keys())
        final_keys = sorted(final_params.keys())
        init_vals = list(init_snap.values())
        final_vals = list(final_params.values())
        if len(init_vals) == len(final_vals):
            for iv, fv in zip(init_vals, final_vals):
                if iv.shape == fv.shape:
                    diff = (fv.float() - iv.float())
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


def main():
    results = {"condition": "mhc_default", "seeds": SEEDS, "per_seed": {}}

    val_losses = []
    grad_medians = []
    drifts = []

    for seed in SEEDS:
        summary = load_summary(seed)
        diag_rows = load_diagnostics(seed)

        best_val = summary["best_val_loss"]
        final_val = summary["last_eval"]["val"]
        val_losses.append(final_val)

        post_warmup = [r for r in diag_rows if int(r["iter"]) > 200]
        all_h_res_grad_medians = [r["h_res_grad_median"] for r in post_warmup]
        overall_grad_median = float(np.median(all_h_res_grad_medians)) if all_h_res_grad_medians else 0.0
        overall_grad_mean = float(np.mean(all_h_res_grad_medians)) if all_h_res_grad_medians else 0.0
        grad_medians.append(overall_grad_median)

        drift_fro, drift_count = compute_param_drift(seed)
        drifts.append(drift_fro)

        final_row = diag_rows[-1]

        spike_ratio = compute_grad_spike_ratio(diag_rows)

        sinkhorn_ranges = [r["sinkhorn_range_mean"] for r in post_warmup]

        seed_result = {
            "best_val_loss": best_val,
            "final_val_loss": final_val,
            "final_train_loss": summary["last_eval"]["train"],
            "h_res_grad_median_post_warmup": overall_grad_median,
            "h_res_grad_mean_post_warmup": overall_grad_mean,
            "h_res_param_drift_fro": drift_fro,
            "h_res_param_drift_layers": drift_count,
            "ds_row_error_mean_final": final_row["ds_row_error_mean"],
            "ds_row_error_max_final": final_row["ds_row_error_max"],
            "ds_col_error_mean_final": final_row["ds_col_error_mean"],
            "ds_col_error_max_final": final_row["ds_col_error_max"],
            "entropy_mean_final": final_row["entropy_mean"],
            "grad_norm_spike_ratio": spike_ratio,
            "sinkhorn_range_mean": float(np.mean(sinkhorn_ranges)),
            "sinkhorn_range_max": float(np.max(sinkhorn_ranges)),
            "sinkhorn_range_p50": float(np.median(sinkhorn_ranges)),
            "elapsed_s": summary["elapsed_s"],
        }
        results["per_seed"][str(seed)] = seed_result

    val_arr = np.array(val_losses)
    grad_arr = np.array(grad_medians)
    drift_arr = np.array(drifts)

    results["aggregate"] = {
        "val_loss_mean": float(val_arr.mean()),
        "val_loss_std": float(val_arr.std()),
        "val_loss_min": float(val_arr.min()),
        "val_loss_max": float(val_arr.max()),
        "h_res_grad_median_mean": float(grad_arr.mean()),
        "h_res_grad_median_std": float(grad_arr.std()),
        "h_res_param_drift_fro_mean": float(drift_arr.mean()),
        "h_res_param_drift_fro_std": float(drift_arr.std()),
    }

    out_path = RESULTS_DIR / "mhc_default_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
