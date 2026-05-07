"""
Collect r_cap ablation results (r_cap=20, 30, 40) for seed=42 and produce
comparison table, JSON, and 3 figures.
"""

import csv
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1] / "examples" / "nanogpt"
LOGS_DIR = BASE / "results" / "logs"
FIG_DIR = BASE / "results" / "figures"
TRAIN_LOGS = Path(__file__).resolve().parents[2] / ".train_service_logs"

RCAP_DIRS = {
    20: LOGS_DIR / "rrcs_r20_seed42",
    30: LOGS_DIR / "rrcs_seed42",
    40: LOGS_DIR / "rrcs_r40_seed42",
}


def load_diagnostics(run_dir):
    csv_path = run_dir / "diagnostics.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_summary(run_dir):
    with open(run_dir / "summary.json") as f:
        return json.load(f)


def compute_param_drift(run_dir):
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
            diff = final_val.float() - init_val.float()
            total_drift_sq += diff.pow(2).sum().item()
            count += 1
    return total_drift_sq ** 0.5, count


def parse_val_losses_from_logs():
    """Parse val loss data from all train service logs, keyed by run_id."""
    val_data = {}
    if not TRAIN_LOGS.exists():
        return val_data
    for logdir in sorted(TRAIN_LOGS.iterdir()):
        log_file = logdir / "output.log"
        if not log_file.exists():
            continue
        try:
            text = log_file.read_text()
        except Exception:
            continue
        for line in text.splitlines():
            m = re.match(r"iter (\d+): train loss ([\d.]+), val loss ([\d.]+)", line)
            if m:
                it = int(m.group(1))
                vl = float(m.group(3))
                job_dir = logdir.name
                if job_dir not in val_data:
                    val_data[job_dir] = []
                val_data[job_dir].append((it, vl))
    return val_data


def find_val_losses_for_run(run_dir, all_val_data):
    """Find val loss trajectory for a run by matching the host name from run_metadata."""
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        host = meta.get("host", "")
        job_id = host.replace("-master-0", "")
        if job_id in all_val_data:
            return sorted(all_val_data[job_id])

    run_id = run_dir.name
    for job_id, pairs in all_val_data.items():
        log_file = TRAIN_LOGS / job_id / "output.log"
        if log_file.exists():
            try:
                text = log_file.read_text(errors="ignore")
                if run_id in text:
                    return sorted(pairs)
            except Exception:
                continue
    return []


def collect_metrics():
    results = {}
    for r_cap, run_dir in RCAP_DIRS.items():
        summary = load_summary(run_dir)
        diag = load_diagnostics(run_dir)

        post_warmup = [r for r in diag if int(r["iter"]) > 200]
        grad_medians = [r["h_res_grad_median"] for r in post_warmup]
        drift_fro, drift_layers = compute_param_drift(run_dir)
        final_row = diag[-1]

        s_fracs = [r["rrcs_s_frac_active"] for r in diag]
        pct_capped = 100.0 * np.mean([1.0 if f > 0 else 0.0 for f in s_fracs])

        results[r_cap] = {
            "best_val_loss": summary["best_val_loss"],
            "final_val_loss": summary["last_eval"]["val"],
            "h_res_grad_median": float(np.median(grad_medians)),
            "h_res_grad_mean": float(np.mean(grad_medians)),
            "h_res_param_drift_fro": drift_fro,
            "h_res_param_drift_layers": drift_layers,
            "ds_row_error_mean": final_row["ds_row_error_mean"],
            "ds_col_error_mean": final_row["ds_col_error_mean"],
            "entropy_mean": final_row["entropy_mean"],
            "pct_steps_capped": pct_capped,
            "rrcs_s_mean": float(np.mean([r["rrcs_s_mean"] for r in post_warmup])),
            "elapsed_s": summary["elapsed_s"],
            "diagnostics": diag,
        }
    return results


def make_figures(results):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {20: "#e63946", 30: "#457b9d", 40: "#2a9d8f"}
    labels = {20: "r_cap=20", 30: "r_cap=30", 40: "r_cap=40"}

    all_val_data = parse_val_losses_from_logs()

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for r_cap in [20, 30, 40]:
        run_dir = RCAP_DIRS[r_cap]
        vl_pairs = find_val_losses_for_run(run_dir, all_val_data)
        summary = load_summary(run_dir)
        final_it = summary.get("last_eval_iter", 5000)
        final_vl = summary["last_eval"]["val"]
        if vl_pairs:
            existing_iters = {p[0] for p in vl_pairs}
            if final_it not in existing_iters:
                vl_pairs.append((final_it, final_vl))
            vl_pairs = sorted(vl_pairs)
            iters, losses = zip(*vl_pairs)
            ax1.plot(iters, losses, "o-", label=labels[r_cap], color=colors[r_cap], markersize=4)
        else:
            ax1.scatter([final_it], [final_vl], marker="*", s=100, color=colors[r_cap], label=f"{labels[r_cap]} (final only)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss Curves (r_cap Ablation, seed=42)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "rcap_ablation_val_loss.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'rcap_ablation_val_loss.png'}")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for r_cap in [20, 30, 40]:
        diag = results[r_cap]["diagnostics"]
        iters = [int(r["iter"]) for r in diag]
        h_res_grad = [r["h_res_grad_median"] for r in diag]
        ax2.plot(iters, h_res_grad, label=labels[r_cap], color=colors[r_cap], alpha=0.8, linewidth=0.8)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("H_res Gradient Median")
    ax2.set_title("H_res Gradient Norm Time Series (r_cap Ablation, seed=42)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "rcap_ablation_grad_norm.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'rcap_ablation_grad_norm.png'}")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for r_cap in [20, 30, 40]:
        diag = results[r_cap]["diagnostics"]
        iters = [int(r["iter"]) for r in diag]
        sk_range = [r["sinkhorn_range_mean"] for r in diag]
        ax3.plot(iters, sk_range, label=labels[r_cap], color=colors[r_cap], alpha=0.8, linewidth=0.8)
    ax3.axhline(y=20, color=colors[20], linestyle="--", alpha=0.4, label="r_cap=20 threshold")
    ax3.axhline(y=30, color=colors[30], linestyle="--", alpha=0.4, label="r_cap=30 threshold")
    ax3.axhline(y=40, color=colors[40], linestyle="--", alpha=0.4, label="r_cap=40 threshold")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Sinkhorn Input Log-Range (mean)")
    ax3.set_title("Sinkhorn Input Log-Range Over Training (r_cap Ablation, seed=42)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "rcap_ablation_log_range.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'rcap_ablation_log_range.png'}")
    plt.close(fig3)


def main():
    print("Collecting r_cap ablation results...")
    results = collect_metrics()

    print("\n=== r_cap Ablation Comparison Table ===")
    header = f"{'r_cap':>6} | {'Val Loss':>10} | {'H_res Grad Med':>15} | {'||dH_res||_F':>13} | {'DS Error':>12} | {'Entropy':>10} | {'% Capped':>10}"
    print(header)
    print("-" * len(header))
    for r_cap in [20, 30, 40]:
        m = results[r_cap]
        ds_err = max(m["ds_row_error_mean"], m["ds_col_error_mean"])
        print(f"{r_cap:>6} | {m['best_val_loss']:>10.4f} | {m['h_res_grad_median']:>15.4e} | {m['h_res_param_drift_fro']:>13.4e} | {ds_err:>12.4e} | {m['entropy_mean']:>10.4f} | {m['pct_steps_capped']:>9.1f}%")

    out_json = {}
    for r_cap in [20, 30, 40]:
        m = results[r_cap].copy()
        del m["diagnostics"]
        out_json[str(r_cap)] = m
    json_path = BASE / "results" / "rcap_ablation_comparison.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    print("\nGenerating figures...")
    make_figures(results)
    print("Done.")

    return out_json


if __name__ == "__main__":
    main()
