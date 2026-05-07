"""Aggregate results from mHC training runs across multiple seeds.
Supports both Sinkhorn and orthostochastic configs (with optional orth_residual)."""
import json
import os
import sys
import numpy as np

LOGS_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_seed_results(prefix, seeds):
    results = []
    for seed in seeds:
        run_dir = os.path.join(LOGS_BASE, f"{prefix}_seed{seed}")
        summary_path = os.path.join(run_dir, "summary.json")
        diag_summary_path = os.path.join(run_dir, "diagnostics", "diagnostics_summary.json")
        ds_error_path = os.path.join(run_dir, "diagnostics", "ds_error.json")
        grad_spikes_path = os.path.join(run_dir, "diagnostics", "gradient_spikes.json")

        with open(summary_path) as f:
            summary = json.load(f)
        with open(diag_summary_path) as f:
            diag = json.load(f)
        with open(ds_error_path) as f:
            ds_records = json.load(f)
        with open(grad_spikes_path) as f:
            grad_data = json.load(f)

        ds_last_200 = [r for r in ds_records if r["iter"] >= summary["max_iters"] - 200]
        if ds_last_200:
            avg_max_row = np.mean([r["max_row_err"] for r in ds_last_200])
            avg_max_col = np.mean([r["max_col_err"] for r in ds_last_200])
            avg_ds_err = max(avg_max_row, avg_max_col)
        else:
            avg_ds_err = 0.0

        orth_residual_avg = None
        orth_residual_path = os.path.join(run_dir, "diagnostics", "orth_residual.json")
        if os.path.exists(orth_residual_path):
            with open(orth_residual_path) as f:
                orth_records = json.load(f)
            orth_last_200 = [r for r in orth_records if r["iter"] >= summary["max_iters"] - 200]
            if orth_last_200:
                all_layer_means = []
                for rec in orth_last_200:
                    layer_residuals = [l["orth_residual"] for l in rec["per_layer"]]
                    all_layer_means.append(np.mean(layer_residuals))
                orth_residual_avg = float(np.mean(all_layer_means))

        results.append({
            "seed": seed,
            "ok": summary["ok"],
            "final_val_loss": summary["last_eval"]["val"],
            "best_val_loss": summary["best_val_loss"],
            "last_eval_iter": summary["last_eval_iter"],
            "r_max": diag["r_max"],
            "ds_error_avg_last200": avg_ds_err,
            "orth_residual_avg_last200": orth_residual_avg,
            "elapsed_s": summary["elapsed_s"],
        })
    return results


def aggregate(results):
    val_losses = [r["final_val_loss"] for r in results]
    best_val_losses = [r["best_val_loss"] for r in results]
    r_maxes = [r["r_max"] for r in results if r["r_max"] is not None]
    ds_errors = [r["ds_error_avg_last200"] for r in results]

    orth_residuals = [r["orth_residual_avg_last200"] for r in results
                      if r.get("orth_residual_avg_last200") is not None]

    agg = {
        "n_seeds": len(results),
        "seeds": [r["seed"] for r in results],
        "final_val_loss": {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0,
            "values": val_losses,
        },
        "best_val_loss": {
            "mean": float(np.mean(best_val_losses)),
            "std": float(np.std(best_val_losses, ddof=1)) if len(best_val_losses) > 1 else 0.0,
            "values": best_val_losses,
        },
        "r_max": {
            "mean": float(np.mean(r_maxes)) if r_maxes else None,
            "std": float(np.std(r_maxes, ddof=1)) if len(r_maxes) > 1 else 0.0,
            "values": r_maxes,
        },
        "ds_error_avg_last200": {
            "mean": float(np.mean(ds_errors)),
            "std": float(np.std(ds_errors, ddof=1)) if len(ds_errors) > 1 else 0.0,
            "values": ds_errors,
        },
        "per_seed": results,
    }

    if orth_residuals:
        agg["orth_residual_avg_last200"] = {
            "mean": float(np.mean(orth_residuals)),
            "std": float(np.std(orth_residuals, ddof=1)) if len(orth_residuals) > 1 else 0.0,
            "values": orth_residuals,
        }

    return agg


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "setting_a_sinkhorn"
    seeds_str = sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5"
    seeds = [int(s) for s in seeds_str.split(",")]

    results = load_seed_results(prefix, seeds)
    agg = aggregate(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"Aggregated {len(results)} seeds -> {out_path}")
    print(f"  Final val loss: {agg['final_val_loss']['mean']:.6f} +/- {agg['final_val_loss']['std']:.6f}")
    print(f"  Best val loss:  {agg['best_val_loss']['mean']:.6f} +/- {agg['best_val_loss']['std']:.6f}")
    print(f"  r_max:          {agg['r_max']['mean']:.6f} +/- {agg['r_max']['std']:.6f}")
    print(f"  DS error (avg last 200): {agg['ds_error_avg_last200']['mean']:.6f} +/- {agg['ds_error_avg_last200']['std']:.6f}")
    if "orth_residual_avg_last200" in agg:
        print(f"  Orth residual (avg last 200): {agg['orth_residual_avg_last200']['mean']:.6f} +/- {agg['orth_residual_avg_last200']['std']:.6f}")
