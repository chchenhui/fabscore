# Calibration protocol for controller thresholds.
# Loads nominal (no-perturbation) probe traces and computes innovation statistics.
# Calibrates epsilon (Or) and h (CUSUM) to match target rollback rate p_0.
# Usage: python -m cusum_controller.calibration.calibrate [--traces_dir ...] [--output_dir ...]

import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_innovations_from_probe_trace(probe_losses, alpha=0.1):
    y_hat = probe_losses[0]
    innovations = []
    for t in range(1, len(probe_losses)):
        nu = probe_losses[t] - y_hat
        innovations.append(nu)
        y_hat = (1.0 - alpha) * y_hat + alpha * probe_losses[t]
    return np.array(innovations, dtype=np.float64)


def calibrate_or_epsilon(traces_dir, output_dir, alpha=0.1, p0=0.002, num_seeds=20):
    os.makedirs(output_dir, exist_ok=True)

    all_innovations = []
    for seed in range(num_seeds):
        path = os.path.join(traces_dir, f"seed{seed}.npy")
        probe_losses = np.load(path)
        innovations = compute_innovations_from_probe_trace(probe_losses, alpha=alpha)
        all_innovations.append(innovations)

    pooled = np.concatenate(all_innovations)
    mu_0 = float(np.mean(pooled))
    sigma_0 = float(np.std(pooled, ddof=1))
    percentile = (1.0 - p0) * 100.0
    epsilon = float(np.percentile(pooled, percentile))

    empirical_rate = float(np.mean(pooled > epsilon))

    params = {
        "epsilon": epsilon,
        "mu_0": mu_0,
        "sigma_0": sigma_0,
        "alpha": alpha,
        "p0_target": p0,
        "percentile_used": percentile,
        "num_seeds": num_seeds,
        "total_innovations": len(pooled),
        "empirical_rollback_rate": empirical_rate,
    }

    out_path = os.path.join(output_dir, "calibration_params.json")
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Calibration results:")
    print(f"  epsilon = {epsilon:.6f}")
    print(f"  mu_0 = {mu_0:.6f}")
    print(f"  sigma_0 = {sigma_0:.6f}")
    print(f"  Target p_0 = {p0}")
    print(f"  Empirical rollback rate = {empirical_rate:.6f}")
    print(f"  Total innovations pooled = {len(pooled)}")
    print(f"  Saved to {out_path}")

    return params


def calibrate_cusum_h(traces_dir, output_dir, mu_0, sigma_0, alpha=0.1, k=0.5,
                      p0=0.002, candidates=None, num_seeds=20, reset_fraction=0.5):
    if candidates is None:
        candidates = [h * 0.5 for h in range(4, 50)]
    os.makedirs(output_dir, exist_ok=True)

    all_innovations = []
    for seed in range(num_seeds):
        path = os.path.join(traces_dir, f"seed{seed}.npy")
        probe_losses = np.load(path)
        innovations = compute_innovations_from_probe_trace(probe_losses, alpha=alpha)
        all_innovations.append(innovations)

    results_table = []
    for h in candidates:
        total_alarms = 0
        total_steps = 0
        for innovations in all_innovations:
            r_vals = (innovations - mu_0) / (sigma_0 + 1e-8)
            S = 0.0
            for r_t in r_vals:
                S = max(0.0, S + r_t - k)
                total_steps += 1
                if S > h:
                    total_alarms += 1
                    S = h * reset_fraction
        alarm_rate = total_alarms / total_steps if total_steps > 0 else 0.0
        results_table.append({"h": h, "alarm_rate": alarm_rate,
                              "total_alarms": total_alarms, "total_steps": total_steps})

    print(f"\nFIR-CUSUM h calibration (k={k}, reset_fraction={reset_fraction}, target p_0={p0}):")
    print(f"  {'h':>6s}  {'alarm_rate':>12s}  {'total_alarms':>14s}  {'total_steps':>12s}")
    for row in results_table:
        print(f"  {row['h']:6.1f}  {row['alarm_rate']:12.6f}  {row['total_alarms']:14d}  {row['total_steps']:12d}")

    best = None
    best_diff = float("inf")
    for row in results_table:
        diff = abs(row["alarm_rate"] - p0)
        if diff < best_diff or (diff == best_diff and row["h"] > best["h"]):
            best_diff = diff
            best = row

    selected_h = best["h"]
    selected_rate = best["alarm_rate"]
    print(f"\n  Selected h = {selected_h} (alarm rate = {selected_rate:.6f}, target = {p0})")

    out_path = os.path.join(output_dir, "calibration_params.json")
    with open(out_path, "r") as f:
        params = json.load(f)

    params["cusum_h"] = selected_h
    params["cusum_k"] = k
    params["cusum_reset_fraction"] = reset_fraction
    params["cusum_alarm_rate"] = selected_rate
    params["cusum_calibration_table"] = results_table

    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"  Updated {out_path}")
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces_dir",
        type=str,
        default=os.path.join(BASE_DIR, "cusum_controller", "results", "no_controller", "nominal_probe_traces"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(BASE_DIR, "cusum_controller", "results", "calibration"),
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--p0", type=float, default=0.002)
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--mode", type=str, default="all", choices=["or", "cusum", "all"])
    parser.add_argument("--reset_fraction", type=float, default=0.5)
    args = parser.parse_args()

    if args.mode in ("or", "all"):
        params = calibrate_or_epsilon(
            traces_dir=args.traces_dir,
            output_dir=args.output_dir,
            alpha=args.alpha,
            p0=args.p0,
            num_seeds=args.num_seeds,
        )
    else:
        out_path = os.path.join(args.output_dir, "calibration_params.json")
        with open(out_path) as f:
            params = json.load(f)

    if args.mode in ("cusum", "all"):
        calibrate_cusum_h(
            traces_dir=args.traces_dir,
            output_dir=args.output_dir,
            mu_0=params["mu_0"],
            sigma_0=params["sigma_0"],
            alpha=args.alpha,
            p0=args.p0,
            num_seeds=args.num_seeds,
            reset_fraction=args.reset_fraction,
        )


if __name__ == "__main__":
    main()
