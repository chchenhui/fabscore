# Evaluate degree-only SED perturbation baseline on WB2 Z500 @ 5-day.
# Estimates empirical degree power spectrum C_l from calibration residuals,
# samples M=50 isotropic SH perturbations per seed, variance-matches,
# and evaluates CRPS and spread-skill ratio on the evaluation split.
# Outputs mean+/-std across 5 seeds to ased/results/sed_baseline.json.

import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ased.data.wb2_loader import (
    load_graphcast_z500,
    load_era5_z500,
    split_calibration_evaluation,
    align_forecast_truth,
    compute_bias_correction,
    apply_bias_correction,
)
from ased.evaluation.metrics import evaluate_ensemble_chunked
from ased.perturbations.sed import SEDSampler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LEAD_TIME_DAYS = 5
N_MEMBERS = 50
SEEDS = [0, 1, 2, 3, 4]
LMAX = 359
N_BINS = 10
EVAL_CHUNK_SIZE = 5


def load_and_prepare():
    print("Loading GraphCast Z500 forecasts (5-day lead, 2020)...")
    t0 = time.time()
    fc = load_graphcast_z500(year=2020, lead_time_days=LEAD_TIME_DAYS)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("Loading ERA5 Z500 truth (2020)...")
    t0 = time.time()
    truth = load_era5_z500(year=2020)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("Splitting and aligning...")
    fc_calib, fc_eval = split_calibration_evaluation(fc)
    del fc
    gc.collect()

    fc_calib_a, truth_calib_a = align_forecast_truth(fc_calib, truth, lead_time_days=LEAD_TIME_DAYS)
    del fc_calib
    gc.collect()

    print("Computing bias correction on calibration split...")
    bias = compute_bias_correction(fc_calib_a, truth_calib_a)

    fc_calib_bc = apply_bias_correction(fc_calib_a, bias)
    residuals_calib = fc_calib_bc["geopotential"].values - truth_calib_a["geopotential"].values
    V_res = float(np.mean(residuals_calib ** 2))
    print(f"  V_res = {V_res:.2f}")

    del fc_calib_a, truth_calib_a, fc_calib_bc
    gc.collect()

    fc_eval_a, truth_eval_a = align_forecast_truth(fc_eval, truth, lead_time_days=LEAD_TIME_DAYS)
    del fc_eval, truth
    gc.collect()

    print("Applying bias correction to evaluation forecasts...")
    fc_eval_bc = apply_bias_correction(fc_eval_a, bias)
    del fc_eval_a, bias
    gc.collect()

    fc_bc_vals = fc_eval_bc["geopotential"].values.copy()
    truth_vals = truth_eval_a["geopotential"].values.copy()
    coords = {
        "time": fc_eval_bc.time.values.copy(),
        "latitude": fc_eval_bc.latitude.values.copy(),
        "longitude": fc_eval_bc.longitude.values.copy(),
    }
    n_eval = len(coords["time"])
    print(f"  Evaluation pairs: {n_eval}")

    del fc_eval_bc, truth_eval_a
    gc.collect()

    return residuals_calib, fc_bc_vals, truth_vals, coords, V_res


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    residuals_calib, fc_bc_vals, truth_vals, coords, V_res = load_and_prepare()

    print(f"\nMemory: fc_bc={fc_bc_vals.nbytes/1e9:.2f} GB, truth={truth_vals.nbytes/1e9:.2f} GB")
    print(f"        residuals_calib={residuals_calib.nbytes/1e9:.2f} GB")

    print(f"\nEstimating degree spectrum C_l from {residuals_calib.shape[0]} calibration residuals...")
    t0 = time.time()
    sampler = SEDSampler(lmax=LMAX)
    C_l = sampler.estimate_degree_spectrum(residuals_calib, n_bins=N_BINS)
    print(f"  Estimated C_l in {time.time() - t0:.1f}s")
    print(f"  C_l[0]={C_l[0]:.2f}, C_l[10]={C_l[10]:.2f}, C_l[100]={C_l[100]:.2f}, C_l[359]={C_l[359]:.2f}")

    del residuals_calib
    gc.collect()

    all_results = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")

        print(f"  Sampling {N_MEMBERS} perturbation fields...")
        t0 = time.time()
        perturbations = sampler.sample_perturbations(n_members=N_MEMBERS, seed=seed)
        print(f"  Sampled in {time.time() - t0:.1f}s, shape={perturbations.shape}")

        print("  Applying variance matching...")
        perturbations_scaled, alpha = sampler.apply_variance_matching(perturbations, V_res)
        del perturbations
        gc.collect()
        print(f"  alpha = {alpha:.4f}")

        print(f"  Evaluating (chunk_size={EVAL_CHUNK_SIZE})...")
        t0 = time.time()
        results = evaluate_ensemble_chunked(
            fc_bc_vals, truth_vals, perturbations_scaled,
            coords, chunk_size=EVAL_CHUNK_SIZE,
        )
        elapsed = time.time() - t0
        print(f"  Evaluated in {elapsed:.1f}s")
        print(f"  CRPS global:       {results['crps_global']:.4f}")
        print(f"  CRPS extratropics: {results['crps_extratropics_30_60']:.4f}")
        print(f"  SSR global:        {results['spread_skill_ratio_global']:.4f}")
        print(f"  SSR extratropics:  {results['spread_skill_ratio_extratropics_30_60']:.4f}")
        all_results.append(results)

        del perturbations_scaled
        gc.collect()

    print("\n=== Aggregating across seeds ===")
    metric_keys = [
        "crps_global",
        "crps_extratropics_30_60",
        "spread_skill_ratio_global",
        "spread_skill_ratio_extratropics_30_60",
    ]
    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in all_results]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals],
        }
        print(f"  {key}: {summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")

    output = {
        "method": "spectral_error_dressing_degree_only",
        "variable": "geopotential_500hPa",
        "lead_time_days": LEAD_TIME_DAYS,
        "calibration_months": [1, 3, 5, 7, 9, 11],
        "evaluation_months": [2, 4, 6, 8, 10, 12],
        "grid": "0.25deg_721x1440",
        "bias_corrected": True,
        "ensemble_size": N_MEMBERS,
        "lmax": LMAX,
        "n_spectrum_bins": N_BINS,
        "seeds": SEEDS,
        "variance_matching_note": "alpha = sqrt(V_res / V_eta), globally matched",
        "residual_variance": V_res,
        "C_l_sample": {
            "l0": float(C_l[0]),
            "l10": float(C_l[10]),
            "l50": float(C_l[50]),
            "l100": float(C_l[100]),
            "l200": float(C_l[200]),
            "l359": float(C_l[359]),
        },
        "metrics": summary,
        "per_seed_results": all_results,
    }

    out_path = os.path.join(RESULTS_DIR, "sed_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
