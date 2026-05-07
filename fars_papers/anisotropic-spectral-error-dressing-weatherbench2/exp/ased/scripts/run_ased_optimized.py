# Optimized ASED evaluation on WB2 Z500 @ 5-day.
# Key changes from run_ased.py:
#   1. 3 mu-bins (uniform: [0,0.33], [0.33,0.67], [0.67,1.0]) instead of 2
#   2. 4 degree-bands for scale-dependent anisotropy weights

import gc
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import wandb

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
from ased.perturbations.ased import ASEDSampler

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LEAD_TIME_DAYS = 5
N_MEMBERS = 50
SEEDS = [0, 1, 2, 3, 4]
LMAX = 359
N_BINS = 10
MU_SPLIT = 0.5
L_MIN = 10
N_ANISO_BANDS = 4
N_MU_BINS = 3
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


def run_sanity_checks(ased_sampler, sed_sampler, C_l):
    print("\n=== Sanity Checks ===")
    lmax = ased_sampler.lmax
    l_min = ased_sampler.l_min
    all_pass = True

    print("\nCheck 1: SED degeneracy (w_low == w_high => g_{lm} == 1)...")
    g_iso = ased_sampler.compute_per_lm_multiplier(w_low=1.0, w_high=1.0)
    max_dev = float(np.max(np.abs(g_iso - 1.0)))
    if max_dev == 0.0:
        print(f"  PASS: max |g_lm - 1| = {max_dev}")
    else:
        print(f"  FAIL: max |g_lm - 1| = {max_dev}")
        all_pass = False

    g_aniso = ased_sampler.compute_per_lm_multiplier()

    print("\nCheck 2: Per-degree variance conservation: (1/(2l+1)) * sum_m g_{lm} == 1...")
    max_dev_conservation = 0.0
    for l in range(l_min, lmax + 1):
        g_sum = g_aniso[0, l, 0]
        for m in range(1, l + 1):
            g_sum += g_aniso[0, l, m] + g_aniso[1, l, m]
        mean_g = g_sum / (2 * l + 1)
        dev = abs(mean_g - 1.0)
        if dev > max_dev_conservation:
            max_dev_conservation = dev
    if max_dev_conservation < 1e-12:
        print(f"  PASS: max deviation = {max_dev_conservation:.2e} (< 1e-12)")
    else:
        print(f"  FAIL: max deviation = {max_dev_conservation:.2e} (>= 1e-12)")
        all_pass = False

    print("\nCheck 3: Total expected SH coefficient variance match (ASED vs SED)...")
    total_var_sed = 0.0
    total_var_ased = 0.0
    for l in range(lmax + 1):
        cl = C_l[l]
        for m in range(l + 1):
            n = 1 if m == 0 else 2
            total_var_sed += cl * n
            total_var_ased += cl * g_aniso[0, l, m] * n
    rel_diff = abs(total_var_ased - total_var_sed) / total_var_sed if total_var_sed > 0 else float('inf')
    if rel_diff < 1e-10:
        print(f"  PASS: V_sed={total_var_sed:.4f}, V_ased={total_var_ased:.4f}, rel_diff={rel_diff:.2e} (< 1e-10)")
    else:
        print(f"  FAIL: V_sed={total_var_sed:.4f}, V_ased={total_var_ased:.4f}, rel_diff={rel_diff:.2e} (>= 1e-10)")
        all_pass = False

    if not all_pass:
        raise RuntimeError("Sanity checks FAILED — aborting experiment.")
    print("\nAll sanity checks PASSED.\n")
    return True


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    wandb_project = os.environ.get("WANDB_PROJECT", "anisotropic-spectral-error-dressing-weatherbench2")
    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        project=wandb_project,
        name="ased_optimized_3bins_4bands",
        config={
            "method": "ased_optimized",
            "variable": "geopotential_500hPa",
            "lead_time_days": LEAD_TIME_DAYS,
            "ensemble_size": N_MEMBERS,
            "lmax": LMAX,
            "n_spectrum_bins": N_BINS,
            "mu_split": MU_SPLIT,
            "l_min": L_MIN,
            "n_aniso_bands": N_ANISO_BANDS,
            "n_mu_bins": N_MU_BINS,
            "seeds": SEEDS,
            "eval_chunk_size": EVAL_CHUNK_SIZE,
        },
    )

    residuals_calib, fc_bc_vals, truth_vals, coords, V_res = load_and_prepare()

    print(f"\nMemory: fc_bc={fc_bc_vals.nbytes/1e9:.2f} GB, truth={truth_vals.nbytes/1e9:.2f} GB")
    print(f"        residuals_calib={residuals_calib.nbytes/1e9:.2f} GB")

    print(f"\nEstimating degree spectrum C_l from {residuals_calib.shape[0]} calibration residuals...")
    t0 = time.time()
    sed_sampler = SEDSampler(lmax=LMAX)
    C_l = sed_sampler.estimate_degree_spectrum(residuals_calib, n_bins=N_BINS)
    print(f"  Estimated C_l in {time.time() - t0:.1f}s")

    print(f"\nEstimating anisotropy profile (n_bands={N_ANISO_BANDS}, n_mu_bins={N_MU_BINS})...")
    t0 = time.time()
    ased_sampler = ASEDSampler(lmax=LMAX, mu_split=MU_SPLIT, l_min=L_MIN, n_aniso_bands=N_ANISO_BANDS, n_mu_bins=N_MU_BINS)
    w_low, w_high = ased_sampler.estimate_anisotropy_profile(residuals_calib)
    print(f"  Estimated anisotropy in {time.time() - t0:.1f}s")

    wandb.log({
        "w_low_global": w_low,
        "w_high_global": w_high,
        "w_ratio_global": w_low / w_high,
        "V_res": V_res,
    })
    for i, ((blo, bhi), bw) in enumerate(zip(ased_sampler.bands, ased_sampler.band_weights)):
        log_dict = {f"band{i}_l_range": f"{blo}-{bhi}"}
        for j, w in enumerate(bw):
            log_dict[f"band{i}_w{j}"] = w
        if bw[-1] > 0:
            log_dict[f"band{i}_ratio_0_over_last"] = bw[0] / bw[-1]
        wandb.log(log_dict)

    print("Computing per-(l,m) multipliers g_{lm} with band-dependent weights...")
    g_lm = ased_sampler.compute_per_lm_multiplier()
    print(f"  g_lm shape: {g_lm.shape}, range: [{g_lm.min():.4f}, {g_lm.max():.4f}]")

    run_sanity_checks(ased_sampler, sed_sampler, C_l)

    del residuals_calib
    gc.collect()

    all_results = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")

        print(f"  Sampling {N_MEMBERS} ASED perturbation fields...")
        t0 = time.time()
        perturbations = ased_sampler.sample_perturbations(C_l, n_members=N_MEMBERS, seed=seed)
        print(f"  Sampled in {time.time() - t0:.1f}s, shape={perturbations.shape}")

        print("  Applying variance matching...")
        perturbations_scaled, alpha = ased_sampler.apply_variance_matching(perturbations, V_res)
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

        wandb.log({
            "seed": seed,
            "alpha": alpha,
            "crps_global": results["crps_global"],
            "crps_extratropics_30_60": results["crps_extratropics_30_60"],
            "ssr_global": results["spread_skill_ratio_global"],
            "ssr_extratropics_30_60": results["spread_skill_ratio_extratropics_30_60"],
        })

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

    sed_crps_global = 143.80
    sed_crps_extra = 182.90
    imp_global = (sed_crps_global - summary["crps_global"]["mean"]) / sed_crps_global * 100
    imp_extra = (sed_crps_extra - summary["crps_extratropics_30_60"]["mean"]) / sed_crps_extra * 100
    print(f"\n  Improvement over SED:")
    print(f"    CRPS global:       {imp_global:.2f}% (target >= 1%)")
    print(f"    CRPS extratropics: {imp_extra:.2f}% (target >= 1%)")
    print(f"    Both >= 1%: {'YES' if imp_global >= 1.0 and imp_extra >= 1.0 else 'NO'}")

    wandb.log({
        "crps_global_mean": summary["crps_global"]["mean"],
        "crps_global_std": summary["crps_global"]["std"],
        "crps_extratropics_mean": summary["crps_extratropics_30_60"]["mean"],
        "crps_extratropics_std": summary["crps_extratropics_30_60"]["std"],
        "ssr_global_mean": summary["spread_skill_ratio_global"]["mean"],
        "ssr_global_std": summary["spread_skill_ratio_global"]["std"],
        "ssr_extratropics_mean": summary["spread_skill_ratio_extratropics_30_60"]["mean"],
        "ssr_extratropics_std": summary["spread_skill_ratio_extratropics_30_60"]["std"],
        "improvement_crps_global_pct": imp_global,
        "improvement_crps_extratropics_pct": imp_extra,
    })

    band_info = []
    for (blo, bhi), bw in zip(ased_sampler.bands, ased_sampler.band_weights):
        band_info.append({
            "l_range": [blo, bhi],
            "weights": bw,
            "ratio_first_over_last": bw[0] / bw[-1] if bw[-1] > 0 else float('inf'),
        })

    output = {
        "method": "anisotropic_spectral_error_dressing_optimized",
        "variable": "geopotential_500hPa",
        "lead_time_days": LEAD_TIME_DAYS,
        "calibration_months": [1, 3, 5, 7, 9, 11],
        "evaluation_months": [2, 4, 6, 8, 10, 12],
        "grid": "0.25deg_721x1440",
        "bias_corrected": True,
        "ensemble_size": N_MEMBERS,
        "lmax": LMAX,
        "n_spectrum_bins": N_BINS,
        "mu_split": MU_SPLIT,
        "l_min": L_MIN,
        "n_aniso_bands": N_ANISO_BANDS,
        "n_mu_bins": N_MU_BINS,
        "seeds": SEEDS,
        "variance_matching_note": "alpha = sqrt(V_res / V_eta), globally matched",
        "residual_variance": V_res,
        "anisotropy_global": {
            "w_low": w_low,
            "w_high": w_high,
            "w_ratio_low_over_high": w_low / w_high,
        },
        "anisotropy_bands": band_info,
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
        "improvement_over_sed": {
            "crps_global_pct": imp_global,
            "crps_extratropics_pct": imp_extra,
            "meets_threshold": imp_global >= 1.0 and imp_extra >= 1.0,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "ased_optimized.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
