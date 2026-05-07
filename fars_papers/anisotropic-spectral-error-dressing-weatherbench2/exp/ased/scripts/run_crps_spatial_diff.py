# Compute gridpoint-level CRPS for SED and ASED (optimized 3-bin 4-band)
# on the evaluation split (seed=0, M=50). Saves spatial CRPS arrays for
# subsequent plotting of CRPS_SED - CRPS_ASED spatial difference map.
# Pointwise CRPS = mean_t[ E|X-y| - 0.5*E|X-X'| ] following WB2 definition.

import gc
import json
import os
import sys
import time

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ased.data.wb2_loader import (
    load_graphcast_z500,
    load_era5_z500,
    split_calibration_evaluation,
    align_forecast_truth,
    compute_bias_correction,
    apply_bias_correction,
)
from ased.perturbations.sed import SEDSampler
from ased.perturbations.ased import ASEDSampler

from weatherbench2.metrics import _pointwise_crps_skill, _pointwise_crps_spread

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LEAD_TIME_DAYS = 5
N_MEMBERS = 50
SEED = 0
LMAX = 359
N_BINS = 10
MU_SPLIT = 0.5
L_MIN = 10
N_ANISO_BANDS = 4
N_MU_BINS = 3
EVAL_CHUNK_SIZE = 5
ENSEMBLE_DIM = "realization"


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


def compute_spatial_crps(fc_bc_vals, truth_vals, perturbations, coords, chunk_size=5):
    """Compute CRPS at each gridpoint, averaged over time."""
    n_times = fc_bc_vals.shape[0]
    n_members = perturbations.shape[0]
    nlat = fc_bc_vals.shape[1]
    nlon = fc_bc_vals.shape[2]

    skill_accum = np.zeros((nlat, nlon), dtype=np.float64)
    spread_accum = np.zeros((nlat, nlon), dtype=np.float64)
    n_chunks = (n_times + chunk_size - 1) // chunk_size

    for ci, start in enumerate(range(0, n_times, chunk_size)):
        end = min(start + chunk_size, n_times)
        cs = end - start

        ens_arr = np.empty((n_members, cs, nlat, nlon), dtype=fc_bc_vals.dtype)
        fc_chunk = fc_bc_vals[start:end]
        for j in range(n_members):
            ens_arr[j] = fc_chunk + perturbations[j]

        ens_ds = xr.Dataset({
            "geopotential": xr.DataArray(
                ens_arr,
                dims=["realization", "time", "latitude", "longitude"],
                coords={
                    "realization": np.arange(n_members),
                    "time": coords["time"][start:end],
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                },
            )
        })
        truth_ds = xr.Dataset({
            "geopotential": xr.DataArray(
                truth_vals[start:end],
                dims=["time", "latitude", "longitude"],
                coords={
                    "time": coords["time"][start:end],
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                },
            )
        })

        pw_skill = _pointwise_crps_skill(ens_ds, truth_ds, ENSEMBLE_DIM, skipna=False)
        pw_spread = _pointwise_crps_spread(ens_ds, ENSEMBLE_DIM, skipna=False)

        skill_accum += pw_skill["geopotential"].sum("time").values
        spread_accum += pw_spread["geopotential"].sum("time").values

        del ens_arr, ens_ds, truth_ds, pw_skill, pw_spread, fc_chunk
        gc.collect()

        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            print(f"    Chunk {ci+1}/{n_chunks} done")

    skill_mean = skill_accum / n_times
    spread_mean = spread_accum / n_times
    crps_spatial = skill_mean - 0.5 * spread_mean
    return crps_spatial


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    residuals_calib, fc_bc_vals, truth_vals, coords, V_res = load_and_prepare()
    print(f"\nMemory: fc_bc={fc_bc_vals.nbytes/1e9:.2f} GB, truth={truth_vals.nbytes/1e9:.2f} GB")

    # --- SED ---
    print(f"\n=== SED: Estimating degree spectrum C_l ===")
    t0 = time.time()
    sed_sampler = SEDSampler(lmax=LMAX)
    C_l = sed_sampler.estimate_degree_spectrum(residuals_calib, n_bins=N_BINS)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n=== SED: Sampling M={N_MEMBERS} perturbations (seed={SEED}) ===")
    t0 = time.time()
    sed_perts = sed_sampler.sample_perturbations(n_members=N_MEMBERS, seed=SEED)
    sed_perts, alpha_sed = sed_sampler.apply_variance_matching(sed_perts, V_res)
    print(f"  Done in {time.time() - t0:.1f}s, alpha={alpha_sed:.4f}")

    print(f"\n=== SED: Computing spatial CRPS ===")
    t0 = time.time()
    crps_sed = compute_spatial_crps(fc_bc_vals, truth_vals, sed_perts, coords, EVAL_CHUNK_SIZE)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Global mean CRPS_SED = {np.mean(crps_sed):.4f}")

    del sed_perts
    gc.collect()

    # --- ASED ---
    print(f"\n=== ASED: Estimating anisotropy profile ===")
    t0 = time.time()
    ased_sampler = ASEDSampler(
        lmax=LMAX, mu_split=MU_SPLIT, l_min=L_MIN,
        n_aniso_bands=N_ANISO_BANDS, n_mu_bins=N_MU_BINS,
    )
    ased_sampler.estimate_anisotropy_profile(residuals_calib)
    ased_sampler.compute_per_lm_multiplier()
    print(f"  Done in {time.time() - t0:.1f}s")

    del residuals_calib
    gc.collect()

    print(f"\n=== ASED: Sampling M={N_MEMBERS} perturbations (seed={SEED}) ===")
    t0 = time.time()
    ased_perts = ased_sampler.sample_perturbations(C_l, n_members=N_MEMBERS, seed=SEED)
    ased_perts, alpha_ased = ased_sampler.apply_variance_matching(ased_perts, V_res)
    print(f"  Done in {time.time() - t0:.1f}s, alpha={alpha_ased:.4f}")

    print(f"\n=== ASED: Computing spatial CRPS ===")
    t0 = time.time()
    crps_ased = compute_spatial_crps(fc_bc_vals, truth_vals, ased_perts, coords, EVAL_CHUNK_SIZE)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Global mean CRPS_ASED = {np.mean(crps_ased):.4f}")

    del ased_perts
    gc.collect()

    # --- Save ---
    delta_crps = crps_sed - crps_ased
    print(f"\n=== Results ===")
    print(f"  Mean ΔCRPS (SED - ASED, positive=ASED better): {np.mean(delta_crps):.4f}")
    print(f"  Max improvement (ASED better): {np.max(delta_crps):.4f}")
    print(f"  Max degradation (SED better): {np.min(delta_crps):.4f}")

    lat = coords["latitude"]
    extra_mask = ((np.abs(lat) >= 30) & (np.abs(lat) <= 60))
    delta_extra = delta_crps[extra_mask, :]
    delta_tropics = delta_crps[(np.abs(lat) < 30), :]
    print(f"  Mean ΔCRPS extra-tropics 30-60: {np.mean(delta_extra):.4f}")
    print(f"  Mean ΔCRPS tropics <30: {np.mean(delta_tropics):.4f}")

    np.save(os.path.join(RESULTS_DIR, "crps_spatial_sed.npy"), crps_sed)
    np.save(os.path.join(RESULTS_DIR, "crps_spatial_ased.npy"), crps_ased)
    np.save(os.path.join(RESULTS_DIR, "crps_spatial_coords_lat.npy"), coords["latitude"])
    np.save(os.path.join(RESULTS_DIR, "crps_spatial_coords_lon.npy"), coords["longitude"])

    summary = {
        "method_sed": "spectral_error_dressing_degree_only",
        "method_ased": "anisotropic_sed_3bin_4band",
        "seed": SEED,
        "ensemble_size": N_MEMBERS,
        "global_mean_crps_sed": float(np.mean(crps_sed)),
        "global_mean_crps_ased": float(np.mean(crps_ased)),
        "global_mean_delta_crps": float(np.mean(delta_crps)),
        "extratropics_30_60_mean_delta_crps": float(np.mean(delta_extra)),
        "tropics_lt30_mean_delta_crps": float(np.mean(delta_tropics)),
        "alpha_sed": float(alpha_sed),
        "alpha_ased": float(alpha_ased),
    }
    with open(os.path.join(RESULTS_DIR, "crps_spatial_diff_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved spatial CRPS arrays and summary to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
