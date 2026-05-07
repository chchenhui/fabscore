# CRPS and spread-skill ratio computation wrapping WeatherBench 2 evaluation utilities.
# For M=1 deterministic forecast, CRPS = latitude-weighted MAE (spread term is 0).
# Provides global and regional (extra-tropics 30-60 deg) evaluation.

import numpy as np
import xarray as xr
from weatherbench2 import metrics as wb2_metrics
from weatherbench2.regions import SliceRegion


ENSEMBLE_DIM = "realization"


def _ensure_ensemble_dim(ds):
    if ENSEMBLE_DIM not in ds.dims:
        return ds.expand_dims(ENSEMBLE_DIM)
    return ds


def extratropics_30_60_region():
    return SliceRegion(
        lat_slice=[slice(-60, -30), slice(30, 60)]
    )


def compute_crps(forecast, truth, region=None):
    forecast = _ensure_ensemble_dim(forecast)
    crps_metric = wb2_metrics.CRPS(ensemble_dim=ENSEMBLE_DIM)
    result = crps_metric.compute_chunk(forecast, truth, region=region)
    return float(result["geopotential"].mean("time").values)


def compute_crps_skill(forecast, truth, region=None):
    forecast = _ensure_ensemble_dim(forecast)
    skill_metric = wb2_metrics.CRPSSkill(ensemble_dim=ENSEMBLE_DIM)
    result = skill_metric.compute_chunk(forecast, truth, region=region)
    return float(result["geopotential"].mean("time").values)


def compute_crps_spread(forecast, truth, region=None):
    forecast = _ensure_ensemble_dim(forecast)
    spread_metric = wb2_metrics.CRPSSpread(ensemble_dim=ENSEMBLE_DIM)
    result = spread_metric.compute_chunk(forecast, truth, region=region)
    return float(result["geopotential"].mean("time").values)


def compute_spread_skill_ratio(forecast, truth, region=None):
    spread = compute_crps_spread(forecast, truth, region=region)
    skill = compute_crps_skill(forecast, truth, region=region)
    if skill == 0:
        return float("nan")
    return spread / skill


def evaluate_deterministic(forecast, truth):
    region_extratropics = extratropics_30_60_region()
    crps_global = compute_crps(forecast, truth, region=None)
    crps_extratropics = compute_crps(forecast, truth, region=region_extratropics)
    return {
        "crps_global": crps_global,
        "crps_extratropics_30_60": crps_extratropics,
    }


def evaluate_ensemble(forecast, truth):
    region_extratropics = extratropics_30_60_region()
    crps_global = compute_crps(forecast, truth, region=None)
    crps_extratropics = compute_crps(forecast, truth, region=region_extratropics)
    ssr_global = compute_spread_skill_ratio(forecast, truth, region=None)
    ssr_extratropics = compute_spread_skill_ratio(
        forecast, truth, region=region_extratropics
    )
    return {
        "crps_global": crps_global,
        "crps_extratropics_30_60": crps_extratropics,
        "spread_skill_ratio_global": ssr_global,
        "spread_skill_ratio_extratropics_30_60": ssr_extratropics,
    }


def evaluate_ensemble_chunked(forecast_bc_vals, truth_vals, perturbations,
                               coords, chunk_size=10):
    import gc
    region_extratropics = extratropics_30_60_region()
    n_times = forecast_bc_vals.shape[0]
    n_members = perturbations.shape[0]
    nlat = forecast_bc_vals.shape[1]
    nlon = forecast_bc_vals.shape[2]
    skill_metric = wb2_metrics.CRPSSkill(ensemble_dim=ENSEMBLE_DIM)
    spread_metric = wb2_metrics.CRPSSpread(ensemble_dim=ENSEMBLE_DIM)
    regions = {"global": None, "extratropics": region_extratropics}
    accum = {f"{m}_{r}": 0.0 for m in ["skill", "spread"] for r in regions}
    total_time = 0
    n_chunks = (n_times + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, n_times, chunk_size)):
        end = min(start + chunk_size, n_times)
        cs = end - start
        ens_arr = np.empty((n_members, cs, nlat, nlon), dtype=forecast_bc_vals.dtype)
        fc_chunk = forecast_bc_vals[start:end]
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
        for rname, region in regions.items():
            skill_res = skill_metric.compute_chunk(ens_ds, truth_ds, region=region)
            spread_res = spread_metric.compute_chunk(ens_ds, truth_ds, region=region)
            accum[f"skill_{rname}"] += float(skill_res["geopotential"].mean("time").values) * cs
            accum[f"spread_{rname}"] += float(spread_res["geopotential"].mean("time").values) * cs
            del skill_res, spread_res
        total_time += cs
        del ens_arr, ens_ds, truth_ds, fc_chunk
        gc.collect()
        if (ci + 1) % 10 == 0 or ci + 1 == n_chunks:
            print(f"    Chunk {ci+1}/{n_chunks} done")
    for k in accum:
        accum[k] /= total_time
    crps_global = accum["skill_global"] - 0.5 * accum["spread_global"]
    crps_extra = accum["skill_extratropics"] - 0.5 * accum["spread_extratropics"]
    ssr_global = accum["spread_global"] / accum["skill_global"] if accum["skill_global"] != 0 else float("nan")
    ssr_extra = accum["spread_extratropics"] / accum["skill_extratropics"] if accum["skill_extratropics"] != 0 else float("nan")
    return {
        "crps_global": crps_global,
        "crps_extratropics_30_60": crps_extra,
        "spread_skill_ratio_global": ssr_global,
        "spread_skill_ratio_extratropics_30_60": ssr_extra,
    }
