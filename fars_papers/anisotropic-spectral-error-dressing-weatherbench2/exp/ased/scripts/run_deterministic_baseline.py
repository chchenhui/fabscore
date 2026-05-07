# Evaluate bias-corrected deterministic GraphCast Z500 forecast at 5-day lead time.
# Calibration: 2020 odd months. Evaluation: 2020 even months.
# For M=1 deterministic forecast, CRPS = latitude-weighted MAE.
# Outputs results to ased/results/deterministic_baseline.json.

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ased.data.wb2_loader import (
    load_graphcast_z500,
    load_era5_z500,
    split_calibration_evaluation,
    align_forecast_truth,
    compute_bias_correction,
    apply_bias_correction,
)
from ased.evaluation.metrics import evaluate_deterministic

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
LEAD_TIME_DAYS = 5


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading GraphCast Z500 forecasts (5-day lead, 2020)...")
    t0 = time.time()
    fc = load_graphcast_z500(year=2020, lead_time_days=LEAD_TIME_DAYS)
    print(f"  Loaded in {time.time() - t0:.1f}s, shape: {dict(fc.dims)}")

    print("Loading ERA5 Z500 truth (2020)...")
    t0 = time.time()
    truth = load_era5_z500(year=2020)
    print(f"  Loaded in {time.time() - t0:.1f}s, shape: {dict(truth.dims)}")

    print("Splitting calibration / evaluation...")
    fc_calib, fc_eval = split_calibration_evaluation(fc)
    truth_full = truth

    print("Aligning forecast and truth (calibration)...")
    fc_calib_a, truth_calib_a = align_forecast_truth(
        fc_calib, truth_full, lead_time_days=LEAD_TIME_DAYS
    )
    print(f"  Calibration pairs: {len(fc_calib_a.time)}")

    print("Computing bias correction on calibration split...")
    bias = compute_bias_correction(fc_calib_a, truth_calib_a)
    print(f"  Bias field: mean={float(bias.mean()):.2f}, std={float(bias.std()):.2f}")

    print("Aligning forecast and truth (evaluation)...")
    fc_eval_a, truth_eval_a = align_forecast_truth(
        fc_eval, truth_full, lead_time_days=LEAD_TIME_DAYS
    )
    print(f"  Evaluation pairs: {len(fc_eval_a.time)}")

    print("Applying bias correction to evaluation forecasts...")
    fc_eval_bc = apply_bias_correction(fc_eval_a, bias)

    print("Computing CRPS (= lat-weighted MAE for M=1)...")
    results = evaluate_deterministic(fc_eval_bc, truth_eval_a)
    print(f"  CRPS global:         {results['crps_global']:.4f}")
    print(f"  CRPS extratropics:   {results['crps_extratropics_30_60']:.4f}")

    print("Sanity check: raw forecast (no bias correction) CRPS...")
    results_raw = evaluate_deterministic(fc_eval_a, truth_eval_a)
    print(f"  CRPS global (raw):         {results_raw['crps_global']:.4f}")
    print(f"  CRPS extratropics (raw):   {results_raw['crps_extratropics_30_60']:.4f}")

    output = {
        "method": "deterministic_bias_corrected",
        "variable": "geopotential_500hPa",
        "lead_time_days": LEAD_TIME_DAYS,
        "calibration_months": [1, 3, 5, 7, 9, 11],
        "evaluation_months": [2, 4, 6, 8, 10, 12],
        "grid": "0.25deg_721x1440",
        "bias_corrected": True,
        "ensemble_size": 1,
        "metrics": {
            "crps_global": results["crps_global"],
            "crps_extratropics_30_60": results["crps_extratropics_30_60"],
            "note": "For M=1, CRPS equals latitude-weighted MAE",
        },
        "sanity_check_raw": {
            "crps_global": results_raw["crps_global"],
            "crps_extratropics_30_60": results_raw["crps_extratropics_30_60"],
        },
        "bias_field_stats": {
            "mean": float(bias.mean()),
            "std": float(bias.std()),
            "min": float(bias.min()),
            "max": float(bias.max()),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "deterministic_baseline.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
