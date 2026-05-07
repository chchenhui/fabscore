#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path("/home/chenhui/fabscore/agent4sci_rej/submission_345")
SUPP_ROOT = REPO_ROOT / "345_Multi_Agent_AI_System_for__Supplementary Material"
WORKSPACE = REPO_ROOT / "fabscore_codex" / "workspace"

sys.path.insert(0, str(SUPP_ROOT / "src"))
sys.path.insert(0, str(SUPP_ROOT / "validation"))

from models.analogs import AnalogForecaster  # noqa: E402
from phase5_real_validation import RealHistoricalValidator  # noqa: E402


def main() -> None:
    os.chdir(SUPP_ROOT)
    validator = RealHistoricalValidator()
    launches = validator.launches.copy()
    repatha_rows = launches[launches["drug_name"] == "Repatha"].copy()

    forecaster = AnalogForecaster(data_dir=SUPP_ROOT / "data_proc")

    records = []
    for _, drug in repatha_rows.iterrows():
        launch_id = drug["launch_id"]
        actual = validator.get_actual_revenues(launch_id)
        forecast = forecaster.forecast_from_analogs(drug, years=5)
        metrics = validator.calculate_accuracy_metrics(forecast, actual)
        actual_peak = float(np.max(actual))
        forecast_peak = float(np.max(forecast))
        records.append(
            {
                "launch_id": launch_id,
                "drug_name": drug["drug_name"],
                "therapeutic_area": drug.get("therapeutic_area", ""),
                "actual_peak": actual_peak,
                "forecast_peak": forecast_peak,
                "forecast_pct_of_actual": (forecast_peak / actual_peak) if actual_peak else None,
                "peak_ape": float(metrics["peak_ape"]),
                "mape": float(metrics["mape"]),
                "actual_revenues": [float(x) for x in actual.tolist()],
                "forecast_revenues": [float(x) for x in forecast.tolist()],
            }
        )

    summary = {
        "claim": "Repatha enhanced analog: $3.6B peak (238% of actual, 138.3% peak APE)",
        "repatha_row_count": int(len(records)),
        "unique_launch_ids": int(pd.Series([r["launch_id"] for r in records]).nunique()),
        "forecast_peak_min": min(r["forecast_peak"] for r in records) if records else None,
        "forecast_peak_max": max(r["forecast_peak"] for r in records) if records else None,
        "forecast_pct_min": min(r["forecast_pct_of_actual"] for r in records) if records else None,
        "forecast_pct_max": max(r["forecast_pct_of_actual"] for r in records) if records else None,
        "peak_ape_min": min(r["peak_ape"] for r in records) if records else None,
        "peak_ape_max": max(r["peak_ape"] for r in records) if records else None,
    }

    out = {"summary": summary, "records": records}
    out_path = WORKSPACE / "claim_15_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
