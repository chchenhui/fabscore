import os
import pandas as pd
from codecarbon import EmissionsTracker

def run_with_emissions(output_dir, project_name, workload_fn):
    """
    Run a workload with CodeCarbon and save both CSV and JSON logs.
    """
    os.makedirs(output_dir, exist_ok=True)

    tracker = EmissionsTracker(
        output_dir=output_dir,
        project_name=project_name,
    )
    tracker.start()
    workload_fn()
    tracker.stop()

    csv_path = os.path.join(output_dir, "emissions.csv")
    json_path = os.path.join(output_dir, "emissions.json")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.to_json(json_path, orient="records", indent=2)
        print(f"✅ Emissions logs saved: {csv_path}, {json_path}")
    else:
        print("⚠️ No emissions.csv found in", output_dir)
