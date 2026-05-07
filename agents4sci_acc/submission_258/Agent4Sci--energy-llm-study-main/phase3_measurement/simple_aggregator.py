#!/usr/bin/env python3
"""
simple_aggregator.py

Aggregates all result CSVs from a single, curated directory.

This script is designed for a simple workflow where the user manually collects
all the desired result files into one folder. It then reads all of them and
produces a final summary.

Usage:
  1. Manually create a folder (e.g., "./final_results").
  2. Copy the specific *_results.csv files you want to analyze into it.
  3. Run the script:
     python simple_aggregator.py \
       --input_dir ./final_results \
       --output_csv ./final_summary.csv
"""

import argparse
import csv
import statistics
from pathlib import Path
from collections import defaultdict

def aggregate_from_single_folder(input_dir, output_csv):
    """
    Scans a single directory, aggregates all result files found within it,
    and writes a summary CSV.
    """
    input_path = Path(input_dir).resolve()
    print(f"--- Processing directory: {input_path} ---")

    if not input_path.is_dir():
        print(f"Error: The provided input directory does not exist: {input_path}")
        return

    # --- Step 1: Find all result files in the folder ---
    print(f"Searching for '*_results.csv' files...")
    result_files = list(input_path.glob("*_results.csv"))

    if not result_files:
        print(f"Error: No '*_results.csv' files were found in {input_path}.")
        return

    print(f"Found {len(result_files)} result file(s) to aggregate.")

    # --- Step 2: Read and group data from all found files ---
    rows_by_candidate_scale = defaultdict(list)
    for resfile in result_files:
        try:
            with open(resfile, "r", newline="", encoding="utf-8") as rf:
                reader = csv.DictReader(rf)
                for record in reader:
                    # Create a unique key for each combination of task, candidate, and scale
                    key = (record["task"], record["candidate_id"], int(record["scale"]))
                    rows_by_candidate_scale[key].append(record)
        except Exception as e:
            print(f"Warning: Could not process file {resfile.name}. Error: {e}")
            continue
    
    # --- Step 3: Perform the aggregation calculations ---
    def median_or_none(seq):
        return statistics.median(seq) if seq else None

    def std_or_none(seq):
        # Use stdev for sample standard deviation; requires at least 2 data points
        return statistics.stdev(seq) if len(seq) > 1 else 0.0

    summary_rows = []
    for key, recs in rows_by_candidate_scale.items():
        runtimes = [float(x["runtime_s"]) for x in recs if x.get("runtime_s") not in (None, "", "None")]
        energies = [float(x["energy_kwh"]) for x in recs if x.get("energy_kwh") not in (None, "", "None")]
        co2s = [float(x["co2_kg"]) for x in recs if x.get("co2_kg") not in (None, "", "None")]
        corrects = [x["correct"].strip().lower() == 'true' for x in recs if "correct" in x]
        
        task, cand_id, scale = key
        summary_rows.append({
            "task": task,
            "candidate_id": cand_id,
            "scale": scale,
            "num_runs": len(recs),
            "runtime_median": median_or_none(runtimes),
            "runtime_std": std_or_none(runtimes),
            "energy_median_kwh": median_or_none(energies),
            "energy_std_kwh": std_or_none(energies),
            "co2_median_kg": median_or_none(co2s),
            "co2_std_kg": std_or_none(co2s),
            "correct_fraction": sum(corrects) / len(corrects) if corrects else 0.0
        })
    
    summary_rows.sort(key=lambda r: (r['task'], r['candidate_id'], r['scale']))

    # --- Step 4: Write the final summary CSV ---
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task", "candidate_id", "scale", "num_runs",
        "runtime_median", "runtime_std",
        "energy_median_kwh", "energy_std_kwh",
        "co2_median_kg", "co2_std_kg",
        "correct_fraction"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as sf:
        writer = csv.DictWriter(sf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nAggregation complete. Final summary written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregates all *_results.csv files from a single input directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_dir", 
        required=True,
        help="The single directory containing all the curated result CSV files."
    )
    parser.add_argument(
        "--output_csv", 
        required=True,
        help="Path to write the final aggregated summary CSV file."
    )
    args = parser.parse_args()
    
    aggregate_from_single_folder(args.input_dir, args.output_csv)

if __name__ == "__main__":
    main()