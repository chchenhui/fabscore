#!/usr/bin/env python3
"""
phase3_runner.py

Orchestrates Phase 3 energy measurement.

Usage:
 python phase3_runner.py \
   --filtered_dir ./filtered_candidates-codellama:70b-instruct-v5 \
   --out_dir ./outputs/phase3 \
   --scales 10000 100000 1000000 \
   --runs 5 \
   --timeout 600
"""

import os
import argparse
import hashlib
import csv
import json
import subprocess
import time
from pathlib import Path
import sys

# ---------- Configuration defaults ----------
DEFAULT_MEASURE_INTERVAL = 0.1  # seconds
DEFAULT_RUNS = 5
DEFAULT_TIMEOUT = 600  # seconds per run (adjust for large scales)
PYTHON_EXEC = sys.executable if sys else "python3"
WORKER_SCRIPT = "phase3_worker_patched.py"  # assumed in same dir

# ---------- Helpers ----------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def make_candidate_id(task, idx):
    return f"{task}_cand{idx:03d}"

def run_worker_subprocess(candidate_path, task, scale, run_idx, seed, measure_interval, out_json_path, timeout_s):
    cmd = [
        PYTHON_EXEC, WORKER_SCRIPT,
        "--candidate", candidate_path,
        "--task", task,
        "--scale", str(scale),
        "--run_idx", str(run_idx),
        "--seed", str(seed),
        "--out_json", out_json_path,
        "--measure_interval", str(measure_interval)
    ]
    try:
        # run worker in separate process so CodeCarbon binds to it
        subprocess.run(cmd, check=True, timeout=timeout_s, capture_output=True, text=True)
        return True, None
    except subprocess.TimeoutExpired as te:
        return False, f"timeout: {te}"
    except subprocess.CalledProcessError as cpe:
        # Include stderr for better debugging
        return False, f"calledproc_error: {cpe.stderr}"
    except Exception as e:
        return False, f"subprocess_error: {type(e).__name__}: {e}"

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scales", nargs="+", type=int, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--measure_interval", type=float, default=DEFAULT_MEASURE_INTERVAL)
    args = parser.parse_args()

    filtered_dir = Path(args.filtered_dir)
    out_dir = Path(args.out_dir)
    run_dir = out_dir / "raw_runs"
    meta_dir = out_dir / "metadata"
    results_dir = out_dir / "results"
    summary_csv = out_dir / "phase3_aggregated.csv"

    for d in (run_dir, meta_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Allow rerunning specific tasks, or all if not specified
    tasks_to_rerun = ("03_matrix_multiplication")
    tasks = sorted([p for p in filtered_dir.iterdir() if p.is_dir() and p.name in tasks_to_rerun])
    if not tasks:
        tasks = sorted([p for p in filtered_dir.iterdir() if p.is_dir()])


    for task_path in tasks:
        task = task_path.name  # e.g., "01_array_sum"
        print(f"\n=== Phase3: task {task} ===")
        candidate_files = sorted([p for p in task_path.iterdir() if p.suffix == ".py"])
        metadata_rows = []
        meta_csv_path = meta_dir / f"{task}_metadata.csv"

        with open(meta_csv_path, "w", newline="", encoding="utf-8") as mf:
            mw = csv.writer(mf)
            mw.writerow(["task", "candidate_id", "source_hash", "filepath"])

            for idx, cand in enumerate(candidate_files):
                cand_path = str(cand)
                cand_hash = sha256_file(cand_path)
                cand_id = make_candidate_id(task, idx)
                metadata_rows.append((task, cand_id, cand_hash, cand_path))
                mw.writerow([task, cand_id, cand_hash, cand_path])

        task_results_csv = results_dir / f"{task}_results.csv"
        with open(task_results_csv, "w", newline="", encoding="utf-8") as rf:
            writer = csv.writer(rf)
            writer.writerow([
                "task", "candidate_id", "candidate_hash", "scale", "run_idx",
                "runtime_s", "energy_kwh", "co2_kg", "correct", "exception", "worker_ok"
            ])

        for idx, (task_, cand_id, cand_hash, cand_path) in enumerate(metadata_rows):
            for scale in args.scales:
                for run_idx in range(args.runs):
                    seed = 1000 * idx + scale + run_idx
                    out_json = run_dir / f"{task}_{cand_id}_scale{scale}_run{run_idx}.json"
                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Running {cand_id} scale={scale} run={run_idx} ...", end="", flush=True)
                    
                    ok, msg = run_worker_subprocess(
                        candidate_path=cand_path,
                        task=task.split("_", 1)[-1],
                        scale=scale,
                        run_idx=run_idx,
                        seed=seed,
                        measure_interval=args.measure_interval,
                        out_json_path=str(out_json),
                        timeout_s=args.timeout
                    )
                    
                    row_to_write = [task, cand_id, cand_hash, scale, run_idx, None, None, None, False, msg, ok]
                    if ok and out_json.exists():
                        try:
                            with open(out_json, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            row_to_write = [
                                task, cand_id, cand_hash, scale, run_idx,
                                data.get("runtime_s"), data.get("energy_kwh"), data.get("co2_kg"),
                                data.get("correct"), data.get("exception"), True
                            ]
                            print(" OK")
                        except Exception as e:
                            row_to_write[-2] = f"read_json_error: {e}"
                            print(f" READ-JSON-ERR: {e}")
                    else:
                         print(f" FAILED: {msg}")

                    with open(task_results_csv, "a", newline="", encoding="utf-8") as rf:
                        w = csv.writer(rf)
                        w.writerow(row_to_write)

    print("\nAggregating per-task results into summary CSV...")
    summary_rows = []
    for resfile in sorted(results_dir.iterdir()):
        rows_by_candidate_scale = {}
        with open(resfile, "r", newline="", encoding="utf-8") as rf:
            r = csv.DictReader(rf)
            for rec in r:
                key = (rec["task"], rec["candidate_id"], int(rec["scale"]))
                rows_by_candidate_scale.setdefault(key, []).append(rec)
        
        import statistics
        def median_or_none(seq):
            return statistics.median(seq) if seq else None
        def std_or_none(seq):
            return statistics.pstdev(seq) if len(seq) > 1 else 0.0

        for key, recs in rows_by_candidate_scale.items():
            runtimes = [float(x["runtime_s"]) for x in recs if x["runtime_s"] not in (None, "", "None")]
            energies = [float(x["energy_kwh"]) for x in recs if x["energy_kwh"] not in (None, "", "None")]
            co2s = [float(x["co2_kg"]) for x in recs if x["co2_kg"] not in (None, "", "None")]
            corrects = [x["correct"] in ("True", True, "true", "TRUE") for x in recs]
            
            task, cand_id, scale = key
            summary_rows.append({
                "task": task,
                "candidate_id": cand_id,
                "scale": scale,
                "runtime_median": median_or_none(runtimes),
                "runtime_std": std_or_none(runtimes),
                "energy_median_kwh": median_or_none(energies),
                "energy_std_kwh": std_or_none(energies),
                "co2_median_kg": median_or_none(co2s),
                "co2_std_kg": std_or_none(co2s),
                "correct_fraction": sum(corrects)/len(corrects) if corrects else 0.0
            })

    with open(summary_csv, "w", newline="", encoding="utf-8") as sf:
        fieldnames = [
            "task", "candidate_id", "scale", "runtime_median", "runtime_std",
            "energy_median_kwh", "energy_std_kwh", "co2_median_kg", "co2_std_kg",
            "correct_fraction"
        ]
        w = csv.DictWriter(sf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Phase 3 complete. Summary written to: {summary_csv}")

if __name__ == "__main__":
    main()