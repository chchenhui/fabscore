#!/usr/bin/env python3
"""
phase3_worker.py (patched)
- Fixes Task 3 unpacking
- Fixes Task 10 CSV generation (valid input, Phase 2 consistent)
- Ensures runtime errors are logged safely
- Patched Task 10 to expect string values for 'age' to align with Phase 2.
"""

import argparse
import importlib.util
import json
import os
import time
import math
import hashlib
import tempfile
import csv
from codecarbon import EmissionsTracker
import numpy as np
import sys


# ----------------------------
# Helpers
# ----------------------------

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_input_and_ref(task, scale, seed):
    """Return tuple (input_obj, expected_output) for given task and deterministic seed."""
    rng = np.random.default_rng(int(seed))
    n = int(scale)

    if task in ("array_sum", "01_array_sum", "001_array_sum"):
        arr = rng.integers(-1000, 1000, size=n).tolist()
        expected = sum(arr)
        return arr, expected

    if task in ("prefix_sum", "02_prefix_sum"):
        arr = rng.integers(-1000, 1000, size=n).tolist()
        expected = list(np.cumsum(arr))
        return arr, expected

    if task in ("matrix_multiplication", "03_matrix_multiplication"):
        m = int(max(1, math.isqrt(max(1, n))))
        A = rng.integers(0, 10, size=(m, m))
        B = rng.integers(0, 10, size=(m, m))
        expected = (A @ B).tolist()
        return (A.tolist(), B.tolist()), expected

    if task in ("top_k_selection", "04_top_k_selection"):
        arr = rng.integers(0, 1_000_000, size=n).tolist()
        k = max(1, min(5000, n // max(1, n//1000)))
        expected = sorted(__import__("heapq").nlargest(k, arr), reverse=True)
        return (arr, k), expected

    if task in ("stable_sorting", "05_stable_sorting"):
        vals = [(int(i % 1000), f"val{i}") for i in range(n)]
        rng.shuffle(vals)
        expected = sorted(vals, key=lambda x: x[0])
        return vals, expected

    if task in ("histogram", "06_histogram"):
        keys = [str(x) for x in rng.integers(0, 20, size=n)]
        import collections
        expected = dict(collections.Counter(keys))
        return keys, expected

    if task in ("string_deduplication", "07_string_deduplication"):
        arr = [f"s{int(i%100)}" for i in range(n)]
        rng2 = np.random.default_rng(int(seed)+1)
        arr = list(arr)
        rng2.shuffle(arr)
        seen, out = set(), []
        for s in arr:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return arr, out

    if task in ("json_parsing_filtering", "08_json_parsing_filtering"):
        objs = [{"active": bool(int(x%2)), "x": int(x)} for x in range(n)]
        rng2 = np.random.default_rng(int(seed)+2)
        rng2.shuffle(objs)
        expected = [o for o in objs if o.get("active", False)]
        return json.dumps(objs), expected

    if task in ("bfs", "09_bfs"):
        graph = {str(i): [str(i+1)] for i in range(max(0, n-1))}
        if n > 0: graph[str(n-1)] = []
        start = "0" if n > 0 else None
        expected = [str(i) for i in range(n)]
        return (graph, start), expected

    if task in ("csv_etl", "10_csv_etl"):
        # ✅ Always valid CSV (aligns with Phase 2)
        rows = [f"user{i},{20 + (i % 50)}" for i in range(n)]
        csv_str = "name,age\n" + "\n".join(rows) + ("\n" if n > 0 else "")
        # FIX: Cast 'age' to string to match Phase 2's expectation and standard CSV parser output.
        expected = [{"name": f"user{i}", "age": str(20 + (i % 50))} for i in range(n)]
        return csv_str, expected

    raise ValueError(f"Unknown task: {task}")


def load_candidate_func(candidate_path):
    spec = importlib.util.spec_from_file_location("candidate_module", candidate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "solve"):
        return module.solve
    for name in dir(module):
        if name.startswith("_"): continue
        obj = getattr(module, name)
        if callable(obj):
            return obj
    raise AttributeError("No callable found in candidate module")


def find_latest_csv_in_dir(d):
    if not os.path.isdir(d):
        return None
    csvs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv")]
    if not csvs:
        return None
    csvs.sort(key=lambda p: os.path.getmtime(p))
    return csvs[-1]


def parse_energy_and_co2_from_emissions_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None, None
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        if not reader:
            return None, None
        last = reader[-1]
        energy_keys = [k for k in last.keys() if "energy" in k.lower()]
        co2_keys = [k for k in last.keys() if "emissions" in k.lower() or "co2" in k.lower()]
        energy_kwh = None
        co2_kg = None
        for k in energy_keys:
            try:
                energy_kwh = float(last[k])
                break
            except Exception:
                continue
        for k in co2_keys:
            try:
                co2_kg = float(last[k])
                break
            except Exception:
                continue
        return energy_kwh, co2_kg


def run_candidate_and_measure(candidate_path, task, scale, seed, measure_interval, out_dir):
    source_hash = sha256_of_file(candidate_path)

    try:
        outputs = gen_input_and_ref(task, scale, seed)
        if len(outputs) == 2:
            inp, expected = outputs
        elif len(outputs) == 3:
            # This logic was slightly flawed, fixed to unpack correctly
            inp, expected = (outputs[0], outputs[1]), outputs[2]
        else:
            raise ValueError(f"Unexpected output length {len(outputs)} from input_gen")
    except Exception as e:
        return {"runtime_s": None, "energy_kwh": None, "co2_kg": None,
                "correct": False, "exception": f"input_gen_error: {type(e).__name__}: {e}"}

    run_out_dir = os.path.join(out_dir, f"run_{int(time.time()*1000)}_{os.getpid()}")
    os.makedirs(run_out_dir, exist_ok=True)

    try:
        tracker = EmissionsTracker(project_name="phase3",
                                   output_dir=run_out_dir,
                                   measure_power_secs=float(measure_interval))
    except Exception:
        tracker = EmissionsTracker(project_name="phase3", output_dir=run_out_dir)

    try:
        solution_func = load_candidate_func(candidate_path)
    except Exception as e:
        return {"runtime_s": None, "energy_kwh": None, "co2_kg": None,
                "correct": False, "exception": f"import_error: {type(e).__name__}: {e}"}

    result, exception = None, None
    try:
        tracker.start()
        t0 = time.perf_counter()
        try:
            if isinstance(inp, tuple) and not isinstance(inp, (str, bytes)):
                result = solution_func(*inp)
            else:
                result = solution_func(inp)
        except Exception as e:
            exception = f"runtime_exception: {type(e).__name__}: {e}"
            result = None
        runtime_s = time.perf_counter() - t0
        emissions_kg = tracker.stop()
    except Exception as e:
        try: tracker.stop()
        except Exception: pass
        return {"runtime_s": None, "energy_kwh": None, "co2_kg": None,
                "correct": False, "exception": f"measurement_error: {type(e).__name__}: {e}"}

    csv_path = find_latest_csv_in_dir(run_out_dir)
    energy_kwh, co2_kg_fromcsv = parse_energy_and_co2_from_emissions_csv(csv_path)
    co2_kg = co2_kg_fromcsv if co2_kg_fromcsv is not None else emissions_kg
    if energy_kwh is None:
        energy_kwh = None

    correct, correctness_exc = False, None
    try:
        def normalize(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x
        got = normalize(result)
        exp = normalize(expected)
        correct = (got == exp)
    except Exception as e:
        correctness_exc = f"compare_error: {type(e).__name__}: {e}"
        correct = False

    return {
        "runtime_s": runtime_s,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "correct": correct,
        "exception": exception or correctness_exc,
        "candidate_hash": source_hash
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scale", required=True, type=int)
    parser.add_argument("--run_idx", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--measure_interval", type=float, default=0.1)
    args = parser.parse_args()

    outdir = os.path.dirname(args.out_json)
    os.makedirs(outdir, exist_ok=True)
    res = run_candidate_and_measure(
        candidate_path=args.candidate,
        task=args.task,
        scale=args.scale,
        seed=args.seed,
        measure_interval=args.measure_interval,
        out_dir=outdir
    )
    res.update({
        "task": args.task,
        "scale": args.scale,
        "candidate_path": args.candidate,
        "run_idx": args.run_idx
    })
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()