#!/usr/bin/env python3 
# compute_runtimes.py
# Generate a LaTeX table for compute workers and wall-clock runtimes (Table 7),
# plus reproducible artifacts (JSON + logs), using American English throughout.
#
# Usage:
#   python compute_runtimes.py                          # fast: 120 episodes, 1 seed
#   python compute_runtimes.py --full                   # also run sensitivity/robustness
#   python compute_runtimes.py --episodes 5000 --seeds 5 --seed0 42 --full
#
# Requirements: train.py, llm_eval.py in repo root (sensitivity.py/robustness.py only if --full).
# Optional: psutil (for RAM), torch (for GPU name). Script works without them.

import os, sys, time, platform, csv, subprocess, argparse, json
from pathlib import Path

# -------------------- Hardware helpers --------------------
def _cpu_threads():
    try:
        import psutil
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except Exception:
        return os.cpu_count() or 1

def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total/1e9, 1)
    except Exception:
        # Linux fallback (kB)
        try:
            with open('/proc/meminfo','r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = float(line.split()[1])
                        return round(kb/1e6, 1)
        except Exception:
            return -1.0  # unknown

def _gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "None"
    except Exception:
        return "None"

def _cpu_name():
    cpu = platform.processor()
    if not cpu:
        try:
            cpu = platform.uname().processor
        except Exception:
            cpu = ""
    return cpu or platform.machine()

def _escape_latex(s: str) -> str:
    return (s.replace('\\', r'\textbackslash{}')
             .replace('_', r'\_')
             .replace('%', r'\%')
             .replace('&', r'\&'))

def worker_str():
    return f"CPU: {_cpu_name()} ({_cpu_threads()} threads), RAM: {_ram_gb()} GB, GPU: {_gpu_name()}"

# -------------------- Utilities --------------------
def fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0: return f"{h}h {m}m {int(s)}s"
    if m > 0: return f"{m}m {int(s)}s"
    return f"{int(s)}s"

def run_and_time(cmd, cwd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    return p.returncode, dt, p.stdout, p.stderr

def ensure_data_sources(root: Path) -> Path:
    """
    Create a minimal data_sources.csv from manuscript Table 1 if not present.
    """
    ds = root / "data_sources.csv"
    if ds.exists():
        return ds
    rows = []
    def add(case, kv):
        for k, v in kv.items():
            rows.append({"case": case, "key": k, "value": v})
    common = dict(
        building_gfa=10000, operation_years=20, discount_rate=0.03,
        gamma=0.95, energy_noise=0.10, occupants_per_10000m2=500
    )
    us = dict(
        baseline_eui=240, high_perf_eui=150, electricity_price=0.10,
        grid_carbon_intensity=0.42, embodied_carbon_baseline=500,
        embodied_carbon_reduction_green=0.15, construction_cost=2200,
        design_premium_green=0.02, design_premium_ultra=0.05, scc=190,
        productivity_gain_hq_ieq=0.05, avg_salary=80000, value_of_1pct_productivity=400,
        job_creation_baseline=10, job_creation_enhanced=15
    )
    uk = dict(
        baseline_eui=180, high_perf_eui=120, electricity_price=0.18,
        grid_carbon_intensity=0.25, embodied_carbon_baseline=400,
        embodied_carbon_reduction_green=0.20, construction_cost=1800,
        design_premium_green=0.03, design_premium_ultra=0.06, scc=160,
        productivity_gain_hq_ieq=0.04, avg_salary=50000, value_of_1pct_productivity=250,
        job_creation_baseline=12, job_creation_enhanced=14
    )
    add("US", {**common, **us}); add("UK", {**common, **uk})
    with ds.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case","key","value"])
        w.writeheader(); w.writerows(rows)
    return ds

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120,
                    help="Episodes per case (default: 120 for a quick, reliable run).")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Number of seeds per case (default: 1).")
    ap.add_argument("--seed0", type=int, default=42,
                    help="Starting seed (seed0, seed0+1, ...).")
    ap.add_argument("--full", action="store_true",
                    help="Also run sensitivity and robustness and include their times.")
    ap.add_argument("--outdir", default="results",
                    help="Output directory (default: results).")
    args = ap.parse_args()

        root = Path(__file__).resolve().parent
    results = root / args.outdir
    models  = root / "models"
    results.mkdir(exist_ok=True, parents=True); models.mkdir(exist_ok=True, parents=True)

    # Required scripts
    train_py = root / "train.py"
    llm_py   = root / "llm_eval.py"
    for p in [train_py, llm_py]:
        if not p.exists():
            print(f"[ERROR] Missing: {p.name} in repo root.", file=sys.stderr)
                sys.exit(1)

        sens_py  = root / "sensitivity.py"
        rob_py   = root / "robustness.py"
        if args.full:
            for p in [sens_py, rob_py]:
                if not p.exists():
                    print(f"[ERROR] --full requested but missing: {p.name}", file=sys.stderr)
                    sys.exit(1)

        ds_csv = ensure_data_sources(root)

        worker_raw = worker_str()
        worker_tex = _escape_latex(worker_raw)

        times = {}
        logs_path = results / "compute_runtimes_logs.txt"
        with logs_path.open("w", encoding="utf-8") as LOG:
            # Train US/UK over seeds, then average
            for case in ["US", "UK"]:
                acc = []
                for s in range(args.seeds):
                    seed = args.seed0 + s
                    rc, dt, out, err = run_and_time(
                        [sys.executable, str(train_py),
                         "--case", case, "--data", str(ds_csv),
                         "--episodes", str(args.episodes), "--seed", str(seed),
                         "--outdir", str(results), "--modeldir", str(models)], cwd=root)
                    LOG.write(f"\n=== train {case} seed={seed} rc={rc} dt={dt:.3f}s ===\n")
                    LOG.write(out[-600:]); LOG.write("\n---stderr---\n"); LOG.write(err[-600:])
                    if rc != 0:
                        print(f"[ERROR] Training {case} failed (seed={seed}). See {logs_path}.", file=sys.stderr)
                        sys.exit(1)
                    acc.append(dt)
                times[f"train_{case}"] = sum(acc) / len(acc)

            # LLM eval (single run)
            rc, dt, out, err = run_and_time(
                [sys.executable, str(llm_py),
                 "--ref", str(root/"llm_eval_reference.csv"),
                 "--pred", str(root/"llm_eval_outputs.csv"),
                 "--out", str(results/"llm_eval_scores.csv")], cwd=root)
            LOG.write(f"\n=== llm rc={rc} dt={dt:.3f}s ===\n"); LOG.write(out[-600:]); LOG.write("\n---stderr---\n"); LOG.write(err[-600:])
            if rc != 0:
                print("[ERROR] LLM evaluation failed. See logs.", file=sys.stderr)
                sys.exit(1)
            times["llm"] = dt

            # Sensitivity / Robustness (optional; averaged over cases)
            sens_avg = rob_avg = None
            if args.full:
                sens_dts = []
                for case in ["US", "UK"]:
                    rc, dt, out, err = run_and_time(
                        [sys.executable, str(sens_py),
                         "--case", case, "--data", str(ds_csv),
                         "--model", str(models/f"dqn_{case}.pt"),
                         "--out", str(results/f"sensitivity_{case}.csv")], cwd=root)
                    LOG.write(f"\n=== sensitivity {case} rc={rc} dt={dt:.3f}s ===\n"); LOG.write(out[-600:]); LOG.write("\n---stderr---\n"); LOG.write(err[-600:])
                    if rc != 0:
                        print("[ERROR] Sensitivity failed. See logs.", file=sys.stderr)
                        sys.exit(1)
                    sens_dts.append(dt)
                sens_avg = sum(sens_dts) / len(sens_dts)

                rob_dts = []
                for case in ["US", "UK"]:
                    rc, dt, out, err = run_and_time(
                        [sys.executable, str(rob_py),
                         "--case", case, "--data", str(ds_csv),
                         "--model", str(models/f"dqn_{case}.pt"),
                         "--out", str(results/f"robustness_{case}.csv")], cwd=root)
                    LOG.write(f"\n=== robustness {case} rc={rc} dt={dt:.3f}s ===\n"); LOG.write(out[-600:]); LOG.write("\n---stderr---\n"); LOG.write(err[-600:])
                    if rc != 0:
                        print("[ERROR] Robustness failed. See logs.", file=sys.stderr)
                        sys.exit(1)
                    rob_dts.append(dt)
                rob_avg = sum(rob_dts) / len(rob_dts)

        # -------------------- Write JSON --------------------
        out_json = {
            "worker_raw": worker_raw,
            "worker_latex": worker_tex,
            "episodes": args.episodes,
            "seeds": args.seeds,
            "seed0": args.seed0,
            "full": bool(args.full),
            "times_sec": {k: float(v) for k, v in times.items()},
            "sens_avg_sec": (None if sens_avg is None else float(sens_avg)),
            "rob_avg_sec": (None if rob_avg is None else float(rob_avg)),
            "logs": str(logs_path)
        }
        (results / "compute_runtimes_times.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

        # -------------------- Emit LaTeX table --------------------
        # Use the exact manuscript style: tabularx + threeparttable + booktabs
        sens_str = fmt_time(sens_avg) if sens_avg is not None else "---"
        rob_str  = fmt_time(rob_avg)  if rob_avg  is not None else "---"
        tex = (
            "\\begin{table}[h]\n"
            "\\fontsize{10}{12}\\selectfont\n"
            "\\caption{Compute workers and wall-clock runtimes}\n"
            "\\label{Table 7}\n"
            "\\begin{threeparttable}\n"
            "\\begin{tabularx}{\\textwidth}{Y Z Z Z}\n"
            "\\toprule\n"
            "Experiment & Worker (CPU/GPU, RAM) & Episodes/Seeds & Wall-clock time \\\\\n"
            "\\midrule\n"
            f"US training & {worker_tex} & {args.episodes} / $n\\!\\ge\\!{args.seeds}$ (seed={args.seed0}) & {fmt_time(times['train_US'])} \\\\\n"
            f"UK training & {worker_tex} & {args.episodes} / $n\\!\\ge\\!{args.seeds}$ (seed={args.seed0}) & {fmt_time(times['train_UK'])} \\\\\n"
            f"Sensitivity (SCC/Productivity) & {worker_tex} & 5 scenarios $\\times n$ (seed={args.seed0}) & {sens_str} \\\\\n"
            f"Robustness ($\\pm 20\\%$ noise, $+2^\\circ$C) & {worker_tex} & 3 tests $\\times n$ (seed={args.seed0}) & {rob_str} \\\\\n"
            f"LLM evaluation & {worker_tex} & batch size = 3 & {fmt_time(times['llm'])} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabularx}\n"
            "\\begin{tablenotes}\\footnotesize\n"
            "\\item Generated by \\texttt{compute\\_runtimes.py}. For camera-ready reporting, use $n\\!\\ge\\!5$ seeds and full-episode runs, and report medians.\n"
            "\\end{tablenotes}\n"
            "\\end{threeparttable}\n"
            "\\end{table}\n"
        )
        (results / "compute_runtimes_table.tex").write_text(tex, encoding="utf-8")
        print(f"[OK] Wrote: {results/'compute_runtimes_table.tex'} and {results/'compute_runtimes_times.json'}")

    if __name__ == "__main__":
        main()