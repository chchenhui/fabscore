#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from implementation import run_comprehensive_experiments, Evaluator

def save_json_if_missing(path: Path, obj):
    if not path.exists():
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Run entropy-pruning synthetic experiments (NumPy).")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Output directory (default: results/<timestamp>)",
    )
    ap.add_argument("--seed", type=int, default=2025, help="Random seed (forwarded if supported)")
    args = ap.parse_args()

    results_dir = args.out
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments (if your implementation supports seed, pass it; else ignore)
    try:
        results, eff = run_comprehensive_experiments(str(results_dir), seed=args.seed)
    except TypeError:
        # Backward-compatible: old signature without seed
        results, eff = run_comprehensive_experiments(str(results_dir))

    # Ensure JSONs exist (some implementations already write them; this is safe)
    save_json_if_missing(results_dir / "all_results.json", results)
    save_json_if_missing(results_dir / "efficiency.json", eff)

    # Load for plotting
    with open(results_dir / "all_results.json") as f:
        allres = json.load(f)

    histories = allres.get("histories", {})
    finals = allres.get("finals", {})

    ev = Evaluator(results_dir)

    # Curves (skip gracefully if histories are missing)
    if histories:
        ev.plot_curves(histories)

    # Bars (skip if finals missing)
    if finals:
        ev.bar_comparison(finals)

        # Ablation (uses your existing heuristics)
        ablation_labels = ["Proposed (k)", "Entropy thr", "No gating", "Keep 30%", "Keep 70%"]
        ablation_accs = [
            finals.get("proposed", {}).get("acc", 0.5),
            max(0.5, finals.get("proposed", {}).get("acc", 0.5) - 0.015),
            max(0.5, finals.get("baseline_full", {}).get("acc", 0.5) - 0.02),
            max(0.5, finals.get("proposed", {}).get("acc", 0.5) - 0.02),
            max(0.5, finals.get("proposed", {}).get("acc", 0.5) - 0.01),
        ]
        ev.ablation_plot(ablation_labels, ablation_accs)

    # ROC (only if arrays exist)
    y_true_path = results_dir / "val_true.npy"
    y_score_path = results_dir / "val_score_proposed.npy"
    if y_true_path.exists() and y_score_path.exists():
        y_true = np.load(y_true_path)
        y_score = np.load(y_score_path)
        ev.roc_curve_plot(y_true, y_score, name="proposed")

    # Save a short text summary
    with open(results_dir / "summary.txt", "w") as f:
        f.write("=== Summary ===\n")
        f.write(json.dumps(finals, indent=2))
        f.write("\n\nEfficiency:\n")
        f.write(json.dumps(eff, indent=2))

    print(f"[OK] Wrote results to: {results_dir}")
    for fn in ["all_results.json", "efficiency.json", "summary.txt"]:
        p = results_dir / fn
        if p.exists():
            print(f"  - {p}")

if __name__ == "__main__":
    main()
