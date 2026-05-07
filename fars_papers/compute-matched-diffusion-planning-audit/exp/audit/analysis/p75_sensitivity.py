"""p75 wall-clock sensitivity analysis.
Compares median-k vs p75-k best-of-k results to assess robustness of
the compute-matching conclusion to the choice of wall-clock time estimator.
"""
import json
import os
import csv
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR = os.path.join(BASE_DIR, "results", "raw")
TABLE_DIR = os.path.join(BASE_DIR, "results", "tables")

KNOWN_RESULTS = {
    "countdown": {
        "k_median": 35,
        "k_p75": 34,
        "acc_median_k": {"mean": 0.391, "std": 0.014, "seeds": {42: 0.394, 123: 0.402, 456: 0.377}},
        "dream_acc": 0.066,
    },
    "sudoku": {
        "k_median": 39,
        "k_p75": 39,
        "acc_median_k": {"mean": 0.672, "std": 0.004, "seeds": {42: 0.676, 123: 0.670, 456: 0.670}},
        "dream_acc": 0.776,
    },
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_accuracy(filepath):
    data = load_jsonl(filepath)
    correct = sum(1 for d in data if d.get("solved", False))
    return correct / len(data) if data else 0.0


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    seeds = [42, 123, 456]

    rows = []
    for task in ["countdown", "sudoku"]:
        info = KNOWN_RESULTS[task]
        k_median = info["k_median"]
        k_p75 = info["k_p75"]
        dream_acc = info["dream_acc"]
        acc_median_mean = info["acc_median_k"]["mean"]
        acc_median_std = info["acc_median_k"]["std"]

        if k_p75 == k_median:
            acc_p75_mean = acc_median_mean
            acc_p75_std = acc_median_std
            note = "k_p75 == k_median; reusing median-k results"
        else:
            p75_accs = []
            for seed in seeds:
                fpath = os.path.join(RAW_DIR, f"qwen_bok_p75_{task}_seed{seed}.jsonl")
                acc = compute_accuracy(fpath)
                p75_accs.append(acc)
                print(f"  {task} seed={seed}: acc={acc:.4f}")
            acc_p75_mean = float(np.mean(p75_accs))
            acc_p75_std = float(np.std(p75_accs, ddof=0))
            note = ""

        delta_median = dream_acc - acc_median_mean
        delta_p75 = dream_acc - acc_p75_mean

        task_label = "Countdown" if task == "countdown" else "Mini Sudoku"
        rows.append({
            "task": task_label,
            "k_median": k_median,
            "acc_median_k": acc_median_mean,
            "acc_median_k_std": acc_median_std,
            "k_p75": k_p75,
            "acc_p75_k": acc_p75_mean,
            "acc_p75_k_std": acc_p75_std,
            "dream_acc": dream_acc,
            "delta_median": delta_median,
            "delta_p75": delta_p75,
            "note": note,
        })

    print("\n" + "=" * 90)
    print("p75 Wall-Clock Sensitivity Analysis: Median-k vs p75-k")
    print("=" * 90)
    hdr = f"{'Task':<14} {'k_med':>5} {'Acc(med-k)':>14} {'k_p75':>5} {'Acc(p75-k)':>14} {'Dream':>8} {'D(med)':>8} {'D(p75)':>8}"
    print(hdr)
    print("-" * 90)
    for r in rows:
        print(f"{r['task']:<14} {r['k_median']:>5} "
              f"{r['acc_median_k']*100:>5.1f}%+/-{r['acc_median_k_std']*100:.1f}% "
              f"{r['k_p75']:>5} "
              f"{r['acc_p75_k']*100:>5.1f}%+/-{r['acc_p75_k_std']*100:.1f}% "
              f"{r['dream_acc']*100:>6.1f}% "
              f"{r['delta_median']*100:>+6.1f}pp "
              f"{r['delta_p75']*100:>+6.1f}pp")
    print("-" * 90)

    print("\nRobustness Assessment:")
    for r in rows:
        same_sign = (r["delta_median"] > 0) == (r["delta_p75"] > 0)
        mag_change = abs(r["delta_p75"] - r["delta_median"]) * 100
        print(f"  {r['task']}: Delta(median)={r['delta_median']*100:+.1f}pp, "
              f"Delta(p75)={r['delta_p75']*100:+.1f}pp, "
              f"same direction={same_sign}, magnitude change={mag_change:.1f}pp")

    all_same = all((r["delta_median"] > 0) == (r["delta_p75"] > 0) for r in rows)
    if all_same:
        print("\nCONCLUSION: Qualitative conclusions are ROBUST to the choice of "
              "wall-clock time estimator (median vs p75).")
    else:
        print("\nCONCLUSION: Qualitative conclusions DIFFER between median and p75 "
              "estimators on at least one task.")

    csv_file = os.path.join(TABLE_DIR, "p75_sensitivity.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_file}")

    result = {
        "analysis": "p75_wall_clock_sensitivity",
        "description": "Robustness check: median-k vs p75-k best-of-k evaluation",
        "tasks": {},
        "robust": all_same,
    }
    for r in rows:
        result["tasks"][r["task"]] = {
            "k_median": r["k_median"],
            "k_p75": r["k_p75"],
            "acc_median_k": f"{r['acc_median_k']*100:.1f}% +/- {r['acc_median_k_std']*100:.1f}%",
            "acc_p75_k": f"{r['acc_p75_k']*100:.1f}% +/- {r['acc_p75_k_std']*100:.1f}%",
            "dream_acc": f"{r['dream_acc']*100:.1f}%",
            "delta_median_pp": round(r["delta_median"] * 100, 1),
            "delta_p75_pp": round(r["delta_p75"] * 100, 1),
            "same_direction": bool((r["delta_median"] > 0) == (r["delta_p75"] > 0)),
        }

    result_file = os.path.join(TABLE_DIR, "p75_sensitivity.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON saved to {result_file}")

    return result


if __name__ == "__main__":
    main()
