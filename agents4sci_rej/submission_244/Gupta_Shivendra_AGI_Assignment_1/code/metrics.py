#!/usr/bin/env python3
"""
metrics.py — Recompute Acc/AUC from saved NPY artifacts.

What it does
------------
- Loads `val_true.npy` and a score file (default: `val_score_proposed.npy`)
  from one or more run directories.
- Computes Accuracy at a threshold (default 0.5) and AUC (rank/Mann–Whitney with tie handling).
- (Optional) Writes per-run metrics to `metrics.json`.
- If `all_results.json` exists, prints a small diff to help spot mismatches.

Usage
-----
# Single run (recompute and print)
python code/metrics.py --dirs results/20250908_141019

# Multiple runs (aggregate mean ± std) and write metrics.json per run
python code/metrics.py --dirs results/20250908_* --write

# Use a different score filename inside each run dir
python code/metrics.py --dirs results/20250908_141019 --score-name val_score_baseline.npy
"""
import argparse, json
from pathlib import Path
import numpy as np

def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Return 1-based ranks with average for ties, like scipy.stats.rankdata(method='average').
    """
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")   # stable
    ranks = np.empty_like(order, dtype=float)

    # traverse groups of equal values
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        # elements order[i:j] form a tie-group; average rank of [i+1, j]
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks

def auc_mann_whitney(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Binary AUC via Mann–Whitney U statistic with proper tie handling.
    AUC = (sum_ranks_pos - P*(P+1)/2) / (P*N)
    where ranks are 1..(P+N) over scores (higher score = higher rank).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan")

    # Rank by score (ascending), then flip to "higher score => higher rank"
    r = _rankdata_average_ties(y_score)  # 1..(P+N), low score gets low rank
    r = (P + N + 1) - r                  # invert so higher score => higher rank

    sum_ranks_pos = float(r[y_true == 1].sum())
    auc = (sum_ranks_pos - P * (P + 1) / 2.0) / (P * N)
    return auc

def acc_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= thr).astype(int)
    return float((y_pred == y_true).mean())

def eval_run_dir(run_dir: Path, score_name: str, thr: float):
    y_true_p = run_dir / "val_true.npy"
    y_score_p = run_dir / score_name

    if not y_true_p.exists():
        raise FileNotFoundError(f"Missing {y_true_p}")
    if not y_score_p.exists():
        # try a fallback name
        alt = run_dir / "val_score.npy"
        if alt.exists():
            y_score_p = alt
        else:
            raise FileNotFoundError(f"Missing {y_score_p} (and fallback {alt})")

    y_true = np.load(y_true_p)
    y_score = np.load(y_score_p)

    acc = acc_at_threshold(y_true, y_score, thr=thr)
    auc = auc_mann_whitney(y_true, y_score)
    out = {"acc": acc, "auc": auc, "n": int(len(y_true)), "threshold": thr, "score_file": str(y_score_p.name)}
    return out

def compare_with_json(run_dir: Path, metrics_now: dict):
    """
    If all_results.json exists, show a tiny diff against finals['proposed'] or
    a finals entry whose key matches the score filename.
    """
    p = run_dir / "all_results.json"
    if not p.exists():
        return

    try:
        obj = json.loads(p.read_text())
    except Exception:
        return

    finals = obj.get("finals", {})
    # Guess which finals entry to compare to
    key_guess = "proposed" if "proposed" in finals else next(iter(finals), None)
    if key_guess is None:
        return
    ref = finals.get(key_guess, {})
    ref_acc = ref.get("acc")
    ref_auc = ref.get("auc")

    if ref_acc is not None or ref_auc is not None:
        print(f"  ↳ all_results.json / finals['{key_guess}']: acc={ref_acc}, auc={ref_auc}")
        if ref_acc is not None:
            print(f"    Δacc = {metrics_now['acc'] - ref_acc:+.6f}")
        if ref_auc is not None:
            print(f"    Δauc = {metrics_now['auc'] - ref_auc:+.6f}")

def main():
    ap = argparse.ArgumentParser(description="Recompute Acc/AUC from saved NPYs.")
    ap.add_argument("--dirs", nargs="+", required=True, help="Run directories (each has val_true.npy + val_score_*.npy)")
    ap.add_argument("--score-name", default="val_score_proposed.npy", help="Score filename inside each run dir")
    ap.add_argument("--thr", type=float, default=0.5, help="Threshold used for accuracy")
    ap.add_argument("--write", action="store_true", help="Write per-run metrics.json")
    args = ap.parse_args()

    rows = []
    print("=== Recomputed metrics ===")
    for d in args.dirs:
        run_dir = Path(d)
        m = eval_run_dir(run_dir, args.score_name, args.thr)
        rows.append((run_dir, m))
        print(f"{run_dir}: Acc={m['acc']:.6f}  AUC={m['auc']:.6f}  N={m['n']}  ({m['score_file']})")
        compare_with_json(run_dir, m)
        if args.write:
            out = run_dir / "metrics.json"
            out.write_text(json.dumps(m, indent=2))
            print(f"  wrote {out}")

    if len(rows) > 1:
        accs = np.array([m["acc"] for _, m in rows], dtype=float)
        aucs = np.array([m["auc"] for _, m in rows], dtype=float)
        print("\nAggregate (mean ± std over runs):")
        # std with ddof=1 only if >=2 runs
        ddof = 1 if len(rows) >= 2 else 0
        acc_std = float(accs.std(ddof=ddof))
        auc_std = float(aucs.std(ddof=ddof))
        print(f"Acc = {float(accs.mean()):.6f} ± {acc_std:.6f}")
        print(f"AUC = {float(aucs.mean()):.6f} ± {auc_std:.6f}")

if __name__ == "__main__":
    main()
