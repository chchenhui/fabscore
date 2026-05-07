"""Paired bootstrap confidence intervals and significance tests.

Computes 95% CI for accuracy difference between two conditions
using 10,000 bootstrap resamples.
"""

import numpy as np
from typing import Any


def paired_bootstrap_ci(
    correct_a: list[bool],
    correct_b: list[bool],
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    assert len(correct_a) == len(correct_b), "Lists must have equal length"
    n = len(correct_a)
    arr_a = np.array(correct_a, dtype=float)
    arr_b = np.array(correct_b, dtype=float)

    observed_diff = arr_a.mean() - arr_b.mean()

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        diffs[i] = arr_a[idx].mean() - arr_b[idx].mean()

    alpha = 1.0 - ci_level
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))

    excludes_zero = (lo > 0) or (hi < 0)

    return {
        "observed_diff": float(observed_diff),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "ci_level": ci_level,
        "n_resamples": n_resamples,
        "excludes_zero": bool(excludes_zero),
        "n_items": n,
        "mean_a": float(arr_a.mean()),
        "mean_b": float(arr_b.mean()),
    }
