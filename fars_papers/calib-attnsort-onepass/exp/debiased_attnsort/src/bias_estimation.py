# Per-prompt position-bias estimation module.
# Estimates the positional bias curve from attention scores and uses it
# to produce debiased attention scores for document sorting.
# Algorithm: trim top-alpha outliers (gold doc), bin by position, compute
# aggregation per bin, linearly interpolate to get bias curve b_hat(p).
# Supports additive (a - bias) and divisive (a / bias) debiasing modes.

import numpy as np


def estimate_position_bias(a, positions, alpha=0.05, num_bins=20, aggregation="median"):
    a = np.asarray(a, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    threshold = np.quantile(a, 1.0 - alpha)
    mask = a <= threshold
    a_trimmed = a[mask]
    pos_trimmed = positions[mask]

    bin_edges = np.linspace(0, positions.max() + 1, num_bins + 1)
    bin_vals = []
    bin_centers = []
    for j in range(num_bins):
        in_bin = (pos_trimmed >= bin_edges[j]) & (pos_trimmed < bin_edges[j + 1])
        if np.sum(in_bin) > 0:
            vals = a_trimmed[in_bin]
            if aggregation == "median":
                bin_vals.append(np.median(vals))
            elif aggregation == "mean":
                bin_vals.append(np.mean(vals))
            elif aggregation == "p75":
                bin_vals.append(np.percentile(vals, 75))
            else:
                bin_vals.append(np.median(vals))
            bin_centers.append((bin_edges[j] + bin_edges[j + 1]) / 2)

    if len(bin_centers) < 2:
        return np.full_like(a, np.mean(a_trimmed) if len(a_trimmed) > 0 else 0.0)

    bias_curve = np.interp(positions, bin_centers, bin_vals)
    return bias_curve


def debias_scores(a, positions, alpha=0.05, num_bins=20, aggregation="median", mode="additive"):
    bias_curve = estimate_position_bias(a, positions, alpha, num_bins, aggregation)
    if mode == "divisive":
        bias_curve_safe = np.maximum(bias_curve, 1e-10)
        debiased = a / bias_curve_safe
    else:
        debiased = a - bias_curve
    return debiased, bias_curve
