# Evaluation metrics: 4-way NLL, Brier score, ECE, per-class NLL.
# Shared across BT, AB-MNL, and GRK models.

import numpy as np

OUTCOME_MAP = {'A': 0, 'B': 1, 'TIE': 2, 'BOTH_BAD': 3}
OUTCOME_NAMES = ['A', 'B', 'TIE', 'BOTH_BAD']


def _labels_to_indices(true_labels):
    if isinstance(true_labels[0], str):
        return np.array([OUTCOME_MAP[l] for l in true_labels])
    return np.asarray(true_labels)


def four_way_nll(predicted_probs, true_labels):
    predicted_probs = np.asarray(predicted_probs)
    indices = _labels_to_indices(true_labels)
    n = len(indices)
    log_probs = np.log(np.clip(predicted_probs[np.arange(n), indices], 1e-15, None))
    return float(-log_probs.mean())


def per_class_nll(predicted_probs, true_labels):
    predicted_probs = np.asarray(predicted_probs)
    indices = _labels_to_indices(true_labels)
    n = len(indices)
    log_probs = np.log(np.clip(predicted_probs[np.arange(n), indices], 1e-15, None))
    results = {}
    for cls_idx, cls_name in enumerate(OUTCOME_NAMES):
        mask = indices == cls_idx
        if mask.sum() > 0:
            results[cls_name] = float(-log_probs[mask].mean())
        else:
            results[cls_name] = float('nan')
    return results


def brier_score_bothbad(predicted_probs, true_labels):
    predicted_probs = np.asarray(predicted_probs)
    indices = _labels_to_indices(true_labels)
    p_bothbad = predicted_probs[:, 3]
    y_bothbad = (indices == 3).astype(np.float64)
    return float(np.mean((p_bothbad - y_bothbad) ** 2))


def ece_bothbad(predicted_probs, true_labels, n_bins=10):
    predicted_probs = np.asarray(predicted_probs)
    indices = _labels_to_indices(true_labels)
    p_bothbad = predicted_probs[:, 3]
    y_bothbad = (indices == 3).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(p_bothbad)
    for i in range(n_bins):
        mask = (p_bothbad >= bin_edges[i]) & (p_bothbad < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p_bothbad >= bin_edges[i]) & (p_bothbad <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin > 0:
            avg_pred = p_bothbad[mask].mean()
            avg_true = y_bothbad[mask].mean()
            ece += (n_bin / total) * abs(avg_pred - avg_true)
    return float(ece)
