# Bootstrap resampling utilities for confidence intervals on metric differences.
# Resample test battles with replacement, recompute metrics, return stats + 95% CI.

import numpy as np


def bootstrap_metric(test_battles, model, metric_fn, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(test_battles)

    all_probs = model.predict_probs_batch(test_battles)
    true_labels = test_battles['preference'].values

    base_value = metric_fn(all_probs, true_labels)

    boot_values = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_probs = all_probs[idx]
        boot_labels = true_labels[idx]
        boot_values[b] = metric_fn(boot_probs, boot_labels)

    return {
        'value': float(base_value),
        'mean': float(boot_values.mean()),
        'std': float(boot_values.std()),
        'ci_lower': float(np.percentile(boot_values, 2.5)),
        'ci_upper': float(np.percentile(boot_values, 97.5)),
    }
