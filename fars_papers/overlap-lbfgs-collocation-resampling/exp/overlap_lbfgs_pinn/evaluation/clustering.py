# K-means clustering on log-errors to separate converged vs. stuck runs.
# k=2 on [log10(u_err), log10(h_err), log10(B_err)] across seeds.
# If a cluster has <10% of runs, fall back to median/IQR reporting.

import numpy as np
from sklearn.cluster import KMeans


def cluster_errors(error_dicts, min_cluster_frac=0.1):
    u_errs = np.array([d["u_err"] for d in error_dicts])
    h_errs = np.array([d["h_err"] for d in error_dicts])
    B_errs = np.array([d["B_err"] for d in error_dicts])

    X = np.column_stack([np.log10(u_errs + 1e-30),
                         np.log10(h_errs + 1e-30),
                         np.log10(B_errs + 1e-30)])

    n = len(error_dicts)
    if n < 4:
        return _fallback_stats(B_errs, u_errs, h_errs)

    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
    labels = km.labels_
    cluster_sizes = [np.sum(labels == c) for c in range(2)]

    if min(cluster_sizes) / n < min_cluster_frac:
        return _fallback_stats(B_errs, u_errs, h_errs)

    centroids_B = [np.mean(B_errs[labels == c]) for c in range(2)]
    high_cluster = int(np.argmax(centroids_B))
    high_error_frac = cluster_sizes[high_cluster] / n

    return {
        "method": "kmeans",
        "high_error_fraction": high_error_frac,
        "labels": labels.tolist(),
        "cluster_sizes": cluster_sizes,
        "B_err_mean": float(np.mean(B_errs)),
        "B_err_std": float(np.std(B_errs)),
    }


def _fallback_stats(B_errs, u_errs, h_errs):
    return {
        "method": "median_iqr",
        "high_error_fraction": None,
        "B_err_median": float(np.median(B_errs)),
        "B_err_iqr": [float(np.percentile(B_errs, 25)), float(np.percentile(B_errs, 75))],
        "u_err_median": float(np.median(u_errs)),
        "h_err_median": float(np.median(h_errs)),
        "B_err_mean": float(np.mean(B_errs)),
        "B_err_std": float(np.std(B_errs)),
    }
