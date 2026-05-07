"""Evaluation metrics for HNSW topology leakage audit.

compute_recall_at_k: Recall@k = |true ∩ pred| / k averaged over all nodes.
compute_true_knn: exact kNN via FAISS IndexFlatL2 (GPU if available, else CPU).
compute_spearman_correlation: Spearman rank correlation between graph-geodesic
    and true Euclidean distances over sampled node pairs.
"""

import numpy as np
import faiss
from scipy.stats import spearmanr


def compute_recall_at_k(
    predicted_neighbors: dict[int, list[int]],
    true_neighbors: dict[int, list[int]],
    k: int,
) -> float:
    n = len(true_neighbors)
    total_recall = 0.0
    for i in range(n):
        true_set = set(true_neighbors[i][:k])
        pred_set = set(predicted_neighbors.get(i, [])[:k])
        total_recall += len(true_set & pred_set) / k
    return total_recall / n


def compute_true_knn(
    vectors: np.ndarray,
    k: int,
) -> dict[int, list[int]]:
    n, d = vectors.shape
    index = faiss.IndexFlatL2(d)

    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(vectors)
        distances, indices = gpu_index.search(vectors, k + 1)
    except Exception:
        index.add(vectors)
        distances, indices = index.search(vectors, k + 1)

    true_knn = {}
    for i in range(n):
        nbrs = [int(j) for j in indices[i] if j != i][:k]
        true_knn[i] = nbrs
    return true_knn


def compute_spearman_correlation(
    graph_distances: dict[tuple[int, int], float],
    true_distances: dict[tuple[int, int], float],
    n_sample_pairs: int = 10000,
    seed: int = 42,
) -> float:
    pairs = list(set(graph_distances.keys()) & set(true_distances.keys()))
    rng = np.random.default_rng(seed)
    if len(pairs) > n_sample_pairs:
        idx = rng.choice(len(pairs), size=n_sample_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    g_dists = np.array([graph_distances[p] for p in pairs])
    t_dists = np.array([true_distances[p] for p in pairs])

    corr, _ = spearmanr(g_dists, t_dists)
    return float(corr)
