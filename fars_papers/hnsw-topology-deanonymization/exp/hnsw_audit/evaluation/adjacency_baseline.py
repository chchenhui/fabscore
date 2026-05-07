"""Adjacency-only baseline: treat HNSW neighbor lists as predicted kNN sets.

adjacency_only_recall: for each node, take first k neighbors from its HNSW
adjacency list and compute Recall@k against ground-truth kNN.
"""

import numpy as np

from evaluation.metrics import compute_recall_at_k


def adjacency_only_recall(
    adjacency_list: dict[int, list[int]],
    true_knn: dict[int, list[int]],
    k: int,
) -> float:
    predicted = {}
    for node_id in range(len(true_knn)):
        nbrs = adjacency_list.get(node_id, [])
        predicted[node_id] = nbrs[:k]
    return compute_recall_at_k(predicted, true_knn, k)
