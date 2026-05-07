"""Geodesic reconstruction via shortest-path distances + Landmark MDS.

Converts an HNSW adjacency list to a sparse graph, computes shortest-path
distances from L landmark nodes to all nodes via Dijkstra, then embeds into
d dimensions using Landmark MDS. Supports either a callable edge_weight_fn or
a precomputed weight_matrix (csr_matrix) for flexible edge weighting.
"""

from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .landmark_mds import landmark_mds


def geodesic_reconstruct(
    adjacency_list: dict[int, list[int]],
    n_landmarks: int = 256,
    n_components: int = 32,
    edge_weight_fn: Callable[[int, int, dict[int, list[int]]], float] | None = None,
    weight_matrix: csr_matrix | None = None,
    seed: int = 42,
) -> np.ndarray:
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    if weight_matrix is not None:
        graph = weight_matrix
    else:
        if edge_weight_fn is None:
            edge_weight_fn = lambda u, v, adj: 1.0

        rows, cols, weights = [], [], []
        for u in nodes:
            u_idx = node_to_idx[u]
            for v in adjacency_list[u]:
                v_idx = node_to_idx[v]
                w = edge_weight_fn(u, v, adjacency_list)
                rows.append(u_idx)
                cols.append(v_idx)
                weights.append(w)

        graph = csr_matrix((weights, (rows, cols)), shape=(n, n))

    rng = np.random.default_rng(seed)
    landmark_indices = rng.choice(n, size=min(n_landmarks, n), replace=False)
    landmark_indices = np.sort(landmark_indices)

    dist_matrix = dijkstra(graph, directed=False, indices=landmark_indices)

    reachable_mask = np.isfinite(dist_matrix)
    if not reachable_mask.all():
        max_finite = np.nanmax(dist_matrix[reachable_mask])
        dist_matrix[~reachable_mask] = max_finite * 2.0

    embedding = landmark_mds(dist_matrix, n_components, landmark_indices=landmark_indices)

    if n == len(nodes) and nodes == list(range(n)):
        return embedding

    full_embedding = np.zeros((max(nodes) + 1, n_components))
    for node, idx in node_to_idx.items():
        full_embedding[node] = embedding[idx]
    return full_embedding
