"""Edge weight functions for HNSW graph reconstruction.

Provides unit_weight (all 1.0) and degree_penalized_weight (hub-penalty)
returning scipy.sparse.csr_matrix for direct use in geodesic_reconstruct().
"""

import numpy as np
from scipy.sparse import csr_matrix


def unit_weight(adjacency_list: dict[int, list[int]]) -> csr_matrix:
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    rows, cols, weights = [], [], []
    for u in nodes:
        u_idx = node_to_idx[u]
        for v in adjacency_list[u]:
            v_idx = node_to_idx[v]
            rows.append(u_idx)
            cols.append(v_idx)
            weights.append(1.0)

    return csr_matrix((weights, (rows, cols)), shape=(n, n))


def degree_penalized_weight(
    adjacency_list: dict[int, list[int]],
    alpha: float = 1.0,
) -> csr_matrix:
    nodes = sorted(adjacency_list.keys())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    degrees = {node: len(adjacency_list[node]) for node in nodes}

    rows, cols, weights = [], [], []
    for u in nodes:
        u_idx = node_to_idx[u]
        log_deg_u = np.log(1 + degrees[u])
        for v in adjacency_list[u]:
            v_idx = node_to_idx[v]
            log_deg_v = np.log(1 + degrees[v])
            w = 1.0 + alpha * (log_deg_u + log_deg_v) / 2.0
            rows.append(u_idx)
            cols.append(v_idx)
            weights.append(w)

    return csr_matrix((weights, (rows, cols)), shape=(n, n))
