"""FAISS HNSW index construction and layer-0 graph extraction utilities.

build_hnsw_index: builds IndexHNSWFlat with given params.
extract_layer0_adjacency: extracts the undirected layer-0 adjacency list
from the FAISS HNSW internal storage (neighbors + offsets arrays).
"""

from collections import defaultdict

import faiss
import numpy as np


def build_hnsw_index(
    vectors: np.ndarray,
    M: int = 32,
    efConstruction: int = 200,
    seed: int = 42,
) -> faiss.IndexHNSWFlat:
    n, d = vectors.shape
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    faiss.ParameterSpace().set_index_parameter(index, "efSearch", 64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    index.add(vectors[perm])

    return index, perm


def extract_layer0_adjacency(index: faiss.IndexHNSWFlat):
    hnsw = index.hnsw
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets)

    n = index.ntotal
    M = hnsw.nb_neighbors(0)

    adjacency = defaultdict(set)
    degrees = {}

    for node_id in range(n):
        begin = int(offsets[node_id])
        end = begin + M
        nbrs = neighbors[begin:end]
        valid = nbrs[nbrs >= 0]
        adjacency[node_id] = set(int(x) for x in valid)

    for u in range(n):
        for v in adjacency[u]:
            adjacency[v].add(u)

    adjacency_list = {u: sorted(adjacency[u]) for u in range(n)}
    degrees = {u: len(adjacency_list[u]) for u in range(n)}

    return adjacency_list, degrees
