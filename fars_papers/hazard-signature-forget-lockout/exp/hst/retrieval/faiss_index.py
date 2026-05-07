# FAISS index management: build, search, add, and remove operations
# for dense retrieval over memory store embeddings.
# Uses IndexFlatIP (inner product on L2-normalized vectors = cosine similarity).

import faiss
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


class FaissRetriever:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index: Optional[faiss.IndexIDMap] = None

    def build_index(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        base_index = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIDMap(base_index)
        if ids is None:
            ids = np.arange(embeddings.shape[0], dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

    def search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_embeddings)
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    def add(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        start_id = self.index.ntotal
        if ids is None:
            ids = np.arange(start_id, start_id + embeddings.shape[0], dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

    def remove(self, ids: List[int]) -> None:
        id_array = np.array(ids, dtype=np.int64)
        self.index.remove_ids(id_array)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: str) -> None:
        self.index = faiss.read_index(str(path))

    @property
    def ntotal(self) -> int:
        return self.index.ntotal if self.index else 0
