# MemoryStore: wraps FaissRetriever with commit-time tombstone policy.
# Manages add/delete/query operations, tracking blocked and accepted writes.
# Supports both exact-match and fuzzy (set-containment) tombstone policies.

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from hst.memory.policies import CommitTimeTombstone, CommitTimeTombstoneFuzzy
from hst.retrieval.faiss_index import FaissRetriever

PolicyType = Union[CommitTimeTombstone, CommitTimeTombstoneFuzzy, None]


class MemoryStore:
    def __init__(
        self,
        retriever: FaissRetriever,
        policy: PolicyType,
        int_to_record_id: Dict[int, str],
        hs_map: Dict[str, dict],
    ):
        self.retriever = retriever
        self.policy = policy
        self.int_to_record_id = dict(int_to_record_id)
        self.hs_map = hs_map
        self.blocked_writes: List[str] = []
        self.accepted_writes: List[str] = []
        self._next_int_id = max(int_to_record_id.keys()) + 1 if int_to_record_id else 0

    def add_record(self, record_id: str, embedding: np.ndarray) -> bool:
        if self.policy and not self.policy.check_write(record_id, self.hs_map):
            self.blocked_writes.append(record_id)
            return False

        int_id = self._next_int_id
        self._next_int_id += 1
        emb = embedding.reshape(1, -1)
        ids = np.array([int_id], dtype=np.int64)
        self.retriever.add(emb, ids=ids)
        self.int_to_record_id[int_id] = record_id
        self.accepted_writes.append(record_id)
        return True

    def delete_records(self, record_ids: List[str]) -> Set[str]:
        reverse_map = {v: k for k, v in self.int_to_record_id.items()}
        int_ids_to_remove = []
        tombstone_hs = set()

        for rid in record_ids:
            if rid in reverse_map:
                int_ids_to_remove.append(reverse_map[rid])
                del self.int_to_record_id[reverse_map[rid]]
            hs_entry = self.hs_map.get(rid, {})
            if "hs" in hs_entry:
                tombstone_hs.add(hs_entry["hs"])

        if int_ids_to_remove:
            self.retriever.remove(int_ids_to_remove)

        return tombstone_hs

    def delete_records_structured(self, record_ids: List[str]) -> List[Dict]:
        reverse_map = {v: k for k, v in self.int_to_record_id.items()}
        int_ids_to_remove = []
        tombstone_sigs = []

        for rid in record_ids:
            if rid in reverse_map:
                int_ids_to_remove.append(reverse_map[rid])
                del self.int_to_record_id[reverse_map[rid]]
            hs_entry = self.hs_map.get(rid, {})
            if hs_entry:
                tombstone_sigs.append(hs_entry)

        if int_ids_to_remove:
            self.retriever.remove(int_ids_to_remove)

        return tombstone_sigs

    def query(self, query_emb: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        distances, indices = self.retriever.search(query_emb.reshape(1, -1), k=k)
        result_ids = []
        result_scores = []
        for j in range(k):
            int_id = int(indices[0][j])
            if int_id < 0:
                continue
            if int_id in self.int_to_record_id:
                result_ids.append(self.int_to_record_id[int_id])
                result_scores.append(float(distances[0][j]))
        return result_ids, result_scores
