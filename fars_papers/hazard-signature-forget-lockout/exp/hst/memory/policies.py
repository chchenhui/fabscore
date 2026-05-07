# Retrieval-time and commit-time policy implementations for HST experiments.
# Exact-match policies (original) and fuzzy set-containment policies (optimized).
# Fuzzy matching uses hazard-set containment: a record is blocked if its hazard
# set is a subset or superset of any tombstoned entry's hazard set.

from typing import List, Set, Dict, Tuple, FrozenSet
import numpy as np

from hst.retrieval.faiss_index import FaissRetriever


def _parse_tombstone_hazards(tombstone_sigs: List[Dict]) -> List[FrozenSet[str]]:
    return [frozenset(sig.get("hazards", [])) for sig in tombstone_sigs]


def _fuzzy_hazard_match(record_hazards: FrozenSet[str],
                        tombstone_hazard_sets: List[FrozenSet[str]]) -> bool:
    if not record_hazards:
        return False
    for t_hz in tombstone_hazard_sets:
        if not t_hz:
            continue
        if record_hazards <= t_hz or t_hz <= record_hazards:
            return True
    return False


class RetrievalTimeDenylist:
    def __init__(self, tombstone_set: Set[str], delta: int = 6):
        self.tombstone_set = tombstone_set
        self.delta = delta

    def filter_results(
        self,
        query_emb: np.ndarray,
        retriever: FaissRetriever,
        int_to_record_id: Dict[int, str],
        hs_map: Dict[str, str],
        k: int,
    ) -> Tuple[List[str], List[float], int]:
        fetch_k = k + self.delta
        distances, indices = retriever.search(
            query_emb.reshape(1, -1), k=fetch_k
        )

        passing_ids = []
        passing_scores = []
        n_filtered = 0

        for j in range(fetch_k):
            int_id = int(indices[0][j])
            if int_id < 0:
                continue
            record_id = int_to_record_id[int_id]
            record_hs = hs_map.get(record_id, "")
            if record_hs in self.tombstone_set:
                n_filtered += 1
                continue
            passing_ids.append(record_id)
            passing_scores.append(float(distances[0][j]))
            if len(passing_ids) >= k:
                break

        return passing_ids, passing_scores, n_filtered


class RetrievalTimeDenylistFuzzy:
    def __init__(self, tombstone_sigs: List[Dict], delta: int = 6):
        self.tombstone_hazard_sets = _parse_tombstone_hazards(tombstone_sigs)
        self.delta = delta

    def filter_results(
        self,
        query_emb: np.ndarray,
        retriever: FaissRetriever,
        int_to_record_id: Dict[int, str],
        hs_map: Dict[str, dict],
        k: int,
    ) -> Tuple[List[str], List[float], int]:
        fetch_k = k + self.delta
        distances, indices = retriever.search(
            query_emb.reshape(1, -1), k=fetch_k
        )

        passing_ids = []
        passing_scores = []
        n_filtered = 0

        for j in range(fetch_k):
            int_id = int(indices[0][j])
            if int_id < 0:
                continue
            record_id = int_to_record_id[int_id]
            record_sig = hs_map.get(record_id, {})
            record_hz = frozenset(record_sig.get("hazards", []))
            if _fuzzy_hazard_match(record_hz, self.tombstone_hazard_sets):
                n_filtered += 1
                continue
            passing_ids.append(record_id)
            passing_scores.append(float(distances[0][j]))
            if len(passing_ids) >= k:
                break

        return passing_ids, passing_scores, n_filtered


class CommitTimeTombstone:
    def __init__(self, tombstone_set: Set[str]):
        self.tombstone_set = tombstone_set

    def check_write(self, record_id: str, hs_map: Dict[str, dict]) -> bool:
        hs_value = hs_map.get(record_id, {}).get("hs", "")
        return hs_value not in self.tombstone_set


class CommitTimeTombstoneFuzzy:
    def __init__(self, tombstone_sigs: List[Dict]):
        self.tombstone_hazard_sets = _parse_tombstone_hazards(tombstone_sigs)

    def check_write(self, record_id: str, hs_map: Dict[str, dict]) -> bool:
        record_sig = hs_map.get(record_id, {})
        record_hz = frozenset(record_sig.get("hazards", []))
        return not _fuzzy_hazard_match(record_hz, self.tombstone_hazard_sets)
