# Metrics computation: PRP@k (Poisoned Retrieval Proportion),
# Benign Recall@k, WriteBlockRate, BenignFalseBlock.
# These are shared across all experimental conditions.

from typing import List, Set, Dict, Callable, Optional


def prp_at_k(retrieved_ids: List[List[str]], poisoned_id_set: Set[str], k: int) -> float:
    total_slots = k * len(retrieved_ids)
    if total_slots == 0:
        return 0.0
    poisoned_count = sum(
        1 for query_results in retrieved_ids
        for rid in query_results[:k]
        if rid in poisoned_id_set
    )
    return poisoned_count / total_slots


def benign_recall_at_k(retrieved_ids_per_query: List[List[str]],
                       ground_truth_ids: List[str], k: int) -> float:
    assert len(retrieved_ids_per_query) == len(ground_truth_ids)
    if len(ground_truth_ids) == 0:
        return 0.0
    recalled = sum(
        1 for gt_id, results in zip(ground_truth_ids, retrieved_ids_per_query)
        if gt_id in results[:k]
    )
    return recalled / len(ground_truth_ids)


def write_block_rate(attempted_writes: List[str], blocked_writes: List[str]) -> float:
    if len(attempted_writes) == 0:
        return 0.0
    return len(blocked_writes) / len(attempted_writes)


def benign_false_block(benign_records: List[Dict], tombstone_set: Set[str],
                       hs_func: Optional[Callable] = None) -> float:
    if len(benign_records) == 0 or hs_func is None:
        return 0.0
    blocked = sum(
        1 for r in benign_records
        if hs_func(r) in tombstone_set
    )
    return blocked / len(benign_records)
