"""Retrieval evaluation using FAISS exact inner-product search and BEIR metrics.

Given L2-normalized query and corpus embeddings plus qrels, retrieves top-100
results per query using FAISS FlatIP (cosine similarity), then computes
nDCG@10 and Recall@10 via BEIR's EvaluateRetrieval.
"""

import numpy as np
import faiss

from beir.retrieval.evaluation import EvaluateRetrieval


def retrieve_top_k(
    query_embeds: np.ndarray,
    corpus_embeds: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    d = corpus_embeds.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(corpus_embeds.astype(np.float32))

    scores, indices = index.search(query_embeds.astype(np.float32), top_k)

    results = {}
    for i, qid in enumerate(query_ids):
        results[qid] = {}
        for rank in range(top_k):
            idx = indices[i, rank]
            if idx < 0:
                continue
            cid = corpus_ids[idx]
            results[qid][cid] = float(scores[i, rank])
    return results


def evaluate_retrieval(
    query_embeds: np.ndarray,
    corpus_embeds: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    qrels: dict[str, dict[str, int]],
    top_k: int = 100,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    if k_values is None:
        k_values = [10]

    results = retrieve_top_k(query_embeds, corpus_embeds, query_ids, corpus_ids, top_k)

    ndcg, map_scores, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, k_values
    )

    metrics = {}
    for k in k_values:
        metrics[f"NDCG@{k}"] = ndcg[f"NDCG@{k}"]
        metrics[f"Recall@{k}"] = recall[f"Recall@{k}"]
    return metrics
