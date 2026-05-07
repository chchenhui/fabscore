# Condition B: Retrieval-Time HS Denylist with Backfill

## Experiment Overview

Condition B evaluates a retrieval-time defense: after ID-deletion and paraphrase re-injection (same 150-record store as Condition A), apply hazard-signature denylisting at query time. Retrieve top-(k+delta) = top-9 candidates, drop any whose HS matches a tombstoned signature, then return the top-k=3 remaining results.

This tests whether read-time filtering is sufficient to suppress poisoned content without commit-time enforcement.

## Setup

- **Index**: 150 records (100 benign + 50 paraphrased-poisoned), same as Condition A
- **Tombstone set**: HS values from 10 original poisoned records (9 unique HS strings)
- **Retrieval**: k=3, delta=6, fetch top-9, filter, return up to 3
- **Embedding model**: BAAI/bge-m3 (1024-dim, cosine similarity via FAISS IndexFlatIP)
- **HS extraction model**: DeepSeek-V3.2 via LEMMA MaaS proxy
- **Evaluation**: 12 adversarial eval queries, 100 benign self-queries

## Key Results

| Metric | Value |
|--------|-------|
| PRP@3 | 0.9167 |
| Benign Recall@3 | 1.0 |
| WriteBlockRate | 0% |
| Benign FalseBlock | 0% |
| Queries with poisoned in top-3 | 12/12 |
| Total eval candidates filtered | 21/108 |
| Total eval results returned | 35/36 |

## Key Observations

1. **Retrieval-time denylist is largely ineffective**: PRP@3 dropped only marginally from 0.9444 (Condition A) to 0.9167, removing just 1 poisoned slot out of 36 total.

2. **Root cause -- low HS-stability**: HS-stability is 30% (15/50 paraphrases share exact HS with originals). Most paraphrased poisoned records have slightly different HS strings (e.g., an extra hazard category added or removed by the LLM classifier), so they pass the denylist filter.

3. **All 12 queries still contaminated**: Every eval query returns at least one poisoned record in its top-3 results.

4. **No impact on benign utility**: Benign Recall@3 remains 1.0. The backfill buffer (delta=6) ensures enough candidates pass the filter for benign queries. 28 benign-query candidates were filtered (false positives of the denylist matching benign record HS), but the backfill compensated.

5. **Comparison with Condition A**: The 0.0277 PRP@3 reduction is negligible -- retrieval-time filtering cannot compensate for the fundamental instability of HS under paraphrase when matching is exact-string.
