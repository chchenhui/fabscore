# Condition C: Commit-Time HS Tombstone Lockout (HST Method)

## Experiment Overview

Condition C evaluates the proposed Hazard-Signature Tombstone (HST) method: after deleting 10 poisoned records by ID, their hazard signatures are tombstoned. When 50 paraphrased poisoned records attempt re-injection, each write is checked at commit time -- if the record's hazard set matches a tombstoned signature via set containment, the write is rejected. Retrieval then operates on a clean index (100 benign only).

The method uses **fuzzy hazard-set containment matching**: a record is blocked if its hazard set is a subset or superset of any tombstoned entry's hazard set. This tolerates LLM classification variance under paraphrase (add/drop of individual hazard tags).

## Setup

- **Index**: Starts with 110 records (100 benign + 10 poisoned), poisoned deleted, then 50 paraphrases attempted
- **Tombstone set**: 9 unique HS entries from 10 original poisoned records
- **Policy**: CommitTimeTombstoneFuzzy -- hazard set containment matching
- **Embedding model**: BAAI/bge-m3 (1024-dim, cosine similarity via FAISS IndexFlatIP)
- **HS extraction model**: DeepSeek-V3.2 via LEMMA MaaS proxy
- **Retrieval**: k=3, standard top-k (no read-time filtering needed)
- **Evaluation**: 12 adversarial eval queries, 100 benign self-queries, 100 benign false-block checks

## Key Results

| Metric | Value |
|--------|-------|
| WriteBlockRate | 100.0% (50/50) |
| PRP@3 | 0.0000 |
| Benign Recall@3 | 1.0 |
| Benign FalseBlock | 3.0% (3/100) |
| Queries with poisoned in top-3 | 0/12 |
| Final index size | 100 (100 benign + 0 accepted paraphrases) |
| Fuzzy HS-stability | 92% (46/50) |

### Comparison with Other Conditions

| Condition | PRP@3 | Benign Recall@3 | WriteBlockRate | Benign FalseBlock |
|-----------|-------|-----------------|----------------|-------------------|
| No-Forget (110 seeds) | 0.6667 | 1.0 | N/A | N/A |
| A: ID-Delete Only | 0.9444 | 1.0 | 0% | 0% |
| B: Retrieval-Time Denylist (exact) | 0.9167 | 1.0 | 0% | 0% |
| B: Retrieval-Time Denylist (fuzzy) | 0.0000 | 0.97 | 0% | 0% |
| **C: Commit-Time Tombstone (fuzzy)** | **0.0000** | **1.0** | **100%** | **3%** |

## Key Observations

1. **Complete poison blocking**: All 50 paraphrased poisoned records are blocked at commit time (100% WriteBlockRate). PRP@3 drops to 0.0 -- no poisoned content appears in any retrieval results.

2. **Superior to retrieval-time baseline**: Condition C achieves Benign Recall@3=1.0 vs Condition B's 0.97 with fuzzy matching. Commit-time blocking produces a clean 100-record index, while retrieval-time filtering wastes slots filtering from a polluted 150-record index.

3. **Minimal benign false positives**: 3/100 benign records (3%) would be falsely blocked at write-time. These are records with a single hazard tag that is a subset of tombstoned entries. The 3% FP rate is well within acceptable limits (<10%).

4. **High fuzzy HS-stability**: Set containment matching raises HS-stability from 30% (exact) to 92%, confirming that hazard signatures are stable under paraphrase when matching tolerates add/drop variance in individual hazard tags.

5. **Zero retrieval contamination**: 0/12 evaluation queries surface any poisoned content in top-3, compared to 12/12 in the original exact-match version. The commit-time lockout completely eliminates retrieval contamination.
