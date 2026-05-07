# Optimization Iteration 0: Fuzzy Hazard-Signature Tombstone Matching

## Experiment Overview

Replaced exact HS string matching in the Commit-Time Tombstone (Condition C) with fuzzy hazard-set containment matching. The original method matched the full concatenated string `task_type | target_artifact | hazards` exactly, achieving only 38% write block rate due to LLM classification variance under paraphrase. The optimized method matches on hazard sets using subset/superset containment, which tolerates add/drop of individual hazard tags by the LLM.

## Setup

- **Matching strategy**: Hazard set containment -- a record is blocked if its hazard set is a subset or superset of any tombstoned entry's hazard set
- **Embedding model**: BAAI/bge-m3 (1024-dim, cosine similarity via FAISS IndexFlatIP)
- **HS extraction model**: DeepSeek-V3.2 (same cached signatures as original)
- **Retrieval**: k=3, standard top-k
- **Evaluation**: 12 adversarial eval queries, 100 benign self-queries, 100 benign false-block checks

## Key Results

### Condition C Optimized (Commit-Time Fuzzy Tombstone)

| Metric | Original (Exact) | Optimized (Fuzzy) | Change |
|--------|------------------|-------------------|--------|
| WriteBlockRate | 38% (19/50) | 100% (50/50) | +62pp |
| PRP@3 | 0.9444 | 0.0000 | -0.9444 |
| Benign Recall@3 | 1.0 | 1.0 | 0 |
| Benign FalseBlock | 0% (0/100) | 3% (3/100) | +3pp |
| Queries with poison in top-3 | 12/12 | 0/12 | -12 |
| Final index size | 131 | 100 | -31 |
| Fuzzy HS-stability | 30% (exact) | 92% (containment) | +62pp |

### Condition B Optimized (Retrieval-Time Fuzzy Denylist) -- Comparison

| Metric | B Original | B Optimized | C Optimized |
|--------|-----------|-------------|-------------|
| PRP@3 | 0.9167 | 0.0000 | 0.0000 |
| Benign Recall@3 | 1.0 | 0.97 | 1.0 |
| WriteBlockRate | 0% | 0% | 100% |
| Benign FalseBlock | 0% | 0% | 3% |

### Full Comparison Table (All Conditions)

| Condition | PRP@3 | Benign Recall@3 | WriteBlockRate | Benign FalseBlock |
|-----------|-------|-----------------|----------------|-------------------|
| No-Forget (110 seeds) | 0.6667 | 1.0 | N/A | N/A |
| A: ID-Delete Only | 0.9444 | 1.0 | 0% | 0% |
| B: Retrieval-Time Denylist (exact) | 0.9167 | 1.0 | 0% | 0% |
| B: Retrieval-Time Denylist (fuzzy) | 0.0000 | 0.97 | 0% | 0% |
| C: Commit-Time Tombstone (exact) | 0.9444 | 1.0 | 38% | 0% |
| **C: Commit-Time Tombstone (fuzzy)** | **0.0000** | **1.0** | **100%** | **3%** |

## Key Observations

1. **Dramatic improvement**: PRP@3 dropped from 0.9444 to 0.0000 -- all poisoned content is now completely blocked from entering the memory store, achieving perfect defense against paraphrase re-injection.

2. **100% write block rate**: All 50 paraphrased poisoned records are blocked at commit time. The hazard set containment matching tolerates the LLM's tendency to add or drop individual hazard tags during HS extraction.

3. **Minimal benign false positives**: Only 3/100 benign records (3%) would be falsely blocked. These are records with a single hazard tag (skip_validation or remote_exec) that happens to be a subset of a tombstoned entry's hazard set. The 3% rate is well below the 10% acceptability threshold.

4. **Condition C superior to B**: With fuzzy matching, Condition C achieves Benign Recall@3=1.0 vs B's 0.97. This is because commit-time blocking keeps the index clean (100 benign records only), while retrieval-time filtering must still fetch from a polluted 150-record index and wastes some retrieval slots on filtering.

5. **Fuzzy HS-stability = 92%**: The hazard-set containment matching raises effective HS-stability from 30% (exact match) to 92% (set containment), well above the 70% threshold. This confirms the hypothesis that discrete hazard signatures are stable under paraphrase when matching tolerates add/drop variance.
