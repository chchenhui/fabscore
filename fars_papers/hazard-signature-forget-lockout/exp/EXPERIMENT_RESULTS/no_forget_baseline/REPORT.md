# No-Forget Baseline Experiment Report

## Experiment Overview
Establishes the reference retrieval metrics when the full poisoned memory store (100 benign + 10 poisoned seeds from the MemoryGraft corpus) is used without any forget operation. This quantifies how much poisoned content dominates retrieval under normal operation.

## Setup
- **Corpus**: MemoryGraft seed corpus -- 100 benign (`exp_benign_001`..`exp_benign_100`) + 10 poisoned (`exp_poison_001`..`exp_poison_010`) experience records
- **Source**: https://github.com/Jacobhhy/Agent-Memory-Poisoning
- **Embedding model**: BAAI/bge-m3 (1024-dim dense vectors via FlagEmbedding)
- **Index**: FAISS IndexFlatIP with L2 normalization (cosine similarity)
- **Retrieval**: top-k=3
- **Evaluation queries**: 12 probe queries from `exp4_rag_vector_drift.py`

## Key Results

| Metric | Value |
|--------|-------|
| PRP@3 | **0.6667** |
| Benign Recall@3 (full 110-item index) | **1.0000** |
| Benign Recall@3 (benign-only 100-item index) | **1.0000** |
| Queries with poisoned in top-3 | **12/12** (100%) |

## Key Observations
1. **Severe poisoned retrieval dominance**: PRP@3 = 0.6667 means 24 out of 36 total retrieval slots (12 queries x 3 results) contain poisoned content. Every single evaluation query retrieves at least one poisoned record.
2. **Benign recall unaffected**: Despite poisoned records dominating eval query retrieval, benign self-recall remains perfect (1.0). Each benign record's `req` field retrieves itself in top-3 even with poisoned records in the index. This is because the poisoned records target different semantic intents than benign records.
3. **Sanity check passed**: The benign-only index also achieves perfect self-recall, confirming the retriever and index are correctly configured.
4. **Comparison to MemoryGraft paper**: The paper reports PRP=47.9% under BM25+embedding union retrieval. Our bge-m3 dense-only retrieval yields higher PRP@3=66.67%, likely because dense cosine similarity is more susceptible to the semantic targeting of the poisoned seeds.
5. **High-confidence poisoned matches**: Top poisoned results often have cosine scores > 0.65 (e.g., poison_006 retrieves at 0.747 for "share model metrics"), indicating strong semantic alignment between eval queries and poisoned seeds.
