# Backfill Buffer Ablation Study

## Experiment Overview

This ablation isolates the contribution of the backfill buffer in the retrieval-time denylist baseline (Condition B). By comparing B(delta=6) vs B(delta=0), we measure how much backfill mitigates "slot wasting" — the phenomenon where retrieval slots go empty because tombstoned items are dropped without replacement. We also compare against Condition C (commit-time lockout), which avoids slot wasting entirely.

## Setup

- **Index**: 150 records (100 benign + 50 paraphrased-poisoned), same as Condition A post-reinjection store
- **Tombstone set**: 10 original poisoned records' hazard signatures (9 unique HS values, 6 unique hazard sets)
- **Matching**: Fuzzy hazard-set containment (subset/superset matching)
- **Evaluation**: 12 adversarial eval queries (PRP@3) + 100 benign self-queries (Benign Recall@3)
- **B(delta=6)**: Fetches top-9 candidates, filters, returns up to 3
- **B(delta=0)**: Fetches top-3 candidates, filters, returns 0-3 (no replacement)
- **C**: Clean index (100 benign only), always returns 3

## Key Results

| Variant | PRP@3 | Benign Recall@3 | Avg results/eval query | Avg results/benign query |
|---------|-------|-----------------|----------------------|------------------------|
| B (delta=6) | 0.0000 | 0.97 | 1.00 / 3 | 2.87 / 3 |
| B (delta=0) | 0.0000 | 0.97 | 0.17 / 3 | 2.42 / 3 |
| C (commit-time) | 0.0000 | 1.00 | 3.00 / 3 | 3.00 / 3 |

## Key Observations

### 1. Slot Wasting is Severe Without Backfill
B(delta=0) returns only 2/36 eval query results (0.17 avg per query, 94% slot waste). For benign queries, 242/300 results returned (2.42 avg, 19% waste). The backfill buffer substantially helps: B(delta=6) recovers to 12/36 eval results (1.0 avg) and 287/300 benign results (2.87 avg).

### 2. Benign Recall@3 is Unchanged by Backfill
Both B variants achieve identical Benign Recall@3 = 0.97. This is because the recall metric only checks if the ground-truth benign record appears in the result set. Since benign records typically rank #1 for their own self-queries and are not tombstoned, they survive filtering regardless of backfill. The metric masks the true utility degradation.

### 3. Backfill Helps But Cannot Close the Gap to Condition C
Even with delta=6, Condition B returns only 33% of possible eval query results (1.0/3.0) vs Condition C's 100% (3.0/3.0). The fundamental issue is index pollution: when the index contains 50 paraphrased-poisoned records, many of the top candidates for adversarial queries are poisoned. The fuzzy denylist correctly filters them, but there are not enough clean candidates in the top-(k+delta) to fill all slots.

### 4. Condition C Eliminates Slot Wasting Entirely
By blocking poisoned content at commit time, Condition C maintains a clean 100-record index. Every query returns exactly k=3 results. There is no filtering at retrieval time, so no slots are wasted. This is a structural advantage of commit-time prevention over retrieval-time filtering.

### 5. The Backfill Buffer Addresses Symptoms, Not Root Cause
The backfill buffer is a useful mitigation for retrieval-time filtering, improving eval results by 6x (0.17 to 1.0 avg). However, it treats the symptom (empty slots) rather than the root cause (index pollution). Commit-time lockout (Condition C) eliminates the root cause and achieves perfect retrieval utility with no tradeoffs in safety (PRP@3 = 0.0).
