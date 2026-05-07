# Optimization Iteration 1 -- Minimal-Swap Debiased Sorting

## Overview

Replaced the original full-sort debiased approach (which sorted ALL documents by debiased scores) with a minimal-swap strategy: sort by raw attention scores (identical to k=1), then swap only the debiased top-1 document to the last position if it differs from the raw top-1. This preserves the k=1 distractor ordering while correcting recency-bias errors in top-1 identification.

## Diagnosis

Thorough per-example analysis of the original debiased k=1 results (600 examples across 3 seeds) revealed:

1. **Debiased ranking is strictly better**: 444/600 gold-at-last vs 425/600 for raw k=1 (19 extra, 0 regressions).
2. **But full re-sorting hurts generation**: 9 of 13 accuracy losses vs k=1 had identical gold-doc position -- the losses were purely from distractor reordering changing the context the model reads.
3. **Recency bias causes raw top-1 errors**: In 21/600 examples, debiased top-1 differs from raw top-1. In 19 of those, debiased correctly identifies the gold doc while raw incorrectly picks the last-position document due to recency bias.
4. **The fix**: Only swap the debiased top-1 to the last position when it differs from raw top-1, preserving >99% of the k=1 ordering.

## Implementation

Changed `debiased_sorting.py` to:
1. Sort all docs by RAW attention scores (ascending, best-last) -- same as k=1
2. Compute debiased scores using existing bias estimation (alpha=0.05, 20 bins, median)
3. If debiased argmax != raw argmax, swap ONLY the debiased top-1 to last position
4. Generate answer from this minimally-modified ordering

## Results

| Configuration | Seed 42 | Seed 123 | Seed 456 | Mean | Std |
|---------------|---------|----------|----------|------|-----|
| **Minimal-swap debiased k=1** | **94.0%** | **96.5%** | **94.0%** | **94.83%** | **1.44%** |
| Original debiased k=1 (full sort) | 94.5% | 95.5% | 93.0% | 94.33% | 1.26% |
| k=1 uncalibrated | 94.0% | 96.5% | 94.0% | 94.83% | 1.44% |
| k=5 iterative | 96.5% | 95.0% | 95.0% | 95.50% | 0.87% |

Swap statistics: 4 swaps (seed 42), 4 swaps (seed 123), 2 swaps (seed 456) = 10 total across 600 examples (1.67%).

## Key Finding

The minimal-swap approach produces **identical output to k=1 uncalibrated** on all 600 examples. In every swap example, both the swapped and un-swapped orderings generated the same correct answer. This means:

1. The swap correctly identifies the gold doc in all cases (debiased top-1 = gold doc in all 10 swaps)
2. But the model is robust enough to find the answer even when the gold doc is not at the very last position (it's still in the top 3-5 of ~165 docs after k=1 sorting)
3. The +0.50pp improvement over the original method comes entirely from **not disrupting the distractor ordering** (eliminating the 9 regression cases from full re-sorting)

## Conclusion

The minimal-swap strategy improves over the original debiased method by +0.50pp (94.83% vs 94.33%), matching k=1 uncalibrated exactly. The improvement comes from eliminating the distractor reordering noise that caused accuracy losses in the original full-sort approach. However, it does not surpass k=1 uncalibrated or close the gap to k=5 (95.50%), because the swap itself doesn't change the model's answer in any of the 10 swap cases.
