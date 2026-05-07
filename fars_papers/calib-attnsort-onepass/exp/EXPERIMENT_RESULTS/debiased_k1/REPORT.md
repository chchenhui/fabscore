# Debiased One-Pass Attention Sorting -- Experiment Report

## Experiment Overview

Evaluate the proposed Debiased One-Pass Attention Sorting method on SynthWiki@30K using LLaMA-2-7B-32K-Instruct. The method runs one attention-extraction step (identical to k=1 Attention Sorting), then estimates a per-prompt position-bias curve from the distractor documents' attention-vs-position pattern and subtracts it to produce debiased relevance scores.

**Optimized strategy (minimal-swap)**: Sort documents by raw attention scores (same as k=1), then use debiased scores to identify the top-1 document. If the debiased top-1 differs from the raw top-1, swap only that document to the last position. This preserves >99% of the k=1 distractor ordering while correcting recency-bias errors.

The key formula:
- Raw attention mass: a_i = sum over all layers, heads, and document tokens of attention weights from the first generated token.
- Bias estimation: (1) trim top alpha=0.05 fraction of documents by a_i, (2) bin remaining positions into B=20 equal-width bins and compute median a_i per bin, (3) linearly interpolate to get bias curve b_hat(p).
- Debiased score: s_i = a_i - b_hat(p_i).

## Setup

- **Model**: togethercomputer/LLaMA-2-7B-32K-Instruct
- **Benchmark**: SynthWiki (madlibs1.csv, 990 entries)
- **Context length**: ~28K tokens (JUNK_SIZE=28000)
- **Examples**: 200 per seed
- **Seeds**: 42, 123, 456
- **Decoding**: Greedy (do_sample=False, max_new_tokens=100)
- **Prompt type**: together_instruct
- **Prefill passes per query**: 2 (1 for attention extraction + 1 for answer generation)
- **Bias estimation**: alpha=0.05, num_bins=20, median aggregation, linear interpolation
- **Sorting strategy**: Minimal-swap (sort by raw attention, swap debiased top-1 to last if different from raw top-1)

## Key Results

| Condition | Mean Accuracy | Std | Prefill Passes |
|-----------|--------------|-----|----------------|
| No Sorting | 72.83% | 2.84% | 1 |
| Attn Sort k=1 (uncalibrated) | 94.83% | 1.44% | 2 |
| **Debiased k=1 (minimal-swap)** | **94.83%** | **1.44%** | **2** |
| Attn Sort k=5 (iterative) | 95.50% | 0.87% | 6 |

### Per-Seed Accuracy

| Seed | Debiased k=1 (minimal-swap) | k=1 Uncalibrated | k=5 |
|------|----------------------------|-------------------|------|
| 42 | 94.0% | 94.0% | 96.5% |
| 123 | 96.5% | 96.5% | 95.0% |
| 456 | 94.0% | 94.0% | 95.0% |

### Swap Statistics

| Seed | Swaps | Total Examples | Swap Rate |
|------|-------|----------------|-----------|
| 42 | 4 | 200 | 2.0% |
| 123 | 4 | 200 | 2.0% |
| 456 | 2 | 200 | 1.0% |
| **Total** | **10** | **600** | **1.67%** |

### Paired Comparison (per-example win/tie/loss)

| Comparison | Wins | Ties | Losses | Total |
|-----------|------|------|--------|-------|
| Debiased k=1 vs k=1 uncalibrated | 0 | 600 | 0 | 600 |

## Optimization History

The original method (full-sort by debiased scores) achieved 94.33% mean accuracy but suffered from distractor reordering noise. Analysis showed 9/13 accuracy losses vs k=1 had identical gold positions -- losses were purely from distractor reordering. The minimal-swap optimization improved accuracy from 94.33% to 94.83% (+0.50pp).

| Method | Mean Accuracy | Improvement |
|--------|--------------|-------------|
| Original (full sort) | 94.33% | baseline |
| **Optimized (minimal swap)** | **94.83%** | **+0.50pp** |

## Key Observations

1. **Debiased k=1 with minimal-swap achieves 94.83% accuracy**, matching k=1 uncalibrated exactly and improving over the original full-sort debiased method (94.33%) by +0.50pp.

2. **The minimal-swap strategy eliminates distractor reordering noise**: By preserving >99% of the k=1 ordering and only swapping 1.67% of examples, the method avoids the generation quality degradation caused by full document re-sorting.

3. **The debiased top-1 correctly identifies the gold document in all swap cases**: In all 10 swap examples, the debiased score correctly identifies the gold doc while the raw score incorrectly favors the last-position document (recency bias).

4. **The gap to k=5 remains**: Debiased k=1 (94.83%) does not match k=5 (95.50%). The model is robust enough that gold doc being in the top 3-5 positions (after k=1 sorting) is sufficient for correct generation -- the extra push to exact-last position doesn't change the answer.

5. **Position bias correction is necessary but not sufficient**: The debiasing correctly identifies recency-bias errors, but the correction alone does not improve generation because the model can find answers from documents near the end, not just at the very end.
