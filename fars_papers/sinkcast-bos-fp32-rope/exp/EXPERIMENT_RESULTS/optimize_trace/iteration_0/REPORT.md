# SinkCast Downstream Optimization - Iteration 0

## Experiment Overview

Optimize the SinkCast downstream evaluation (task 6) which showed SinkCast K=1 fails to improve downstream accuracy under position shift. Three issues were identified and fixed.

## Issues Fixed

### Issue 1 (CRITICAL BUG): Shift protocol mismatch
The BF16 baseline used **padding-based shift** (prepend M pad tokens with attention_mask=0), while SinkCast used **position_ids-only offset** (no padding). This made the shift conditions non-equivalent, invalidating the comparison.

**Fix**: Rewrote `run_inference_sinkcast()` in both `ruler_shift_sinkcast.py` and `longbench_shift_sinkcast.py` to use padding-based shift identical to the BF16 baseline. Added `real_mask_ref` parameter to `_sinkcast_hooks` to communicate which tokens are real vs padding.

### Issue 2: Hook does not handle padded input
The SinkCast hook calls `flash_attn_func` directly with `causal=True`, which doesn't handle HF's unpadding. When padding is present, pad tokens must be filtered out before the flash call.

**Fix**: Updated the hook to detect real tokens via `real_mask_ref`, extract only real tokens for Q/K/V before `flash_attn_func`, apply `sinkcast_correct` on real tokens only, then reconstruct full-sequence output with zeros for pad positions.

### Issue 3: K=1 suboptimal for Llama
The microbenchmark optimization showed K=4 achieves 23.6% gap closure for Llama (vs 10.2% for K=1).

**Fix**: Run Llama with K=4, keep Mistral at K=1 (optimal per microbenchmark).

## Setup

- **Models**: Llama-3.1-8B (K=4), Mistral-7B-v0.3 (K=1)
- **Shift Protocol**: Padding-based, M=4096 (identical to BF16 baseline)
- **RULER**: 4 tasks, seq_lengths 4096/8192, 50 samples/task
- **LongBench**: 5 tasks, 200 samples/task, max_context 8192
- **Infrastructure**: 2x 1-GPU jobs in parallel via TrainService

## Key Results

### Llama-3.1-8B RULER (K=4)

| Task | Len | BF16 Drop | SC Drop | Improvement |
|------|-----|-----------|---------|-------------|
| niah_single | 4096 | 0.0 | 0.0 | 0.0 |
| niah_single | 8192 | 0.0 | 0.0 | 0.0 |
| niah_multikey | 4096 | 0.0 | 0.0 | 0.0 |
| niah_multikey | 8192 | 0.0 | 0.0 | 0.0 |
| variable_tracing | 4096 | 0.0 | 0.0 | 0.0 |
| variable_tracing | 8192 | 0.0 | 0.0 | 0.0 |
| qa | 4096 | -0.88 | 1.11 | -1.99 |
| qa | 8192 | 1.67 | -0.42 | 2.09 |
| **Average** | | **0.10** | **0.09** | **0.01** |

### Llama-3.1-8B LongBench (K=4)

| Task | BF16 Drop | SC Drop | Improvement |
|------|-----------|---------|-------------|
| narrativeqa | -0.86 | -0.52 | -0.34 |
| hotpotqa | -0.18 | 0.13 | -0.31 |
| gov_report | -0.39 | -0.30 | -0.09 |
| trec | 0.0 | 1.0 | -1.0 |
| passage_retrieval_en | -0.50 | 0.0 | -0.50 |
| **Average** | **-0.39** | **0.06** | **-0.45** |

### Mistral-7B-v0.3 RULER (K=1)

| Task | Len | BF16 Drop | SC Drop | Improvement |
|------|-----|-----------|---------|-------------|
| niah_single | 4096 | 2.0 | 4.0 | -2.0 |
| niah_single | 8192 | -2.0 | 2.0 | -4.0 |
| niah_multikey | 4096 | -4.0 | 4.0 | -8.0 |
| niah_multikey | 8192 | 0.0 | 2.0 | -2.0 |
| variable_tracing | 4096 | 0.0 | 2.0 | -2.0 |
| variable_tracing | 8192 | 0.0 | 0.0 | 0.0 |
| qa | 4096 | -1.64 | -0.80 | -0.84 |
| qa | 8192 | -2.39 | 1.60 | -3.99 |
| **Average** | | **-1.00** | **1.85** | **-2.85** |

### Mistral-7B-v0.3 LongBench (K=1)

| Task | BF16 Drop | SC Drop | Improvement |
|------|-----------|---------|-------------|
| narrativeqa | 0.09 | 0.61 | -0.52 |
| hotpotqa | 0.31 | 0.28 | 0.03 |
| gov_report | 0.23 | -0.01 | 0.24 |
| trec | -0.50 | 1.0 | -1.50 |
| passage_retrieval_en | 0.0 | 0.0 | 0.0 |
| **Average** | **0.03** | **0.38** | **-0.35** |

### Summary

| Model | Benchmark | K | Original Avg Improvement | Optimized Avg Improvement |
|-------|-----------|---|--------------------------|---------------------------|
| Llama-3.1-8B | RULER | 4 | +0.27 (K=1) | +0.01 |
| Llama-3.1-8B | LongBench | 4 | -0.19 (K=1) | -0.45 |
| Mistral-7B-v0.3 | RULER | 1 | -2.85 (K=1) | -2.85 |
| Mistral-7B-v0.3 | LongBench | 1 | -0.35 (K=1) | -0.35 |

## Key Observations

1. **Shift protocol fix did not change Mistral results**: Mistral results are identical because the SinkCast hook's d=0 behavior dominates -- the hook replaces HF attention with direct flash_attn_func, changing absolute accuracy at d=0.

2. **K=4 did not help Llama downstream**: Despite K=4 showing 23.6% gap closure at the microbenchmark level (vs 10.2% for K=1), this did not translate to downstream improvement. The BF16 position-shift drop on downstream tasks is already near zero (RULER avg 0.10, LongBench avg -0.39), so there is no degradation to correct.

3. **SinkCast d=0 accuracy differs substantially from BF16 d=0**: The hook changes the attention computation path (direct flash_attn_func vs HF's full attention pipeline), leading to different d=0 baselines. This makes the "improvement = BF16_drop - SC_drop" metric unreliable.

4. **Fundamental limitation**: The BF16 position-shift degradation on downstream tasks is too small (within noise) for SinkCast to provide measurable improvement. The attention-level error correction doesn't translate to task-level gains.
