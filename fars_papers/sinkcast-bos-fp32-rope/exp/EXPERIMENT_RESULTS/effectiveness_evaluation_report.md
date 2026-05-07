# Effectiveness Evaluation Report

## Verdict: bad

## Summary

SinkCast (K=1) fails all three success criteria for restoring BF16 RoPE shift-invariance. The core localization premise is refuted: D_logit(0) accounts for only 5-9% of total shift-error across measured key indices, far below the 50% threshold. The microbenchmark gap closure peaks at 36.3% (Mistral K=1) and 23.6% (Llama K=4), well below the 80% target. Downstream, BF16 position-shift degradation is already near zero (<2 points), leaving no room for SinkCast to demonstrate improvement; the overall average improvement is -0.91 points. Two optimization iterations could not resolve these fundamental limitations.

## Experiment Feasibility Check

All experiments ran successfully and produced complete results:

- **BF16 FlashAttention Shift Microbenchmark**: Completed for both Llama-3.1-8B and Mistral-7B-v0.3 across 12 configurations each (4 seq lengths x 3 shift pairs). D_logit(j) values and output-logit drift recorded.
- **FP32 Oracle Shift Microbenchmark**: Completed for both models across 9 configurations (3 seq lengths x 3 shift pairs; T=4096 skipped due to OOM with eager FP32 attention).
- **SinkCast Microbenchmark**: Completed for K=1,4,8,16 on both models. Two optimization iterations applied (bug fixes for causal mask, k_raw indexing, shift protocol, and hook infrastructure).
- **BF16 Downstream Baseline**: Completed on RULER (4 tasks x 2 seq lengths) and LongBench (5 tasks) for both models.
- **SinkCast Downstream**: Completed with optimized hooks (K=4 for Llama, K=1 for Mistral) on both benchmarks for both models. Two optimization iterations applied.

There are no missing results. Both main experiment and all baselines produced data for comparison. The verdict is not "failed".

## Results Analysis

### Criterion 1: Localization Premise (D_logit(0) >= 50% of total)

The localization premise posits that the attention-sink key (j=0) dominates the BF16 shift-error, justifying a K=1 correction. We measure the fraction `D_logit(0) / sum(D_logit(j))` for j in {0, 1, 2, 8, 64}.

**Llama-3.1-8B: j0_fraction across all configurations**

| seq_len | shift | D_logit(0) | D_logit(1) | D_logit(2) | D_logit(8) | D_logit(64) | Sum | j0_fraction |
|---------|-------|------------|------------|------------|------------|-------------|-----|-------------|
| 512 | (0,16) | 3.57 | 17.25 | 16.20 | 17.94 | 19.06 | 74.02 | 4.8% |
| 512 | (0,256) | 3.76 | 17.53 | 17.12 | 16.77 | 17.81 | 72.99 | 5.1% |
| 512 | (0,4096) | 4.01 | 18.33 | 19.96 | 20.87 | 19.79 | 82.96 | 4.8% |
| 1024 | (0,16) | 3.51 | 18.39 | 16.98 | 15.28 | 17.11 | 71.27 | 4.9% |
| 1024 | (0,256) | 3.64 | 19.15 | 17.45 | 16.74 | 19.12 | 76.10 | 4.8% |
| 1024 | (0,4096) | 4.08 | 22.50 | 19.44 | 17.21 | 20.02 | 83.25 | 4.9% |
| 2048 | (0,16) | 3.76 | 19.52 | 18.41 | 17.13 | 20.03 | 78.85 | 4.8% |
| 2048 | (0,256) | 3.97 | 21.01 | 19.16 | 17.63 | 21.54 | 83.31 | 4.8% |
| 2048 | (0,4096) | 4.48 | 22.57 | 20.49 | 19.45 | 22.65 | 89.64 | 5.0% |
| 4096 | (0,16) | 3.83 | 20.07 | 18.91 | 18.36 | 22.25 | 83.42 | 4.6% |
| 4096 | (0,256) | 4.05 | 19.94 | 19.67 | 18.55 | 20.79 | 83.00 | 4.9% |
| 4096 | (0,4096) | 4.56 | 23.41 | 21.87 | 20.32 | 23.92 | 94.08 | 4.8% |

**Llama average j0_fraction: 4.9%**

**Mistral-7B-v0.3: j0_fraction across all configurations**

| seq_len | shift | D_logit(0) | D_logit(1) | D_logit(2) | D_logit(8) | D_logit(64) | Sum | j0_fraction |
|---------|-------|------------|------------|------------|------------|-------------|-----|-------------|
| 512 | (0,16) | 2.93 | 8.13 | 8.05 | 9.10 | 8.75 | 36.96 | 7.9% |
| 512 | (0,256) | 3.30 | 8.16 | 8.00 | 9.94 | 9.47 | 38.87 | 8.5% |
| 512 | (0,4096) | 3.35 | 8.27 | 8.34 | 9.39 | 9.17 | 38.52 | 8.7% |
| 1024 | (0,16) | 2.82 | 8.06 | 7.67 | 8.25 | 7.92 | 34.72 | 8.1% |
| 1024 | (0,256) | 3.02 | 8.12 | 7.75 | 8.90 | 9.52 | 37.31 | 8.1% |
| 1024 | (0,4096) | 3.34 | 8.58 | 8.94 | 9.16 | 8.78 | 38.80 | 8.6% |
| 2048 | (0,16) | 3.31 | 9.31 | 9.42 | 10.74 | 9.47 | 42.25 | 7.8% |
| 2048 | (0,256) | 3.42 | 9.84 | 9.59 | 10.21 | 10.01 | 43.07 | 8.0% |
| 2048 | (0,4096) | 3.84 | 10.14 | 10.39 | 10.72 | 9.78 | 44.87 | 8.5% |
| 4096 | (0,16) | 3.43 | 10.20 | 9.89 | 10.49 | 9.55 | 43.56 | 7.9% |
| 4096 | (0,256) | 3.70 | 10.18 | 10.41 | 10.79 | 10.79 | 45.87 | 8.1% |
| 4096 | (0,4096) | 4.08 | 10.86 | 11.37 | 11.89 | 10.57 | 48.77 | 8.4% |

**Mistral average j0_fraction: 8.2%**

**Decision**: The localization premise is **refuted** on both models. D_logit(0) accounts for only 4.9% (Llama) and 8.2% (Mistral) of total D_logit across the 5 measured key indices. D_logit(0) is consistently the *smallest* among the sampled keys, typically 3-5x smaller than D_logit for other indices. The BF16 shift-error is broadly distributed across all key positions, not concentrated at the attention sink.

Note: The BF16 microbenchmark report observes that Wang et al.'s finding about j=0 dominance pertains to *attention weight* (post-softmax) differences rather than *pre-softmax logit* differences. However, the SinkCast correction operates on pre-softmax logits, so the pre-softmax metric is the appropriate one for evaluating the K=1 premise.

### Criterion 2: Microbenchmark Gap Closure (>= 80%)

Gap closure is computed as `(drift_bf16 - drift_sinkcast) / (drift_bf16 - drift_fp32)` for configurations where FP32 data exists (T <= 2048).

**Llama-3.1-8B Gap Closure (best K=4)**

| Config | BF16 max_drift | SC K=4 max_drift | FP32 max_drift | Gap Closure |
|--------|---------------|------------------|----------------|-------------|
| s=512, d=(0,16) | 1.114 | 1.281 | 0.000401 | -15.0% |
| s=512, d=(0,256) | 1.625 | 1.125 | 0.000372 | 30.8% |
| s=512, d=(0,4096) | 1.438 | 1.094 | 0.000940 | 23.9% |
| s=1024, d=(0,16) | 1.566 | 1.125 | 0.001568 | 28.2% |
| s=1024, d=(0,256) | 1.769 | 1.406 | 0.001013 | 20.5% |
| s=1024, d=(0,4096) | 1.758 | 1.313 | 0.003830 | 25.4% |
| s=2048, d=(0,16) | 5.359 | 3.195 | 0.004976 | 40.4% |
| s=2048, d=(0,256) | 6.125 | 3.984 | 0.003269 | 35.0% |
| s=2048, d=(0,4096) | 5.250 | 4.031 | 0.011818 | 23.3% |

**Average gap closure (Llama K=4): 23.6%**

At K=1, Llama achieves only 10.2% average gap closure.

**Mistral-7B-v0.3 Gap Closure (best K=1)**

| Config | BF16 max_drift | SC K=1 max_drift | FP32 max_drift | Gap Closure |
|--------|---------------|------------------|----------------|-------------|
| s=512, d=(0,16) | 0.316 | 0.250 | 0.000120 | 21.0% |
| s=512, d=(0,256) | 0.656 | 0.375 | 0.000097 | 42.9% |
| s=512, d=(0,4096) | 0.844 | 0.375 | 0.000656 | 55.6% |
| s=1024, d=(0,16) | 1.281 | 0.469 | 0.000391 | 63.4% |
| s=1024, d=(0,256) | 0.625 | 0.422 | 0.000237 | 32.5% |
| s=1024, d=(0,4096) | 1.625 | 0.938 | 0.000687 | 42.3% |
| s=2048, d=(0,16) | 6.375 | 3.293 | 0.001244 | 48.4% |
| s=2048, d=(0,256) | 5.625 | 4.527 | 0.001394 | 19.5% |
| s=2048, d=(0,4096) | 3.891 | 3.840 | 0.007902 | 1.3% |

**Average gap closure (Mistral K=1): 36.3%**

**K sensitivity analysis** (avg max_drift gap closure, seq 512-2048):

| Model | K=1 | K=4 | K=8 | K=16 |
|-------|-----|-----|-----|------|
| Llama-3.1-8B | 10.2% | 23.6% | 15.4% | -19.6% |
| Mistral-7B-v0.3 | 36.3% | 34.8% | 32.0% | 29.8% |

**Decision**: The 80% gap closure criterion is **not met** on either model. The best configuration (Mistral K=1) achieves 36.3%, which is less than half the target. Llama at K=4 achieves only 23.6%. Higher K values do not monotonically improve results (Llama K=16 has negative gap closure), indicating a fundamental ceiling caused by FlashAttention's tiled BF16 computation differing from single-pass FP32 einsum.

The optimization report notes that even correcting all keys (K=S) in a single layer achieves only ~26.5% gap closure, confirming that the SinkCast correction approach has an inherent ceiling well below 80%.

### Criterion 3: Downstream Accuracy Improvement (>= 2 absolute points)

BF16 position-shift drops and SinkCast improvements on downstream benchmarks:

**Llama-3.1-8B RULER (K=4)**

| Task | Seq Len | BF16 Drop | SC Drop | Improvement |
|------|---------|-----------|---------|-------------|
| niah_single | 4096 | 0.0 | 0.0 | 0.0 |
| niah_single | 8192 | 0.0 | 0.0 | 0.0 |
| niah_multikey | 4096 | 0.0 | 0.0 | 0.0 |
| niah_multikey | 8192 | 0.0 | 0.0 | 0.0 |
| variable_tracing | 4096 | 0.0 | 0.0 | 0.0 |
| variable_tracing | 8192 | 0.0 | 0.0 | 0.0 |
| qa | 4096 | -0.88 | 1.11 | -1.99 |
| qa | 8192 | 1.67 | -0.42 | **+2.09** |

The +2.09 on Llama RULER QA 8192 superficially meets the criterion. However, this result is **unreliable** because SinkCast changes the d=0 baseline accuracy from 11.59 to 46.34 (a 34.75-point shift). The "improvement" comes from comparing drops against entirely different d=0 baselines, making the comparison meaningless.

**Llama-3.1-8B LongBench (K=4)**

| Task | BF16 Drop | SC Drop | Improvement |
|------|-----------|---------|-------------|
| narrativeqa | -0.86 | -0.52 | -0.34 |
| hotpotqa | -0.18 | 0.13 | -0.31 |
| gov_report | -0.39 | -0.30 | -0.09 |
| trec | 0.0 | 1.0 | -1.0 |
| passage_retrieval_en | -0.50 | 0.0 | -0.50 |
| **Average** | **-0.39** | **0.06** | **-0.45** |

**Mistral-7B-v0.3 RULER (K=1)**

| Task | Seq Len | BF16 Drop | SC Drop | Improvement |
|------|---------|-----------|---------|-------------|
| niah_single | 4096 | 2.0 | 4.0 | -2.0 |
| niah_single | 8192 | -2.0 | 2.0 | -4.0 |
| niah_multikey | 4096 | -4.0 | 4.0 | -8.0 |
| niah_multikey | 8192 | 0.0 | 2.0 | -2.0 |
| variable_tracing | 4096 | 0.0 | 2.0 | -2.0 |
| variable_tracing | 8192 | 0.0 | 0.0 | 0.0 |
| qa | 4096 | -1.64 | -0.80 | -0.84 |
| qa | 8192 | -2.39 | 1.60 | -3.99 |
| **Average** | **-1.00** | **1.85** | **-2.85** |

SinkCast *increases* shift-induced degradation on Mistral RULER by 2.85 points on average.

**Mistral-7B-v0.3 LongBench (K=1)**

| Task | BF16 Drop | SC Drop | Improvement |
|------|-----------|---------|-------------|
| narrativeqa | 0.09 | 0.61 | -0.52 |
| hotpotqa | 0.31 | 0.28 | 0.03 |
| gov_report | 0.23 | -0.01 | 0.24 |
| trec | -0.50 | 1.0 | -1.50 |
| passage_retrieval_en | 0.0 | 0.0 | 0.0 |
| **Average** | **0.03** | **0.38** | **-0.35** |

**Summary of downstream improvements across all model-benchmark pairs:**

| Model | Benchmark | K | Avg BF16 Drop | Avg SC Drop | Avg Improvement |
|-------|-----------|---|---------------|-------------|-----------------|
| Llama-3.1-8B | RULER | 4 | 0.10 | 0.09 | +0.01 |
| Llama-3.1-8B | LongBench | 4 | -0.39 | 0.06 | -0.45 |
| Mistral-7B-v0.3 | RULER | 1 | -1.00 | 1.85 | -2.85 |
| Mistral-7B-v0.3 | LongBench | 1 | 0.03 | 0.38 | -0.35 |
| **Overall** | | | | | **-0.91** |

**Decision**: The downstream criterion is **not met**. The only configuration with >= 2-point improvement (Llama RULER QA 8192: +2.09) is confounded by SinkCast altering the d=0 baseline from 11.59 to 46.34. Excluding this confounded result, no combination achieves >= 2-point improvement. The overall average improvement is -0.91 (SinkCast slightly worsens accuracy). The fundamental issue is that BF16 position-shift degradation on downstream tasks is already near zero, leaving no measurable signal for SinkCast to improve.

## Statistical Significance

Formal statistical tests are not applicable in the standard sense because:

1. **Localization premise**: The j0_fraction values (4.6-5.1% for Llama, 7.8-8.7% for Mistral) are so far from the 50% threshold that the conclusion is unambiguous without statistical testing.

2. **Microbenchmark gap closure**: The gap closure values (10-36%) are deterministic single-run measurements (same seed, same model, same input). The gap to the 80% threshold is large enough that measurement variance cannot explain it.

3. **Downstream accuracy**: The BF16 shift-induced drops are themselves near zero (avg |drop| < 1 point for most model-benchmark pairs), making it impossible for SinkCast to demonstrate a 2-point improvement. The signal-to-noise ratio is fundamentally unfavorable.

4. **Effect direction**: On Mistral RULER, SinkCast consistently *increases* shift-sensitivity (avg improvement = -2.85), which is a negative effect in the wrong direction.

## Verdict Justification

**Verdict: bad**

All three success criteria are not met, and the root causes are fundamental rather than fixable through further optimization:

1. **Localization premise refuted**: The BF16 shift-error measured by D_logit(j) is broadly distributed across all key positions. D_logit(0) accounts for only 5% (Llama) and 8% (Mistral) of the total error across the 5 sampled indices. This means correcting only the sink key (K=1) can address at most ~5-8% of the error. This is a fundamental invalidation of the SinkCast premise that error is concentrated at j=0.

2. **Microbenchmark ceiling well below target**: Even at the best K value, gap closure reaches only 23.6% (Llama K=4) and 36.3% (Mistral K=1). The optimization report reveals that correcting *all* keys (K=S) in a single layer achieves only ~26.5% gap closure, indicating an inherent ceiling. The remaining gap is caused by FlashAttention's tiled BF16 accumulation differing from single-pass FP32 einsum -- a source of error that SinkCast's per-key correction cannot address.

3. **No downstream benefit**: BF16 position-shift degradation on downstream benchmarks is already negligible (avg |drop| < 1 point). SinkCast cannot improve what is already near-optimal. Worse, SinkCast's FP32 correction pathway introduces its own behavioral changes (especially on RULER tasks where d=0 accuracy shifts dramatically), confounding any comparison.

4. **Two optimization iterations exhausted**: Both optimization rounds addressed real bugs (shift protocol mismatch, hook infrastructure replacing HF attention path) but could not change the fundamental outcome. The optimized results (-0.91 overall improvement) are slightly worse than the initial results (-0.78).

5. **The approach is not ready for deployment**: SinkCast (K=1) does not reliably restore BF16 shift-invariance on either model. Extensions to larger K help marginally on the microbenchmark but do not translate to downstream improvements. The fundamental issue is that BF16 precision error in attention is distributed across all keys, not concentrated at the sink, making a sparse correction strategy insufficient. A viable solution would likely need to address the full attention computation (e.g., mixed-precision attention kernels) rather than post-hoc correction of individual key logits.
