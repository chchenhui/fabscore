# SinkCast Shift Microbenchmark (Optimized)

## Experiment Overview

Evaluated the SinkCast correction algorithm on the shift microbenchmark for two models: Llama-3.1-8B and Mistral-7B-v0.3. SinkCast keeps BF16 FlashAttention as the fast path, then recomputes sink-key logits in FP32 and applies a closed-form output correction using FlashAttention's row-wise `softmax_lse`.

After optimization (iteration 0), two critical bugs were fixed:
1. Missing causal mask in the correction formula for K>1 (caused catastrophic errors)
2. k_raw indexing bug that always used index 0 for all sink keys

Additionally, the implementation was extended from K=1 to support batch multi-key correction for arbitrary K.

## Setup

- **Models**: meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-v0.3
- **Method**: BF16 FlashAttention-2 + SinkCast correction (FP32 logit recomputation, batch multi-key)
- **K values tested**: 1, 4, 8, 16
- **Sequence lengths**: 512, 1024, 2048, 4096
- **Shift pairs**: (0,16), (0,256), (0,4096)
- **GPU**: 1x NVIDIA A100 per model

## Key Results

### Best Configuration Per Model

- **Llama-3.1-8B**: K=4 with **23.6% avg max_drift gap closure** (up from 10.2% at K=1)
- **Mistral-7B-v0.3**: K=1 with **36.3% avg max_drift gap closure** (unchanged)

### Gap Closure Summary (avg max_drift, seq 512-2048 where FP32 oracle available)

| Model | K=1 | K=4 | K=8 | K=16 |
|-------|-----|-----|-----|------|
| Llama-3.1-8B | 10.2% | **23.6%** | 15.4% | -19.6% |
| Mistral-7B-v0.3 | **36.3%** | 34.8% | 32.0% | 29.8% |

### Llama-3.1-8B (Best: K=4)

| Config | BF16 max_drift | SC K=4 max_drift | FP32 max_drift | Gap Closure (max) |
|--------|---------------|------------------|----------------|-------------------|
| s=512, d=(0,16) | 1.1143 | 1.2812 | 0.0004 | -0.1499 |
| s=512, d=(0,256) | 1.6250 | 1.1250 | 0.0004 | 0.3078 |
| s=512, d=(0,4096) | 1.4375 | 1.0938 | 0.0009 | 0.2393 |
| s=1024, d=(0,16) | 1.5657 | 1.1250 | 0.0016 | 0.2817 |
| s=1024, d=(0,256) | 1.7688 | 1.4062 | 0.0010 | 0.2051 |
| s=1024, d=(0,4096) | 1.7578 | 1.3125 | 0.0038 | 0.2539 |
| s=2048, d=(0,16) | 5.3594 | 3.1953 | 0.0050 | 0.4042 |
| s=2048, d=(0,256) | 6.1250 | 3.9844 | 0.0033 | 0.3497 |
| s=2048, d=(0,4096) | 5.2500 | 4.0312 | 0.0118 | 0.2327 |

**Average max_drift gap closure: 0.2360 (23.6%)**

### Mistral-7B-v0.3 (Best: K=1)

| Config | BF16 max_drift | SC K=1 max_drift | FP32 max_drift | Gap Closure (max) |
|--------|---------------|------------------|----------------|-------------------|
| s=512, d=(0,16) | 0.3164 | 0.2500 | 0.0001 | 0.2100 |
| s=512, d=(0,256) | 0.6562 | 0.3750 | 0.0001 | 0.4286 |
| s=512, d=(0,4096) | 0.8438 | 0.3750 | 0.0007 | 0.5560 |
| s=1024, d=(0,16) | 1.2812 | 0.4688 | 0.0004 | 0.6343 |
| s=1024, d=(0,256) | 0.6250 | 0.4219 | 0.0002 | 0.3251 |
| s=1024, d=(0,4096) | 1.6250 | 0.9375 | 0.0007 | 0.4233 |
| s=2048, d=(0,16) | 6.3750 | 3.2930 | 0.0012 | 0.4835 |
| s=2048, d=(0,256) | 5.6250 | 4.5273 | 0.0014 | 0.1952 |
| s=2048, d=(0,4096) | 3.8906 | 3.8398 | 0.0079 | 0.0131 |

**Average max_drift gap closure: 0.3632 (36.3%)**

## Key Observations

1. **Bug fixes enable K>1**: The causal mask fix was critical; without it K>1 had catastrophic -1200% gap closure. With fix, K=4 provides 2.3x improvement for Llama.

2. **Model-dependent optimal K**: Llama benefits from K=4, while Mistral is best at K=1. Llama has lower D_logit(0) fraction (~5%), so correcting more keys helps. Mistral already captures more with K=1 (~7-9% fraction).

3. **K>4 degrades for Llama**: Higher K introduces more correction noise from the precision mismatch between FlashAttention's tiled BF16 computation and our FP32 einsum, which compounds across 32 layers.

4. **Fundamental ceiling**: Even at single-layer level with K=S (all keys), gap closure reaches only 26.5%. The remaining gap comes from FlashAttention's tiled computation differing from single-pass FP32 einsum.

5. **Mean drift improvement minimal**: While max_drift shows meaningful improvement, mean_drift remains in the 1-3% range across all configurations. The correction primarily helps with outlier positions.
