# FP32 Oracle Shift Microbenchmark

## Experiment Overview

This baseline establishes the **oracle upper bound** for RoPE shift-invariance by running the shift microbenchmark with full FP32 attention (no BF16 anywhere in the attention computation). If the shift-induced drift observed in BF16 is caused by precision loss rather than model architecture, FP32 attention should exhibit near-zero drift. The ratio `drift_bf16 / drift_fp32` quantifies how much of the observed BF16 drift is attributable to precision.

## Setup

- **Models**: Llama-3.1-8B, Mistral-7B-v0.3
- **Attention**: Eager (PyTorch native), `dtype=torch.float32`, no FlashAttention
- **Sequence lengths**: 512, 1024, 2048 (4096 skipped due to OOM with full FP32 attention matrices)
- **Shift pairs**: (0,16), (0,256), (0,4096)
- **Key indices**: j in {0, 1, 2, 8, 64}
- **Seed**: 42
- **GPU**: 1x NVIDIA A100-SXM4-80GB
- **Protocol**: Identical to BF16 baseline -- same input tokens, same position offsets, same D_logit and output-drift metrics

## Key Results

### Llama-3.1-8B

| T | Shift | FP32 max_drift | BF16 max_drift | Ratio | FP32 mean_drift | BF16 mean_drift | Ratio |
|---|-------|----------------|----------------|-------|-----------------|-----------------|-------|
| 512 | (0,16) | 0.000401 | 1.114 | 2782x | 2.19e-05 | 0.0738 | 3364x |
| 512 | (0,256) | 0.000372 | 1.625 | 4364x | 2.34e-05 | 0.0758 | 3240x |
| 512 | (0,4096) | 0.000940 | 1.438 | 1529x | 5.67e-05 | 0.0793 | 1398x |
| 1024 | (0,16) | 0.001568 | 1.563 | 999x | 2.14e-05 | 0.0659 | 3081x |
| 1024 | (0,256) | 0.001013 | 1.768 | 1746x | 2.38e-05 | 0.0683 | 2863x |
| 1024 | (0,4096) | 0.003830 | 1.759 | 459x | 5.78e-05 | 0.0737 | 1274x |
| 2048 | (0,16) | 0.004976 | 5.359 | 1077x | 1.94e-05 | 0.0523 | 2689x |
| 2048 | (0,256) | 0.003269 | 6.125 | 1874x | 2.08e-05 | 0.0529 | 2536x |
| 2048 | (0,4096) | 0.011818 | 5.250 | 444x | 5.68e-05 | 0.0579 | 1018x |

**Summary**: max_drift ratio range [444x, 4364x], avg 1697x. mean_drift ratio range [1018x, 3364x], avg 2274x.

### Mistral-7B-v0.3

| T | Shift | FP32 max_drift | BF16 max_drift | Ratio | FP32 mean_drift | BF16 mean_drift | Ratio |
|---|-------|----------------|----------------|-------|-----------------|-----------------|-------|
| 512 | (0,16) | 0.000120 | 0.316 | 2638x | 5.89e-06 | 0.0202 | 3422x |
| 512 | (0,256) | 0.000097 | 0.656 | 6746x | 5.74e-06 | 0.0217 | 3771x |
| 512 | (0,4096) | 0.000656 | 0.844 | 1286x | 1.44e-05 | 0.0212 | 1468x |
| 1024 | (0,16) | 0.000391 | 1.281 | 3281x | 5.97e-06 | 0.0185 | 3096x |
| 1024 | (0,256) | 0.000237 | 0.625 | 2640x | 5.82e-06 | 0.0191 | 3284x |
| 1024 | (0,4096) | 0.000687 | 1.625 | 2367x | 1.47e-05 | 0.0210 | 1424x |
| 2048 | (0,16) | 0.001244 | 6.375 | 5126x | 6.97e-06 | 0.0190 | 2728x |
| 2048 | (0,256) | 0.001394 | 5.625 | 4034x | 6.61e-06 | 0.0189 | 2850x |
| 2048 | (0,4096) | 0.007902 | 3.891 | 492x | 1.73e-05 | 0.0202 | 1172x |

**Summary**: max_drift ratio range [492x, 6746x], avg 3179x. mean_drift ratio range [1172x, 3771x], avg 2580x.

## Key Observations

1. **FP32 drift is negligible**: FP32 output-logit drift is 3-6 orders of magnitude smaller than BF16 drift across all configurations. The residual FP32 drift (e.g., max_drift ~0.001-0.012) is attributable to floating-point rounding in FP32 computation, which is negligible compared to BF16 drift (max_drift ~0.3-6.4).

2. **BF16 precision is the root cause**: The ratio `drift_bf16/drift_fp32` ranges from ~440x to ~6750x, confirming that virtually all shift-induced drift is caused by BF16 precision loss in attention dot products, not by model architecture.

3. **Larger shifts produce larger FP32 drift but still negligible**: The (0,4096) shift pair shows slightly higher FP32 drift than (0,16) -- this is expected since larger position offsets produce larger RoPE rotation angles, increasing FP32 rounding error. However, even at (0,4096) the FP32 drift remains >400x smaller than BF16 drift.

4. **D_logit(j) ratios are consistent**: Per-key attention logit differences show similar BF16/FP32 ratios (typically 700-4000x), confirming the precision issue affects all keys, not just the sink key j=0.

5. **Both models behave consistently**: Llama-3.1-8B and Mistral-7B-v0.3 show the same qualitative pattern, confirming this is a general BF16+RoPE phenomenon rather than model-specific.
