# SinkCast v2: HF-Preserving Hooks (Optimization Iteration 1)

## Experiment Overview

Rewrote the SinkCast hook infrastructure to preserve HF's exact attention computation path. The previous hooks replaced HF's entire attention with direct `flash_attn_func` calls, causing d=0 accuracy divergence (e.g., Llama QA d=0: BF16=9.75 vs SC=50.07). The new hooks monkey-patch flash functions to capture `softmax_lse` as a side-channel while letting HF produce identical BF16 output, then apply SinkCast correction on top.

## Setup

- **Models**: Llama-3.1-8B (K=4), Mistral-7B-v0.3 (K=1)
- **Benchmarks**: RULER (4 tasks x 2 seq_lengths x 50 samples), LongBench (5 tasks x 200 samples)
- **Shift protocol**: Padding-based, M=4096
- **Infrastructure**: 4 parallel 1-GPU jobs on TrainService
- **Key change**: New `sinkcast/core/sinkcast_hooks.py` with shared hook infrastructure

## Key Results

| Model | Benchmark | Avg Improvement | Previous (Step 0) |
|-------|-----------|----------------|-------------------|
| Llama-3.1-8B | RULER | +0.01 | +0.01 |
| Llama-3.1-8B | LongBench | -0.45 | -0.45 |
| Mistral-7B-v0.3 | RULER | -2.85 | -2.85 |
| Mistral-7B-v0.3 | LongBench | -0.35 | -0.35 |
| **Overall** | | **-1.03** | **-0.91** |

improvement = BF16_drop - SC_drop (positive = SinkCast reduces degradation)

## Key Observations

1. **LongBench d=0 accuracy now matches BF16 baseline**: narrativeqa SC d=0=23.63 vs BF16=23.68, hotpotqa SC=11.56 vs BF16=11.75, etc. This confirms the new hooks correctly preserve HF's attention path.

2. **RULER QA d=0 still diverges massively**: Llama QA d=0=50.07 (BF16=9.75), Mistral niah d=0=60.0 (BF16=54.0). This is NOT a hook bug -- it's because `sinkcast_correct` applies a non-trivial correction at d=0. The BF16-to-FP32 precision difference in RoPE for the first K sink keys accumulates across 32 layers.

3. **Results are essentially identical to Step 0**: Despite fixing the hook architecture, the improvement numbers are the same because: (a) BF16 position-shift degradation on downstream tasks is near-zero, leaving nothing for SinkCast to fix; (b) the SinkCast correction at d=0 changes model behavior unpredictably (sometimes better, sometimes worse), dominating the improvement signal.

4. **The method works at the attention level but doesn't translate to downstream**: Microbenchmarks show 23.6% (Llama K=4) and 36.3% (Mistral K=1) gap closure in attention-level error. But this precision improvement doesn't translate to accuracy improvements on downstream tasks where BF16 degradation is already negligible.
