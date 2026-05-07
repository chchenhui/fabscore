# MIGU Baseline on TRACE Default Order

## Experiment Overview

MIGU (Magnitude-based Gradient Updating) baseline on TRACE default task order with Qwen2-1.5B (seed=42).
MIGU is a replay-free gradient masking method that selectively updates parameters based on activation magnitudes in linear layers.

**Published reference**: FGGM Table 1 reports MIGU TRACE-OP = 44.08, General = 55.21.

## Setup

- **Model**: Qwen2-1.5B
- **Task order**: C-STANCE -> FOMC -> MeetingBank -> Py150 -> ScienceQA -> NumGLUE-cm -> NumGLUE-ds -> 20Minuten
- **Per-task epochs**: [5, 3, 7, 5, 3, 5, 5, 7]
- **Optimizer**: AdamW, LR=1e-5, constant with warmup, weight_decay=0
- **Batch size**: 128 (16 per device x 8 GPUs)
- **Precision**: BF16
- **DeepSpeed**: ZeRO Stage 2
- **MIGU threshold**: T=0.7 (70% of output dimensions masked, 30% updated)
- **MIGU applied**: Tasks 1-7 only (task 0 = pure SFT, matching reference)
- **GPUs**: 8x A100-80GB

### MIGU Implementation Details

MIGU uses `param.register_hook()` to intercept and modify gradients during backward pass.
This is necessary because DeepSpeed ZeRO-2 manages gradients internally -- `module.weight.grad` is always `None` after backward, making post-backward gradient masking a no-op.

Algorithm per batch:
1. **Forward hook**: For each `nn.Linear`, compute `activation = sum(|output|, dim=0)` across all tokens
2. **Backward hook** (via `param.register_hook`): All-reduce activations across GPUs, compute threshold at 70th percentile via `torch.quantile`, construct binary mask, multiply gradient element-wise

## Key Results

| Metric | Our Result | Published | Diff | Tolerance | Status |
|--------|-----------|-----------|------|-----------|--------|
| TRACE-OP | 47.43 | 44.08 | +3.35 | +/-2.0 | MARGINAL FAIL |
| BWT | -8.05 | N/A | - | - | - |
| OP_T (final row avg) | 44.76 | N/A | - | - | - |
| General | Not evaluated | 55.21 | - | - | - |

### OP per Step

| Step | OP_t |
|------|------|
| 1 (C-STANCE) | 50.70 |
| 2 (FOMC) | 51.24 |
| 3 (MeetingBank) | 40.97 |
| 4 (Py150) | 42.30 |
| 5 (ScienceQA) | 48.71 |
| 6 (NumGLUE-cm) | 49.72 |
| 7 (NumGLUE-ds) | 51.06 |
| 8 (20Minuten) | 44.76 |

### Performance Matrix (8x8)

| Checkpoint | C-STANCE | FOMC | MeetingBank | Py150 | ScienceQA | NumGLUE-cm | NumGLUE-ds | 20Minuten |
|------------|----------|------|-------------|-------|-----------|------------|------------|-----------|
| 0 | **50.70** | | | | | | | |
| 1 | 47.85 | **54.64** | | | | | | |
| 2 | 47.25 | 35.89 | **39.79** | | | | | |
| 3 | 47.90 | 26.61 | 38.75 | **55.93** | | | | |
| 4 | 47.95 | 22.18 | 37.72 | 54.08 | **81.65** | | | |
| 5 | 47.45 | 33.87 | 39.76 | 56.16 | 80.35 | **40.74** | | |
| 6 | 47.35 | 53.02 | 40.08 | 57.18 | 79.70 | 28.40 | **51.69** | |
| 7 | 47.45 | 30.44 | 34.20 | 55.06 | 80.15 | 33.33 | 38.15 | **39.26** |

Bold = diagonal (score immediately after training on that task).

### Comparison with SFT Baseline

| Metric | MIGU | SFT | Delta |
|--------|------|-----|-------|
| TRACE-OP | 47.43 | 49.31 | -1.88 |
| BWT | -8.05 | -34.25 | +26.20 |
| OP_T | 44.76 | 24.10 | +20.66 |

MIGU dramatically reduces catastrophic forgetting (BWT improves by 26 points). The final-step average (OP_T) is 20+ points higher because MIGU preserves earlier task knowledge.

## Key Observations

1. **TRACE-OP is 3.35 above published value** (47.43 vs 44.08). This is outside the +/-2.0 tolerance.

2. **Qualitative behavior is correct**: MIGU achieves much better BWT than SFT (-8.05 vs -34.25), confirming the gradient masking is effective at mitigating forgetting.

3. **MIGU TRACE-OP < SFT TRACE-OP** (47.43 < 49.31), consistent with the published trend. The higher SFT OP comes from better peak performance on each task (no masking constraint), despite worse forgetting.

4. **Possible sources of the gap**:
   - Our implementation uses `param.register_hook()` for DeepSpeed ZeRO-2 compatibility, while the FGGM paper's MIGU re-implementation used Accelerate's monkey-patched backward. These may intercept gradients at slightly different points in the computation graph.
   - The FGGM paper does not provide its specific MIGU re-implementation code, making exact comparison impossible.
   - Our SFT baseline was well-calibrated (49.31 vs published 49.22, diff=0.09), suggesting the evaluation pipeline is correct.

5. **Critical debugging finding**: Under DeepSpeed ZeRO-2, `module.weight.grad` is always `None` for all parameters after backward. The initial implementation (using post-backward `module.weight.grad *= mask`) was a complete no-op. This was fixed by using `param.register_hook()` which intercepts gradients during autograd backward before DeepSpeed processes them.

## Files

- MIGU implementation: `audit/methods/migu.py`
- Config: `audit/configs/migu_default.yaml`
- Training script: `audit/scripts/run_migu_train.sh`
- Checkpoints: `audit/results/migu_default_seed42/checkpoints/{0-7}/`
- Eval results: `audit/results/migu_default_seed42/eval/`
- Summary: `audit/results/migu_default_seed42/eval/summary.json`
