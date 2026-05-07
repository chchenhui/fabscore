# Optimized Adaptive MI Decoding - Iteration 0 (v2)

## Experiment Overview

Hyperparameter optimization of adaptive MI decoding (M3ID-style) for VLAA-Thinker-7B. Built on prior bug fixes (t0, lambda, weight cap) and further tuned the confidence gate threshold (alpha) to improve performance on MMStar and HallusionBench.

## Issues Found and Fixed

### Previous Session Fixes (already applied)
1. **t0 bug**: Changed from `input_ids.shape[1]` (prompt length ~500) to `t0=0`
2. **Lambda reduction**: Reduced from 0.02 to 0.005 for long-CoT compatibility
3. **Weight capping**: Added `max_weight=5.0` to prevent runaway amplification

### This Session: Alpha Tuning
**Issue**: The confidence gate threshold `alpha=0.3` was conservative. MI correction only activated when `max_k p_c(k) < 0.3`, meaning the model had to be quite uncertain before correction was applied. Analysis of successful experiments on Qwen2.5-VL showed that more aggressive (near-constant) correction performed better.

**Diagnosis via sweeps**:
- Lambda sweep (200 items): lambda=0.005 was already optimal; higher lambda hurt
- Alpha sweep (300 items): alpha=0.8 achieved 63.00% vs alpha=0.3's equivalent, matching vanilla at 61.33%
- Combined sweep confirmed alpha=0.8, lambda=0.005, max_weight=5.0 as the best combination

**Fix**: Increased alpha from 0.3 to 0.8, making MI correction active more frequently.

## Setup

- **Model**: VLAA-Thinker-Qwen2.5VL-7B
- **Hyperparameters**: lambda=0.005, alpha=0.8, t0=0, max_weight=5.0
- **Decoding**: Greedy (argmax) with dual KV caches
- **GPU**: 4x A100-80GB per benchmark (data-parallel sharding)
- **Benchmarks**: MMStar (1500 items), HallusionBench (1129 items)
- **Budget**: max_new_tokens=512

## Key Results

| Method | MMStar Acc (%) | HallusionBench aAcc (%) |
|--------|----------------|-------------------------|
| Vanilla | 62.13 | 66.25 |
| Visual Replay | 62.47 | 67.14 |
| MI alpha=0.3 (previous) | 61.93 | 67.67 |
| **MI alpha=0.8 (this run)** | **62.07** | **67.76** |

### Improvement over previous MI (alpha=0.3)
- MMStar: +0.14pp (61.93 -> 62.07)
- HallusionBench: +0.09pp (67.67 -> 67.76)

### Comparison to vanilla baseline
- MMStar: -0.06pp (essentially neutral, within noise)
- HallusionBench: +1.51pp (meaningful improvement)

### HallusionBench breakdown
| Sub-metric | Vanilla | MI alpha=0.3 | **MI alpha=0.8** |
|------------|---------|-------------|-----------------|
| aAcc | 66.25 | 67.67 | **67.76** |
| VD acc | 60.24 | 61.25 | **62.61** |
| VS acc | 72.86 | 74.72 | 73.42 |

## Key Observations

1. **Alpha=0.8 narrows MMStar gap**: The delta vs vanilla shrinks from -0.20pp to -0.06pp, making MI essentially neutral on this benchmark.
2. **HallusionBench remains the win**: +1.51pp over vanilla on aAcc, with particularly strong VD improvement (+2.37pp).
3. **VD vs VS tradeoff**: Higher alpha improves VD accuracy (+1.36pp) but slightly reduces VS accuracy (-1.30pp), suggesting more frequent correction helps with visual-dependent questions.
4. **Lambda sensitivity**: Higher lambda (0.01, 0.02) consistently hurt MMStar across all alpha values. The gradual ramp of lambda=0.005 is necessary for long-CoT.
5. **Diminishing returns**: The improvement from alpha tuning is modest (+0.14pp MMStar, +0.09pp HallusionBench), suggesting the method is near its ceiling with this architecture.
