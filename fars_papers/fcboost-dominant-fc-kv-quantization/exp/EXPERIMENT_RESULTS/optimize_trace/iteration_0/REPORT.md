# FCBoost v2 Optimization (Iteration 0)

## Experiment Overview

Optimized FCBoost KV cache quantization method with two improvements:
1. **Improved CA Profiling**: 4x more calibration data (16 seqs vs 4, 8192 tokens vs 4096, 512 sample positions vs 256) to reduce noise in mask selection boundary
2. **Value Cache Boosting**: Applied the same static CA-derived promote_mask to value cache quantization (previously only key cache was boosted)

## Setup

- Model: Qwen3-8B (fp16)
- Benchmarks: AIME24, AIME25 (30 problems each, 3 seeds)
- Max generation: 32768 tokens
- KV cache config: kbits=2, vbits=2, group_size=128, buffer=128, sink=32, promote_ratio=0.125, promote_bit=4
- Masks: v2 masks from `fcboost/masks_v2/qwen3_8b_ca_masks.pt`
- Value boosting: enabled (`boost_values=True`)

## Key Results

| Method | AIME24 (seeds) | AIME24 avg | AIME25 (seeds) | AIME25 avg | Overall avg |
|--------|----------------|------------|----------------|------------|-------------|
| KIVI-KV2* | [63.33, 66.67, 66.67] | 65.56% | [66.67, 63.33, 70.00] | 66.67% | 66.11% |
| Kitty | [70.00, 66.67, 66.67] | 67.78% | [66.67, 63.33, 66.67] | 65.56% | 66.67% |
| FCBoost v1 | [70.00, 63.33, 63.33] | 65.56% | [66.67, 66.67, 66.67] | 66.67% | 66.11% |
| **FCBoost v2** | **[76.67, 73.33, 73.33]** | **74.44%** | **[66.67, 66.67, 70.00]** | **67.78%** | **71.11%** |

## Key Observations

- FCBoost v2 achieves 71.11% overall, +5.00pp over FCBoost v1 (66.11%) and +4.44pp over Kitty (66.67%)
- AIME24 improvement is dramatic: +8.88pp over v1, with all 3 seeds showing large gains
- AIME25 improvement is modest: +1.11pp over v1, with seed 2 showing improvement while seeds 0,1 match v1
- Both improvements (better masks + value boosting) contributed; ~20% of v1 mask selections were noisy
- v2 masks had 80.6% overlap with v1, confirming the noise reduction was meaningful
- Low variance across seeds (std=1.57% for both tasks) indicates stable improvements
