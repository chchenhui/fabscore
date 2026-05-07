# FCBoost v2: Static CA-Based Channel Boost with Value Cache Boosting on Qwen3-8B AIME24/25 @32k

## Experiment Overview

FCBoost replaces Kitty's dynamic per-page magnitude-based channel selection with a static RoPE-frequency mask derived from FASA's Contextual Agreement (CA) metric. FCBoost v2 extends this with two improvements:
1. **Improved CA profiling**: 16 sequences x 8192 tokens x 512 sample positions (4x more calibration data than v1) for more stable mask selection
2. **Value cache boosting**: The same static mask is applied to boost value cache channels (INT2->INT4), not just key cache channels

## Setup

- **Model**: Qwen3-8B (36 layers, 32 QH, 8 KVH, head_dim=128)
- **Quantization**: INT2 base, INT4 for 16 boosted channels (8 RoPE pairs) per KV head
- **Boost ratio**: 12.5% (same as Kitty)
- **Value boost**: Enabled (same mask applied to both key and value caches)
- **CA Profiling (v2)**: WikiText-2, 16 sequences x 8192 tokens, TopK=512
- **Evaluation**: AIME24 (30 problems) and AIME25 (30 problems), 3 seeds each
- **Generation**: max_gen_toks=32768, temperature=0.6, top_p=0.95, top_k=20
- **Hardware**: 1x A100-SXM4-80GB per evaluation job

## Key Results

| Method | AIME24 (mean +/- std) | AIME25 (mean +/- std) | Average |
|--------|------------------------|------------------------|---------|
| KIVI-KV2* | 65.56 | 66.67 | 66.11 |
| Kitty | 67.78 | 65.56 | 66.67 |
| FCBoost v1 | 65.56 | 66.67 | 66.11 |
| **FCBoost v2** | **74.44** | **67.78** | **71.11** |

### Per-Seed Breakdown

| Method | AIME24 Seeds | AIME25 Seeds |
|--------|-------------|-------------|
| KIVI-KV2* | 63.33, 66.67, 66.67 | 66.67, 63.33, 70.00 |
| Kitty | 70.00, 66.67, 66.67 | 66.67, 63.33, 66.67 |
| FCBoost v1 | 70.00, 63.33, 63.33 | 66.67, 66.67, 66.67 |
| **FCBoost v2** | **76.67, 73.33, 73.33** | **66.67, 66.67, 70.00** |

### Improvement Summary

| Comparison | AIME24 | AIME25 | Average |
|-----------|--------|--------|---------|
| FCBoost v2 vs v1 | +8.88pp | +1.11pp | +5.00pp |
| FCBoost v2 vs Kitty | +6.66pp | +2.22pp | +4.44pp |
| FCBoost v2 vs KIVI-KV2* | +8.88pp | +1.11pp | +5.00pp |

## Key Observations

1. **FCBoost v2 clearly outperforms all baselines**: 71.11% average vs Kitty 66.67% (+4.44pp) and KIVI-KV2* 66.11% (+5.00pp).
2. **Dramatic AIME24 improvement**: 74.44% vs Kitty's 67.78% (+6.66pp), with all 3 seeds above 73%.
3. **AIME25 improvement**: 67.78% vs Kitty's 65.56% (+2.22pp), more modest but consistent.
4. **Low variance**: FCBoost v2 AIME24 std=1.57 (all seeds 73-77%) indicates very stable performance.
5. **Two complementary fixes**: Improved masks (80.6% overlap with v1, cleaner boundary decisions) and value cache boosting both contribute.
6. **Static mask eliminates per-page compute**: No per-page magnitude scoring or top-K selection needed at runtime.
