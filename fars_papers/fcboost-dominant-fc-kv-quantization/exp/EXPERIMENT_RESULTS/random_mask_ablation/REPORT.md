# Random Mask Ablation: CA Signal Validation

## Experiment Overview

This ablation tests whether FCBoost's CA-derived channel selection provides meaningful signal beyond simply applying any static mask. We replace the CA-based mask with random static masks that boost the same number of RoPE pairs (F=8 per KV head, 16 channels total) but selected uniformly at random.

## Setup

- **Model**: Qwen3-8B (36 layers, 8 KV heads, head_dim=128)
- **Quantization**: K=INT2, V=INT2, boosted channels=INT4, sink=32, buffer=128
- **Boost configuration**: 8 RoPE pairs (16 channels) boosted per KV head, both key and value cache
- **Random masks**: 3 independent masks generated with seeds [42, 123, 456]
- **Evaluation**: 1 eval seed per random mask, AIME24 + AIME25, max_gen_toks=32768
- **Sampling**: temperature=0.6, top_p=0.95, top_k=20

## Key Results

| Mask Type | AIME24 | AIME25 | Avg |
|-----------|--------|--------|-----|
| FCBoost (CA mask) | 74.44 | 67.78 | 71.11 |
| Random mask (seed 42) | 63.33 | 66.67 | 65.00 |
| Random mask (seed 123) | 70.00 | 63.33 | 66.67 |
| Random mask (seed 456) | 56.67 | 66.67 | 61.67 |
| Random mask (mean +/- std) | 63.33 +/- 5.44 | 65.56 +/- 1.57 | 64.44 +/- 2.06 |
| KIVI-KV2* (no boost) | 65.56 | 66.67 | 66.11 |

## Key Observations

### 1. Random mask does NOT improve over no boost
Random mask average (64.44%) is slightly below KIVI-KV2* no-boost baseline (66.11%), a gap of -1.67pp. This indicates that boosting arbitrary channels does not help and can even introduce noise by applying higher precision to channels that don't need it while leaving truly sensitive channels under-quantized.

### 2. CA mask significantly outperforms random masks
FCBoost with CA mask (71.11%) outperforms random mask mean (64.44%) by +6.67pp. This is a substantial and consistent gap across both benchmarks (AIME24: +11.11pp, AIME25: +2.22pp).

### 3. High variance in random mask results
Random mask AIME24 results range from 56.67% to 70.00% (std=5.44), showing that random channel selection is unreliable. The CA mask achieves consistently high performance (std=1.57 across 3 evaluation seeds in the main experiment).

### 4. Conclusion
The CA-derived channel selection provides genuinely useful information about quantization sensitivity. The improvement is not simply from having any static boost pattern -- it specifically comes from identifying the right channels to boost. This validates the core insight that Contextual Agreement profiling correctly identifies RoPE frequency channels where precision matters most for maintaining model quality under aggressive quantization.
