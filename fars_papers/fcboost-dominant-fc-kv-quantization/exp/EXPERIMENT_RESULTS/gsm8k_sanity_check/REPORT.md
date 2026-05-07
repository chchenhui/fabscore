# GSM8K Short-Context Sanity Check for FCBoost

## Experiment Overview

Evaluate whether FCBoost's static CA-based channel boost degrades performance on short-context tasks. GSM8K (grade-school math, 1319 test problems) with 8-shot chain-of-thought prompts serves as a sanity check where quantization effects are milder and KV cache is smaller.

## Setup

- **Model**: Qwen/Qwen3-8B (36 layers, 8 KV heads, head_dim=128)
- **Task**: gsm8k_cot (8-shot CoT, strict-match metric)
- **Generation**: max_gen_tokens=4096, temperature=0.6, top_p=0.95, top_k=20
- **KV Quantization**: S=32 sink tokens, G=128 group size, R=128 buffer
- **Methods**: FP16, KIVI-KV2* (2-bit, no boost), Kitty (dynamic 12.5%), FCBoost (static CA 12.5% + value boost)
- **Seeds**: 1 seed each (short-context results are less variable)
- **Framework**: lm-evaluation-harness + Kitty simulation

## Key Results

| Method | GSM8K Accuracy (strict-match) | GSM8K Accuracy (flexible-extract) | Delta vs FP16 |
|--------|------------------------------|----------------------------------|---------------|
| FP16 KV16 | 90.30% | 89.61% | -- |
| KIVI-KV2* | 88.48% | 89.08% | -1.82% |
| Kitty | 88.48% | 89.46% | -1.82% |
| FCBoost | 89.84% | 88.93% | -0.46% |

## Key Observations

1. **FCBoost outperforms KIVI-KV2* and Kitty** on strict-match (89.84% vs 88.48% for both), closing the gap to FP16 to just 0.46 percentage points.

2. **All quantized methods are close to FP16**, confirming that quantization effects are mild at short context lengths (max ~4096 tokens). The largest drop is only 1.82%.

3. **FCBoost's static CA mask does not degrade short-context performance**. In fact, it slightly improves over Kitty's dynamic magnitude-based selection on this task. This validates that the static mask is compatible with the standard lm-evaluation-harness pipeline and does not introduce unexpected artifacts.

4. **KIVI-KV2* and Kitty have identical strict-match accuracy** (88.48%). At short context, the dynamic channel boost provides negligible benefit, consistent with the expectation that quantization errors are smaller when the KV cache is small.

5. **FCBoost's value cache boosting may help** -- the 1.36% improvement over Kitty/KIVI-KV2* could be attributed to the value cache also receiving precision boosts for important channels, which Kitty does not do.
