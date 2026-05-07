# Optimization Iteration 0: Tiny-LR Proxy SFT with Increased LR and Steps

## Experiment Overview

Optimized the Tiny-LR Proxy SFT method by increasing the learning rate from 5e-6 to 1e-5 and doubling training steps from 500 to 1000. The original tiny-LR proxy produced PDA=0.349 with an inverted ranking (Spearman rho=-0.378). The root cause was insufficient learning at LR=5e-6, causing the proxy to rank datasets by base-model affinity (favoring long-response CoT datasets) rather than learned math skills.

## Setup

- **Model**: Qwen/Qwen2.5-1.5B (proxy model)
- **Method**: LoRA SFT (rank=16, alpha=32, dropout=0.05, target=all)
- **Learning rate**: 1e-5 (increased from 5e-6; still 5x below standard 5e-5)
- **Training steps**: 1000 (doubled from 500)
- **Effective batch size**: 16 (4 x 4 grad_accum)
- **Schedule**: cosine, warmup_ratio=0.1, bf16
- **Evaluation**: GSM8K (0-shot) + MATH-500 (0-shot, math_verify)
- **Seeds**: 42, 123, 456 (3 seeds per dataset, scores averaged)
- **Total runs**: 12 datasets x 3 seeds = 36 training + 36 evaluation
- **Hardware**: 8x A100-80GB for training, 1x A100-80GB for evaluation

## Key Results

### Ranking Agreement with Target (Qwen2.5-7B)

| Metric | Original Tiny-LR (5e-6, 500 steps) | Optimized Tiny-LR (1e-5, 1000 steps) | Std-LR Proxy (5e-5, 500 steps) | Base NLL |
|--------|-------------------------------------|----------------------------------------|----------------------------------|----------|
| PDA (composite) | 0.349 [0.242, 0.470] | **0.500** [0.379, 0.621] | 0.712 [0.606, 0.818] | 0.636 |
| Spearman rho | -0.378 (p=0.226) | **-0.091** (p=0.779) | 0.594 (p=0.042) | 0.371 |
| Top-1 match | No | No | Yes | No |
| **PDA (MATH500-only)** | 0.758 | **0.818** | 0.864 | N/A |
| Spearman rho (MATH500) | 0.699 (p=0.011) | **0.846** (p=0.0005) | 0.874 (p=0.0002) | N/A |
| PDA (GSM8K-only) | 0.470 | 0.515 | 0.667 | N/A |

### Dataset Rankings

| Rank | Optimized Tiny-LR v2 | Target (Ground Truth) |
|------|----------------------|----------------------|
| 1 | QwQ-LongCoT-130K-math (0.358) | dart-math-hard (0.603) |
| 2 | Magpie-Reasoning (0.357) | openmathinstruct-2 (0.592) |
| 3 | **dart-math-hard (0.335)** | R1-Distill-SFT-math (0.482) |
| 4 | AM-Thinking (0.229) | mathplus (0.432) |
| 5 | numinamath1_5 (0.227) | numinamath-cot (0.381) |
| 6 | openmathinstruct-2 (0.202) | DeepMath-309K (0.338) |
| 7 | mathplus (0.198) | OpenR1-Math (0.302) |
| 8 | numinamath-cot (0.193) | QwQ-LongCoT-130K (0.272) |
| 9 | R1-Distill-SFT-math (0.188) | numinamath1_5 (0.241) |
| 10 | OpenR1-Math (0.176) | Magpie-Reasoning (0.228) |
| 11 | Maths-College (0.151) | Maths-College (0.175) |
| 12 | DeepMath-309K (0.132) | AM-Thinking (0.146) |

## Key Observations

1. **Composite PDA improved from 0.349 to 0.500**: The ranking is no longer inverted (Spearman rho improved from -0.378 to -0.091), but composite PDA is at chance level, still below the standard-LR proxy (0.712) and base NLL (0.636).

2. **MATH500-only ranking is excellent**: PDA=0.818 with Spearman rho=0.846 (p=0.0005), approaching the standard-LR proxy (PDA=0.864). This suggests the optimized tiny-LR proxy captures real math skill differences when evaluated on harder problems.

3. **GSM8K still confounded by response length**: CoT-heavy datasets (QwQ, Magpie) score disproportionately high on GSM8K, dragging down the composite ranking. This is a fundamental limitation of the 1.5B proxy model on simpler benchmarks.

4. **dart-math-hard improved to rank 3** (from rank 8 in original tiny-LR): The target-best dataset now ranks near the top, confirming the model learned meaningful math skills.

5. **Loss diagnostics healthy**: Average loss drop 22.1% (up from 16.4%), confirming more substantial learning with the higher LR and longer training.

6. **The tiny-LR hypothesis remains partially supported for harder benchmarks (MATH500) but not for the composite metric**: The proxy-to-target transfer is benchmark-dependent.
