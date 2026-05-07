# Target Ground-Truth Dataset Ranking Experiment

## Experiment Overview

Fine-tuned Qwen2.5-7B with LoRA on 12 math-domain candidate SFT datasets from OpenDataArena's scored-data collection. Each dataset was trained with 3 seeds (42, 123, 456) for a total of 36 runs. All checkpoints were evaluated on GSM8K (0-shot, exact match) and MATH-500 (0-shot, math_verify) to produce ground-truth target scores S_t(D_i).

## Setup

- **Base model**: Qwen/Qwen2.5-7B
- **Fine-tuning**: LoRA (rank=16, alpha=32, dropout=0.05, target=all)
- **Training**: 2000 steps, lr=5e-5, cosine schedule, warmup=0.1, effective batch size=16
- **DeepSpeed**: ZeRO-3, 8x A100 GPUs per run
- **Periodic eval**: Every 200 steps, load_best_model_at_end=true (by eval_loss)
- **Seeds**: 42, 123, 456
- **Framework**: LLaMA-Factory v0.9.5.dev0
- **Evaluation**: lm-eval-harness v0.4.11 with vLLM v0.12.1 backend (tp=4, dp=2)
- **Datasets**: 12 math-domain subsets from OpenDataArena/OpenDataArena-scored-data, each sampled to 50k examples (seed=0)

## Key Results

### Dataset Ranking by Composite Score (avg of GSM8K and MATH-500 means)

| Rank | Dataset | Composite | GSM8K Mean | MATH-500 Mean |
|------|---------|-----------|------------|---------------|
| 1 | hkust-nlp__dart-math-hard | 0.6034 | 0.7887 ± 0.0566 | 0.4180 ± 0.0099 |
| 2 | openmathinstruct-2 | 0.5917 | 0.7895 ± 0.0432 | 0.3940 ± 0.0065 |
| 3 | R1-Distill-SFT-math | 0.4816 | 0.6366 ± 0.0429 | 0.3267 ± 0.0062 |
| 4 | mathplus | 0.4053 | 0.3912 ± 0.0699 | 0.4193 ± 0.0146 |
| 5 | numinamath-cot | 0.3920 | 0.3801 ± 0.0982 | 0.4040 ± 0.0059 |
| 6 | DeepMath-309K | 0.3407 | 0.4273 ± 0.0368 | 0.2540 ± 0.0118 |
| 7 | OpenR1-Math | 0.3302 | 0.4304 ± 0.0163 | 0.2300 ± 0.0193 |
| 8 | QwQ-LongCoT-130K-math | 0.2939 | 0.4759 ± 0.0199 | 0.1120 ± 0.0130 |
| 9 | numinamath1_5 | 0.2689 | 0.2231 ± 0.0450 | 0.3147 ± 0.0259 |
| 10 | Magpie-Reasoning-V2-250K-CoT-QwQ-math | 0.2505 | 0.3556 ± 0.0031 | 0.1453 ± 0.0137 |
| 11 | Maths-College | 0.1747 | 0.0801 ± 0.0058 | 0.2693 ± 0.0115 |
| 12 | AM-Thinking-v1-Distilled-math | 0.1463 | 0.1734 ± 0.0074 | 0.1193 ± 0.0084 |

## Key Observations

1. **Top performers**: dart-math-hard and openmathinstruct-2 are clearly the best datasets, with composite scores >0.59 and GSM8K accuracy near 79%.
2. **R1-Distill-SFT-math** ranks 3rd with strong GSM8K (63.7%) but lower MATH-500 (32.7%).
3. **mathplus and numinamath-cot** are close (4th/5th), with mathplus excelling on MATH-500 and numinamath-cot being balanced.
4. **Large variance**: Some datasets show high seed-to-seed variance (e.g., numinamath-cot GSM8K std=0.098, mathplus GSM8K std=0.070).
5. **Bottom performers**: Maths-College and AM-Thinking show poor GSM8K performance (<18%), suggesting these datasets are less effective for math reasoning SFT.
6. **GSM8K vs MATH-500 divergence**: QwQ-LongCoT shows high GSM8K (47.6%) but very low MATH-500 (11.2%), indicating it trains for simpler math but not harder problems.

## Evaluation Notes

- GSM8K metric: `exact_match,flexible-extract` (standard lm-eval metric)
- MATH-500 metric: `math_verify,none` (uses math_verify package for symbolic equivalence)
- The `exact_match` metric for MATH-500 is degraded due to antlr4 version incompatibility (sympy.parsing.latex requires antlr4==4.11, but omegaconf requires 4.9). The `math_verify` metric is used instead, which is more robust.
- N-gram contamination check was performed during data preparation; no datasets showed >5% 10-gram overlap with test sets.
