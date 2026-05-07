# Standard-LR Proxy SFT Baseline (Condition A)

## Experiment Overview

Fine-tuned the proxy model Qwen2.5-1.5B on each of 12 math-domain candidate datasets using standard SFT learning rate (5e-5), then evaluated on GSM8K and MATH-500. This is Condition A: the conventional proxy-model approach. The resulting dataset ranking is compared against the Qwen2.5-7B target model's ground-truth ranking.

## Setup

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-1.5B |
| Finetuning | LoRA (rank=16, alpha=32, dropout=0.05, target=all) |
| Learning Rate | 5e-5 (cosine, warmup=0.1) |
| Max Steps | 500 |
| Effective Batch Size | 16 (per_device=4, grad_accum=4) |
| Seeds | 42, 123, 456 |
| Evaluation | GSM8K (exact_match, flexible-extract), MATH-500 (math_verify) |
| Hardware | 8x A100-80GB (1 GPU per run, 8 parallel) |
| Best Checkpoint | load_best_model_at_end=true, metric=eval_loss |

## Key Results

### Ranking Agreement with Target (Qwen2.5-7B)

| Metric | Value |
|--------|-------|
| PDA (Pairwise Direction Accuracy) | 0.7121 |
| PDA 95% CI (bootstrap) | [0.6061, 0.8182] |
| Spearman rho | 0.5944 |
| Spearman p-value | 0.0415 |
| Top-1 Match | Yes (dart-math-hard) |

### Proxy Ranking vs Target Ranking

| Rank | Proxy (Std-LR 1.5B) | Composite | Target (7B) | Composite |
|------|---------------------|-----------|-------------|-----------|
| 1 | dart-math-hard | 0.5047 | dart-math-hard | 0.6034 |
| 2 | mathplus | 0.4318 | openmathinstruct-2 | 0.5917 |
| 3 | R1-Distill-SFT-math | 0.3383 | R1-Distill-SFT-math | 0.4816 |
| 4 | numinamath1_5 | 0.3135 | mathplus | 0.4053 |
| 5 | QwQ-LongCoT-130K-math | 0.2994 | numinamath-cot | 0.3920 |
| 6 | openmathinstruct-2 | 0.2979 | DeepMath-309K | 0.3407 |
| 7 | numinamath-cot | 0.2959 | OpenR1-Math | 0.3302 |
| 8 | Magpie-Reasoning-V2-250K-CoT-QwQ-math | 0.2721 | QwQ-LongCoT-130K-math | 0.2939 |
| 9 | AM-Thinking-v1-Distilled-math | 0.2023 | numinamath1_5 | 0.2689 |
| 10 | OpenR1-Math | 0.1727 | Magpie-Reasoning-V2-250K-CoT-QwQ-math | 0.2505 |
| 11 | Maths-College | 0.1396 | Maths-College | 0.1747 |
| 12 | DeepMath-309K | 0.1216 | AM-Thinking-v1-Distilled-math | 0.1463 |

### Key Observations

1. The standard-LR proxy correctly identifies the top-1 dataset (dart-math-hard) and gets the top-3 partially right (2 of 3 match).
2. PDA of 0.7121 is significantly above random (0.5), indicating the proxy provides useful signal.
3. The biggest ranking flips: openmathinstruct-2 drops from rank 2 (target) to rank 6 (proxy), DeepMath-309K drops from rank 6 (target) to rank 12 (proxy).
4. Spearman correlation is moderate (0.5944) and statistically significant (p=0.0415).
5. The bottom-tier datasets (Maths-College, AM-Thinking) are reasonably identified by the proxy.
