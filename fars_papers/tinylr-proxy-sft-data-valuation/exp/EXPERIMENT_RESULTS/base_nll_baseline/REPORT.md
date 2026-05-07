# Base-Model NLL Baseline (Condition C)

## Experiment Overview

Computed per-token negative log-likelihood (NLL) of the base Qwen2.5-1.5B model on assistant response tokens for all 12 candidate math datasets. This training-free baseline measures how well each dataset fits the pretrained model's distribution. Datasets with lower NLL are a better distributional match.

## Setup

- **Model**: Qwen/Qwen2.5-1.5B (base, no fine-tuning)
- **Precision**: bfloat16
- **Datasets**: 12 math-domain datasets, 50k samples each
- **Chat template**: Qwen2.5 ChatML format (system + user + assistant)
- **Loss masking**: Cross-entropy on assistant response tokens only
- **Sequence length**: cutoff_len=4096
- **Batch size**: 8
- **Hardware**: 4x A100-80GB GPUs (3 datasets per GPU, sequential)
- **Runtime**: ~3.6 hours

## Key Results

### NLL Ranking vs Target Ground-Truth

| Metric | Value |
|--------|-------|
| PDA | 0.6364 (95% CI [0.5152, 0.7428]) |
| Spearman rho | 0.3706 (p=0.236) |
| Top-1 match | False (NLL: mathplus, Target: dart-math-hard) |

### Comparison with Standard-LR Proxy (Condition A)

| Metric | Base NLL (C) | Std-LR Proxy (A) |
|--------|-------------|-------------------|
| PDA | 0.636 | 0.712 |
| Spearman rho | 0.371 | 0.594 |
| Top-1 match | No | Yes |

The standard-LR proxy SFT meaningfully outperforms the training-free NLL baseline, confirming that fine-tuning provides useful ranking signal beyond base-model distributional fit.

### Per-Dataset NLL Values (ascending)

| Rank | Dataset | Mean NLL | Tokens |
|------|---------|----------|--------|
| 1 | mathplus | 0.4103 | 10.4M |
| 2 | openmathinstruct-2 | 0.4354 | 19.3M |
| 3 | numinamath-cot | 0.5154 | 16.2M |
| 4 | R1-Distill-SFT-math | 0.5732 | 87.0M |
| 5 | numinamath1_5 | 0.6308 | 17.0M |
| 6 | QwQ-LongCoT-130K-math | 0.6445 | 93.5M |
| 7 | Magpie-Reasoning-V2-250K-CoT-QwQ-math | 0.7326 | 102.4M |
| 8 | AM-Thinking-v1-Distilled-math | 0.7410 | 111.7M |
| 9 | OpenR1-Math | 0.8608 | 169.4M |
| 10 | DeepMath-309K | 0.8619 | 171.2M |
| 11 | hkust-nlp__dart-math-hard | 1.0616 | 24.1M |
| 12 | Maths-College | 1.1189 | 40.2M |

## Key Observations

1. **NLL ranking has weak agreement with target**: PDA=0.636 is above random (0.5) but below the standard-LR proxy (0.712), and Spearman rho is not statistically significant (p=0.236).

2. **NLL fails on Top-1**: The best NLL fit (mathplus, NLL=0.41) does not match the target's top dataset (dart-math-hard). In fact, dart-math-hard has the 2nd worst NLL (1.06), suggesting high NLL does not preclude high training value.

3. **NLL correlates with response length/complexity**: Datasets with short, formulaic answers (mathplus, openmathinstruct-2) have low NLL, while datasets with long chain-of-thought reasoning (OpenR1-Math, DeepMath-309K, dart-math-hard) have high NLL. This suggests NLL primarily captures surface-level distributional similarity rather than training utility.

4. **Critical control finding**: If the Tiny-LR proxy's ranking agreement is no better than NLL (PDA ~0.64), it would suggest the proxy is not meaningfully training. Any PDA improvement above 0.64 demonstrates genuine training signal.
