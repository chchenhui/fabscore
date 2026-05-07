# Ablation Study: Key-Diversity Loss (lambda2=0 vs lambda2=0.5)

## Experiment Overview

This ablation study removes the key-diversity loss component (setting lambda2=0) from the encryptor training to determine whether key-diversity regularization is necessary for the key-search attack to succeed. The hypothesis is that without explicit diversity encouragement, the encryptor may produce nearly identical encrypted embeddings for different keys (key-agnostic collapse), weakening the key-search attack.

## Setup

### Full Diversity Encryptor (lambda2=0.5)
- Existing high-diversity encryptor from task 4 optimization
- Config: lr=3e-5, lambda1=0.2, lambda2=0.5, margin=2.0, max_steps=5000
- Selected checkpoint: seed_123_diverse/best_checkpoint.pt
- KL=0.031, ASR@10=0.526, Diversity L2=0.430

### No-Diversity Encryptor (lambda2=0)
- Task-specified hyperparams: lr=1e-4, lambda1=1.0, lambda2=0.0, max_steps=3000
- 3 seeds: 42, 123, 456 (seed 123 collapsed, seeds 42/456 converged)
- Selected checkpoint: seed_42/last_checkpoint.pt (lowest KL=0.039)
- KL=0.039, ASR@10=0.0002, Diversity L2=0.117

### Key-Search Attack
- K values: {1, 2, 4, 8, 16, 32, 64}
- 3 probe seeds per encryptor (42, 123, 456)
- Thresholds calibrated on 52k Alpaca pool

## Key Results

### Embedding Diversity (200 prompts, 32 keys each)

| Variant | Mean Pairwise L2 | Std |
|---|---|---|
| Full (lambda2=0.5) | 0.430 | 0.043 |
| No diversity (lambda2=0) | 0.117 | 0.026 |

Without diversity loss, pairwise L2 distances drop 3.7x, confirming key-agnostic collapse.

### Key-Search Attack Comparison (TPR@FPR=1e-3)

| K | Full (lambda2=0.5) | No diversity (lambda2=0) |
|---|---|---|
| 1 | 0.849 +/- 0.022 | 0.863 +/- 0.018 |
| 2 | 0.833 +/- 0.018 | 0.849 +/- 0.022 |
| 4 | 0.819 +/- 0.019 | 0.839 +/- 0.017 |
| 8 | 0.793 +/- 0.027 | 0.833 +/- 0.018 |
| 16 | 0.750 +/- 0.031 | 0.826 +/- 0.014 |
| 32 | 0.693 +/- 0.034 | 0.818 +/- 0.020 |
| 64 | 0.599 +/- 0.034 | 0.813 +/- 0.024 |

### Summary

| Variant | KL Div | ASR@10 | Mean L2 | TPR@1e-3 (K=1) | TPR@1e-3 (K=32) | TPR Drop K=1->K=32 |
|---|---|---|---|---|---|---|
| Full (lambda2=0.5) | 0.031 | 0.526 | 0.430 | 0.849 | 0.693 | 15.6pp |
| No diversity (lambda2=0) | 0.039 | 0.000 | 0.117 | 0.863 | 0.818 | 4.4pp |

## Key Observations

1. **Key-diversity loss is essential for the key-search attack.** Without it, the TPR drop at K=32 is only 4.4pp (vs 15.6pp with diversity), making the attack ineffective.

2. **Key-agnostic collapse confirmed.** The no-diversity encryptor produces embeddings with 3.7x lower pairwise distances (0.117 vs 0.430), meaning different keys yield nearly identical encryptions.

3. **Tradeoff between attack surface and defense properties.** The no-diversity encryptor has much better privacy (ASR@10 ~0 vs 0.526) because its encryption is more consistent/predictable. The diversity loss trades privacy for key sensitivity, which is what enables the attack.

4. **K=1 baseline is comparable.** Both encryptors achieve similar TPR at K=1 (0.85-0.86), confirming that the single-key monitor works equally well regardless of diversity.

5. **Training stability note.** Seed 123 collapsed with lr=1e-4 (KL stayed at 2.02), while seeds 42 and 456 converged. The higher learning rate makes training less stable than the optimized config.

## Figures

- `results/figures/key_diversity_ablation.pdf` - Violin plot of pairwise L2 distances
- `results/figures/ablation_diversity_tpr_curve.pdf` - TPR@FPR vs K curves comparison
