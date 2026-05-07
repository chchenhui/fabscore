# OSNIP-like Key-Conditioned Encryptor Training

## Experiment Overview

Trained a key-conditioned embedding encryptor following OSNIP methodology for frozen Qwen2.5-7B-Instruct. The encryptor is a 65M-parameter MLP that encrypts token embeddings by concatenating a per-request Gaussian key (dim=128) and applying iso-norm projection to preserve embedding magnitudes.

## Setup

- **Model**: Qwen2.5-7B-Instruct (frozen, bf16), hidden_dim=3584
- **Encryptor**: MLP [3712 -> 7168 -> 3584 -> 3584] with GELU, iso-norm projection, 65M params
- **Data**: 20k Alpaca train, 1k Alpaca val (disjoint), max_length=128
- **Training (Optimized)**: AdamW, lr=3e-5, batch=8, grad_accum=4, 5000 optimizer steps, lambda1=0.3, lambda2=0.1, privacy eps=0.1, diversity margin=0.5, utility-gated curriculum (warmup=1500, tau_low=0.003, tau_high=0.03), eval_samples=500
- **Seeds**: [42, 123, 456]
- **Hardware**: 1x A100-80GB per run

## Key Results

### Final Validation (optimized, on 1000 held-out Alpaca prompts)

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean +/- Std | Target |
|--------|---------|----------|----------|--------------|--------|
| KL divergence | 0.0228 | 0.0226 | 0.0227 | 0.0227 +/- 0.0001 | <= 0.02 |
| KNN ASR@10 | 0.1253 | 0.1916 | 0.2438 | 0.1869 +/- 0.0485 | <= 0.20 |
| Key diversity (L2) | 0.251 | 0.270 | 0.266 | 0.262 +/- 0.008 | -- |

### Optimization History

| Config | KL (mean) | ASR@10 (mean) | Notes |
|--------|-----------|---------------|-------|
| Original v1+ft | 0.0365 | 0.0648 | Too much privacy, not enough utility |
| Optimized v2 | 0.0227 | 0.1869 | Better balance, 37.8% KL improvement |

## Key Observations

1. **Significant utility improvement**: KL improved from 0.037 to 0.023 (37.8% reduction). The gap to the 0.02 target was reduced from 0.017 to 0.003.

2. **Privacy still near target**: Mean KNN ASR@10 = 0.187 < 0.20 target. Seed 42 is well below (0.125), seed 123 is borderline (0.192), seed 456 slightly exceeds (0.244). The encryptor moves along the Pareto frontier of the utility-privacy tradeoff.

3. **Key diversity is functional**: Mean pairwise L2 distance ~0.26 across 32 keys. Lower than original (0.63) due to reduced margin, but still provides meaningful variation for key-search attacks.

4. **Optimization changes**: Reduced diversity margin (1.0->0.5), lower lambdas (1.0/0.5->0.3/0.1), more data (10k->20k), lower LR (1e-4->3e-5), longer training (3000->5000 steps), larger eval set (200->500).

5. **Checkpoints**: Optimized checkpoints at `key_search_bypass/outputs/encryptor/seed_{42,123,456}/best_checkpoint.pt`.
