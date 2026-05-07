# Optimization Iteration 0: Reduce r_cap from 30 to 2.0

## Experiment Overview

The original RRCS experiment used r_cap=30, which was too conservative for a 4x4 DS matrix. With exp(-30) ~ 1e-13, the Sinkhorn output remained a near-exact permutation matrix, and H_res gradients were ~1e-15 (effectively zero). This optimization reduces r_cap to restore meaningful gradient flow through the Sinkhorn projection.

## Diagnosis

**Root cause**: r_cap=30 was borrowed from mHC-lite's convergence warning threshold, not designed for gradient flow. For a 4x4 matrix, meaningful off-diagonal entries require r_cap in the range of ~2-8.

**Fix**: Swept r_cap in {2.0, 5.0, 8.0} with 3 seeds each (9 total runs, 5000 iters each).

**Additional fix**: DiagnosticLogger was not receiving the correct r_cap from train.py and was computing DS error/entropy using the uncapped Sinkhorn. Fixed to pass mhc_r_cap and mhc_rrcs to the logger, and modified the internal Sinkhorn call to apply RRCS capping when enabled.

## Setup

- **Model**: nanoGPT 48-layer, n_embd=150, n_head=6, ~20.8M params
- **Dataset**: FineWeb10B (900M train/100M val tokens, GPT-2 tokenizer)
- **Training**: 5000 iters, batch_size=8, grad_accum=4, lr=6e-4 cosine to 6e-5, warmup=200
- **mHC config**: hc_num_streams=4, sinkhorn_tau=0.05, sinkhorn_iters=10, mhc_rrcs=True
- **r_cap sweep**: {2.0, 5.0, 8.0}
- **Seeds**: 42, 123, 456
- **Hardware**: 1x A100-SXM4-80GB per run (9 parallel runs)
- **Runtime**: ~3.2 hours per run

## Key Results

### r_cap Comparison

| r_cap | Best Val Loss (mean +/- std) | H_res Grad Median | H_res Param Drift | Entropy | Stability |
|-------|------------------------------|-------------------|--------------------|---------|-----------|
| 2.0   | **4.775 +/- 0.010**         | 4.1e-6            | 4.19               | 0.933   | 1.28-1.34 |
| 5.0   | 4.791 +/- 0.015             | 1.7e-5            | 5.19               | *       | *         |
| 8.0   | 4.804 +/- 0.015             | 7.6e-6            | 5.35               | *       | *         |
| orig30| 4.771 +/- 0.009             | 1.9e-15           | 2.1e-7             | 0.0     | 1.22-1.55 |
| mhc_def| 4.772 +/- 0.011            | 0.0               | 0.0                | 0.0     | 1.23-1.55 |

(*) Diagnostics not computed with RRCS-capped Sinkhorn for r_cap=5/8 since the DiagnosticLogger fix only applies to r_cap passed at runtime; the original runs' diagnostics.csv used the old logger with r_cap=30 hardcoded. But gradient/drift metrics are trustworthy as they come from autograd directly.

### Best Condition: r_cap=2.0

| Seed | Best Val Loss | Final Val Loss | H_res Grad Median | H_res Drift | Entropy |
|------|---------------|----------------|-------------------|-------------|---------|
| 42   | 4.776         | 4.776          | 4.13e-6           | 4.22        | 0.933   |
| 123  | 4.786         | 4.795          | 4.07e-6           | 4.15        | 0.933   |
| 456  | 4.763         | 4.763          | 4.09e-6           | 4.20        | 0.933   |

### Success Criteria Evaluation

1. **H_res gradient increase >= 100x**: 4.1e-6 / 1.9e-15 = 2.2 billion x. **PASS**
2. **H_res param drift >= 10x**: 4.19 / 2.1e-7 = 20 million x. **PASS**
3. **Val loss not worse by > 0.5x baseline std**: 4.775 - 4.771 = 0.004, threshold = 0.5 * 0.009 = 0.005. Difference of 0.004 < 0.005. **PASS**

## Key Observations

1. **r_cap=2.0 restores gradient flow**: H_res_logits gradients are ~4e-6, billions of times larger than the ~1e-15 with r_cap=30. The Sinkhorn output is no longer a permutation matrix.

2. **H_res actually learns**: Parameter drift of 4.19 Frobenius norm units (vs ~0 before). Individual logits shift by ~0.01-0.09 over training, with some off-diagonal entries changing by ~0.2-0.3 in the most active layers.

3. **DS matrix has meaningful mixing**: Entropy ~0.93 (vs 0.0) indicates the doubly-stochastic matrix has non-trivial routing between streams. DS error remains negligible (~1e-6).

4. **Val loss is essentially unchanged**: 4.775 vs 4.771 -- within seed variance. This is expected for a 20M model on 5000 iters; the benefit of learned routing may become clearer at larger scales or longer training.

5. **r_cap=2.0 is the best trade-off**: It gives the lowest val loss among the sweep while still achieving massive gradient flow improvement. r_cap=5 and 8 have slightly worse val loss, possibly because the higher cap allows more logit range and thus "sharper" routing that is harder to optimize.

6. **Training is stable**: Gradient norm spike ratios of 1.28-1.34 are comparable to or better than the baseline (1.22-1.55). No NaN, no loss explosions.
