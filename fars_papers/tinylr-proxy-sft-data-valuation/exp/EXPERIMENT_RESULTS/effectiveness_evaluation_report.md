# Effectiveness Evaluation Report

## Verdict: bad

## Summary

The Tiny-LR Proxy SFT method (Condition B) does **not** produce dataset rankings that match the target model's ground truth better than the Standard-LR Proxy SFT baseline (Condition A). After two optimization rounds, the best tiny-LR configuration (LR=1e-5, 1000 steps) achieves composite PDA=0.500 -- indistinguishable from random -- while the standard-LR proxy achieves PDA=0.712. The hypothesis that reducing the proxy learning rate improves dataset ranking transfer is refuted on the composite metric. A partial success on MATH-500-only ranking (PDA=0.818) does not compensate for complete failure on GSM8K (PDA=0.515).

## Experiment Feasibility Check

All experiments ran successfully without infrastructure issues:

- **Target ground truth**: 36 runs (12 datasets x 3 seeds) on Qwen2.5-7B completed. Results in `results/target_scores.csv`.
- **Standard-LR Proxy (A)**: 36 runs on Qwen2.5-1.5B at LR=5e-5, 500 steps completed. Results in `results/proxy_std_scores.csv`.
- **Training-Free NLL (C)**: NLL computation on all 12 datasets completed. Results in `results/base_nll_scores.csv`.
- **Tiny-LR Proxy (B) original**: 36 runs at LR=5e-6, 500 steps completed. PDA=0.349 (inverted).
- **Tiny-LR Proxy (B) optimized**: 36 runs at LR=1e-5, 1000 steps completed. PDA=0.500.
- **Loss diagnostics**: All 36 optimized tiny-LR runs show loss trajectories; avg 22.1% loss drop confirms non-degenerate training.

No missing results. Both main experiment and all baselines produced valid outputs.

## Results Analysis

### Main Comparison Table (Composite Ranking)

| Method | PDA | 95% CI | Spearman rho | p-value | Top-1 |
|--------|-----|--------|-------------|---------|-------|
| Random Baseline | 0.500 | [analytical] | 0.000 | N/A | N/A |
| Training-Free NLL (C) | 0.636 | [0.515, 0.743] | 0.371 | 0.236 | False |
| Standard-LR Proxy SFT (A) | 0.712 | [0.606, 0.818] | 0.594 | 0.042 | True |
| **Tiny-LR Proxy SFT (B)** | **0.500** | **[0.379, 0.621]** | **-0.091** | **0.779** | **False** |

**Key finding**: Tiny-LR PDA (0.500) is at random level and substantially worse than both Standard-LR (0.712) and Training-Free NLL (0.636).

### Per-Benchmark Breakdown

| Method | GSM8K PDA | GSM8K rho | MATH500 PDA | MATH500 rho |
|--------|-----------|-----------|-------------|-------------|
| Standard-LR (A) | 0.667 [0.560, 0.773] | 0.427 (p=0.167) | 0.864 [0.773, 0.939] | 0.874 (p=0.0002) |
| Tiny-LR (B) | 0.515 [0.394, 0.636] | 0.063 (p=0.846) | 0.818 [0.727, 0.909] | 0.846 (p=0.0005) |
| Difference (B-A) | -0.152 | -0.364 | -0.045 | -0.028 |

**MATH-500**: Tiny-LR performs nearly as well as Standard-LR (PDA 0.818 vs 0.864, CIs overlap). Both methods achieve high Spearman rho (>0.84, p<0.001).

**GSM8K**: Tiny-LR collapses to random (PDA=0.515, rho=0.063) while Standard-LR maintains above-chance performance (PDA=0.667).

### Root Cause: Response-Length Confound on GSM8K

The GSM8K failure is driven by a response-length confound:

| Dataset | Avg Response Length | Tiny-LR GSM8K Score | Target GSM8K Score | Direction Match |
|---------|-------------------|--------------------|--------------------|-----------------|
| QwQ-LongCoT | ~7K chars | 0.546 (rank 1) | 0.476 (rank 4) | No |
| Magpie-CoT | ~10K chars | 0.569 (rank 2*) | 0.356 (rank 6) | No |
| dart-math-hard | ~400 chars | 0.337 (rank 3) | 0.789 (rank 1) | No |

*By gsm8k_mean in proxy_tiny_v2_scores.csv

At tiny LR, the 1.5B proxy learns to pattern-match long reasoning chains on simpler GSM8K problems, inflating scores for long-response datasets. The 7B target model does not exhibit this bias because it actually learns math reasoning. This confound is less problematic for MATH-500 (harder problems where surface pattern-matching is less effective).

### Optimization Trajectory

| Configuration | PDA | Spearman rho | Status |
|--------------|-----|-------------|--------|
| Tiny-LR original (5e-6, 500 steps) | 0.349 | -0.378 | Inverted ranking |
| Tiny-LR optimized (1e-5, 1000 steps) | 0.500 | -0.091 | Random-level |
| Standard-LR baseline (5e-5, 500 steps) | 0.712 | 0.594 | Above chance |

The optimization (5x LR increase + 2x steps) improved PDA from 0.349 to 0.500 -- eliminating the ranking inversion but only reaching random level. Further optimization is unlikely to close the 0.212 gap to Standard-LR.

### Non-Degeneracy Check

Loss diagnostics for the optimized tiny-LR runs (from `proxy_tiny_v2_loss_diagnostics.csv`):
- Average loss drop: 22.1% (range: -27.1% to 46.8%)
- 33 of 36 runs show positive loss decrease
- 3 runs show loss increase (dart-math-hard seeds 123, 456 and numinamath1_5 seed 123), which are edge cases
- Conclusion: Training is non-degenerate; the proxy model does learn, but what it learns does not transfer to composite ranking

## Statistical Significance

### Composite PDA

- Standard-LR 95% CI: [0.606, 0.818]
- Tiny-LR 95% CI: [0.379, 0.621]
- CIs barely overlap at the edges (0.606-0.621), indicating the difference is near statistical significance
- The gap (0.712 - 0.500 = 0.212) exceeds the CI width of both methods

### Spearman Correlation

- Standard-LR: rho=0.594, p=0.042 (significant at alpha=0.05)
- Tiny-LR: rho=-0.091, p=0.779 (not significant; no ranking signal detected)
- NLL: rho=0.371, p=0.236 (not significant)

### MATH-500 Sub-Analysis

- Standard-LR MATH500: rho=0.874, p=0.0002
- Tiny-LR MATH500: rho=0.846, p=0.0005
- Both are highly significant and comparable -- the tiny-LR method works well for MATH-500 alone

## Decision Rule Application

Pre-registered criteria:

1. **Proceed** (PDA_tiny > PDA_std outside bootstrap CI, PDA_std > 0.55):
   - PDA_std = 0.712 > 0.55: YES
   - PDA_tiny = 0.500 < PDA_std = 0.712: **NO** -- criterion NOT met

2. **Pivot** (both above chance, |diff| <= 0.03):
   - |0.500 - 0.712| = 0.212 >> 0.03: **NO** -- criterion NOT met

3. **Refute** (PDA_tiny <= PDA_std, or frequent degenerate runs):
   - PDA_tiny (0.500) <= PDA_std (0.712): **YES** -- criterion MET
   - Non-degenerate training confirmed (avg 22.1% loss drop)

4. **Additional check** (PDA_tiny > PDA_nll):
   - PDA_tiny (0.500) < PDA_nll (0.636): **FAILED** -- tiny-LR does not even beat training-free baseline

**Decision: REFUTE**

## Verdict Justification

The verdict is **bad** based on the following evidence:

1. **Primary criterion refuted**: PDA(tiny)=0.500 <= PDA(standard)=0.712. The proposed tiny-LR method produces worse composite rankings than the standard-LR baseline.

2. **Worse than training-free baseline**: PDA(tiny)=0.500 < PDA(NLL)=0.636. The tiny-LR proxy does not even capture more ranking signal than a zero-cost NLL computation, making it strictly dominated.

3. **Random-level composite performance**: PDA=0.500 with Spearman rho=-0.091 (p=0.779) indicates no detectable ranking signal in the composite metric.

4. **Optimization plateau**: After tuning LR from 5e-6 to 1e-5 and doubling steps, PDA improved from 0.349 to 0.500 but plateaued at random level. The remaining gap (0.212) to the standard-LR baseline is too large to close with further hyperparameter tuning.

5. **Partial MATH-500 success is insufficient**: While MATH-500-only PDA (0.818) is competitive with standard-LR (0.864), this applies to only one of two benchmarks and does not rescue the composite metric.

6. **Identified failure mode**: At low learning rates, the 1.5B proxy model learns response-length patterns rather than math skills on easier benchmarks (GSM8K), creating a systematic ranking bias that does not match the 7B target's behavior. This is a fundamental limitation of the tiny-LR approach with small proxy models, not a bug.

**Conclusion**: The hypothesis that "tiny-LR proxy SFT improves dataset ranking transfer" is **refuted**. The standard-LR proxy SFT remains the best method for composite ranking, followed by training-free NLL. The tiny-LR approach should be reconsidered -- potential directions include using a larger proxy model, applying benchmark-specific weighting, or using MATH-500-only ranking where tiny-LR already works well.
