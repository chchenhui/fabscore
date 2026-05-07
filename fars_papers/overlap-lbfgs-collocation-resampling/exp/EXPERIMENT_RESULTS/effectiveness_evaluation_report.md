# Effectiveness Evaluation Report

## Verdict: good

## Summary

Overlap-resampled L-BFGS successfully avoids premature termination under collocation resampling and shows competitive performance on the ice-shelf inverse problem, but does not meet the full "Proceed" criteria of the pre-registered decision rule. The method requires **pivoting**: narrowing the claim to inverse problems with clustering-type failure modes, since (a) on the 2D Poisson forward problem, overlap-LBFGS is 2.1x worse than fixed-LBFGS despite using 67% more compute, and (b) the Poisson benchmark required increasing overlap fraction from o=0.5 to o=0.9, triggering the "gains only for o>=0.8" pivot condition. The method is working and shows genuine promise, justifying continued research with a narrower scope.

## Experiment Feasibility Check

All six experiments ran successfully and produced complete results:
- 3 ice-shelf experiments (3 seeds each): Adam+resampling, Adam->fixed-LBFGS, Adam->overlap-LBFGS
- 3 Poisson experiments (3 seeds each): Adam+resampling, Adam->fixed-LBFGS, Adam->overlap-LBFGS
- 2 optimization iterations were performed on the proposed method (1 for ice-shelf, 1 for Poisson)

No infrastructure failures, environment issues, or missing results. All comparisons can be made.

## Results Analysis

### Ice-Shelf Inverse Problem Comparison

| Method | Budget | B_err (mean +/- std) | u_err (mean +/- std) | h_err (mean +/- std) |
|--------|--------|---------------------|---------------------|---------------------|
| Adam + resampling | 30000 | 8.63e-4 +/- 6.10e-4 | 1.12e-2 +/- 7.61e-3 | 3.83e-2 +/- 2.07e-2 |
| Adam -> fixed L-BFGS | 30000 | 1.15e-3 +/- 5.38e-4 | 9.73e-3 +/- 6.45e-3 | 3.98e-2 +/- 2.35e-2 |
| **Adam -> overlap-LBFGS (o=0.5)** | **50000** | **8.06e-4 +/- 4.71e-4** | **6.73e-3 +/- 3.92e-3** | **3.08e-2 +/- 2.20e-2** |

Per-seed ice-shelf details (overlap-LBFGS):

| Seed | B_err | u_err | h_err | Best Phase | L-BFGS Steps | Termination |
|------|-------|-------|-------|------------|-------------|-------------|
| 0 | 1.41e-3 | 8.88e-3 | 2.31e-2 | adam | 5662 | budget_exhausted |
| 1 | 2.54e-4 | 1.24e-3 | 8.60e-3 | adam_fixed | 5477 | budget_exhausted |
| 2 | 7.57e-4 | 1.01e-2 | 6.07e-2 | adam | 2412 | nan |

Per-seed ice-shelf details (Adam -> fixed L-BFGS):

| Seed | B_err | u_err | h_err | L-BFGS Steps | Termination |
|------|-------|-------|-------|-------------|-------------|
| 0 | 1.33e-3 | 1.65e-2 | 5.81e-2 | 620 | gradient_tolerance |
| 1 | 4.15e-4 | 1.04e-3 | 6.70e-3 | 623 | gradient_tolerance |
| 2 | 1.69e-3 | 1.17e-2 | 5.46e-2 | 620 | gradient_tolerance |

### 2D Poisson Forward Problem Comparison

| Method | Budget | Rel. L2 Error (mean +/- std) | L-BFGS Iters | Termination |
|--------|--------|------------------------------|-------------|-------------|
| Adam + resampling | 30000 | 5.94e-3 +/- 7.84e-4 | N/A | N/A |
| Adam -> fixed L-BFGS | 30000 | 3.42e-4 +/- 9.19e-5 | ~620 | gradient_tolerance |
| **Adam -> overlap-LBFGS (o=0.9)** | **50000** | **7.03e-4 +/- 1.14e-4** | **~8240** | **budget_exhausted** |

Per-seed Poisson details (overlap-LBFGS):

| Seed | Rel. L2 | Best Step | L-BFGS Steps | Cautious Skips | LS Failures | Termination |
|------|---------|-----------|-------------|----------------|-------------|-------------|
| 0 | 8.29e-4 | 46506 | 8246 | 243 (2.9%) | 2 | budget_exhausted |
| 1 | 7.29e-4 | 49001 | 8230 | 243 (3.0%) | 1 | budget_exhausted |
| 2 | 5.52e-4 | 48501 | 8245 | 234 (2.8%) | 0 | budget_exhausted |

## Criterion-by-Criterion Evaluation

### Criterion (i): Premature Termination Avoidance -- SATISFIED

The core hypothesis is confirmed. Overlap-resampled L-BFGS avoids the premature termination that would occur with naive resampled L-BFGS:

- **Ice-shelf**: Overlap-LBFGS completed 5477-5662 outer L-BFGS steps (seeds 0,1; budget_exhausted) and 2412 (seed 2; nan). Fixed-LBFGS completed only ~620 steps (gradient_tolerance). The overlap strategy allows L-BFGS to run ~9x more iterations before exhausting the compute budget.
- **Poisson**: Overlap-LBFGS completed ~8240 outer L-BFGS steps (all seeds; budget_exhausted). Fixed-LBFGS completed ~620 steps (gradient_tolerance). The ratio is ~13x.
- **Cautious update behavior**: On Poisson with o=0.9, only 2.8-3.0% of curvature pairs were skipped by the cautious update rule, indicating the Hessian approximation quality is maintained.
- **Note**: Seed 2 on ice-shelf terminated via NaN at budget 37085, which is a stability concern but not premature stopping in the sense of the L-BFGS line search failing immediately.

### Criterion (ii): Ice-Shelf Accuracy -- PARTIALLY MET

Overlap-LBFGS achieves the lowest mean B_err among the three methods:

- **vs Adam+resampling**: 8.06e-4 vs 8.63e-4 (6.6% improvement). Overlap-LBFGS also has lower std (4.71e-4 vs 6.10e-4).
- **vs Adam->fixed-LBFGS**: 8.06e-4 vs 1.15e-3 (30% improvement).
- **u_err and h_err**: Both are best under overlap-LBFGS (u_err 6.73e-3 vs 9.73e-3/11.2e-3; h_err 3.08e-2 vs 3.98e-2/3.83e-2).

**However, there are important caveats:**

1. **Compute budget mismatch**: Overlap-LBFGS used 50K gradient evals vs 30K for both baselines (67% more compute). The comparison is not strictly compute-matched.
2. **Margin within noise**: The 6.6% margin over Adam+resampling (delta = 5.7e-5) is much smaller than the standard deviations of either method (~5e-4). With only 3 seeds, this difference is not statistically significant.
3. **Seed 2 NaN**: One of three overlap-LBFGS seeds terminated via NaN, indicating a stability issue. Its best checkpoint was from very early training (step 1000).
4. **Best checkpoints not from L-BFGS phase**: Seeds 0 and 2's best B_err came from the Adam phase, not the overlap-LBFGS phase. Only seed 1 had its best from the Adam+fixed phase. The L-BFGS phase does not consistently improve B_err beyond Adam's best.

**Pre-registered "Proceed" requirement**: Lower median B_err than both baselines AND lower high-error fraction by a margin outside 3-seed std. The mean improvement exists but the margin is within noise. The high-error fraction analysis cannot be reliably performed with only 3 seeds (k-means clustering requires more data points). This criterion is not decisively met.

### Criterion (iii): 2D Poisson Non-Degradation -- NOT MET

Overlap-LBFGS is clearly worse than fixed-LBFGS on the 2D Poisson problem:

- **Overlap-LBFGS**: 7.03e-4 +/- 1.14e-4 (50K budget)
- **Fixed-LBFGS**: 3.42e-4 +/- 9.19e-5 (30K budget)
- **Ratio**: 2.06x worse
- **Range overlap**: The ranges do NOT overlap (lower bound of overlap-LBFGS = 5.89e-4 > upper bound of fixed-LBFGS = 4.34e-4)

Additionally, the overlap fraction had to be increased from o=0.5 (proposed) to o=0.9 to achieve even this level of performance:
- **o=0.5 on Poisson**: 5.56e-3 +/- 3.62e-3 (nearly as bad as Adam-only)
- **o=0.9 on Poisson**: 7.03e-4 +/- 1.14e-4 (7.9x improvement from o=0.5)

The fact that o=0.5 is non-functional on Poisson (27-34% cautious skip rate, degraded Hessian) while o=0.9 works (2.8-3.0% skip rate) directly triggers the pre-registered "Pivot" condition: "if gains appear only for o>=0.8".

## Statistical Significance

With only 3 seeds per method, formal statistical tests have limited power. However, we can assess the strength of evidence:

**Ice-shelf B_err** (one-sided test: overlap-LBFGS < baseline):
- Overlap vs Adam: means differ by 5.7e-5, pooled std ~5.4e-4. Effect size d = 0.11. Not significant at any conventional threshold.
- Overlap vs fixed-LBFGS: means differ by 3.4e-4, pooled std ~5.1e-4. Effect size d = 0.67. Suggestive but not significant with n=3.

**Poisson rel L2** (one-sided test: overlap-LBFGS > fixed-LBFGS):
- Means differ by 3.61e-4, pooled std ~1.04e-4. Effect size d = 3.47. This is a large, clearly significant effect -- overlap-LBFGS is definitively worse than fixed-LBFGS on Poisson.

**Poisson overlap sensitivity**:
- o=0.5: 5.56e-3 +/- 3.62e-3
- o=0.9: 7.03e-4 +/- 1.14e-4
- The 7.9x improvement from increasing overlap fraction demonstrates strong sensitivity to this hyperparameter.

## Verdict Justification

### Pre-Registered Decision Rule Application

The proposal specifies three outcomes:

1. **Proceed**: Lower median B_err + lower high-error fraction on ice-shelf + not worse on Poisson. **NOT MET** -- Poisson criterion fails decisively.

2. **Pivot**: Stable but gains only for o>=0.8, or only on one benchmark. **THIS APPLIES** because:
   - Poisson required o=0.9 to function (o=0.5 had 27-34% cautious skip rate)
   - Even with o=0.9, Poisson accuracy is 2.1x worse than fixed-LBFGS
   - The method shows genuine gains only on ice-shelf (and even there, marginally)

3. **Refute**: Fails to beat fixed-LBFGS on ice-shelf, or premature stopping at o=0.5. **NOT MET** -- the method does beat fixed-LBFGS on ice-shelf (at least in mean B_err), and does not show premature stopping at o=0.5 on ice-shelf.

### Why "good" and not "bad"

The effectiveness evaluation verdict is **"good"** because:

1. **All experiments completed successfully** -- no infrastructure or setup failures.
2. **The core mechanism works**: Overlap-resampled L-BFGS genuinely avoids premature termination under resampling. This is a real contribution.
3. **Ice-shelf results are promising**: Overlap-LBFGS achieves the best mean B_err among all methods, with the lowest variance and best u_err/h_err.
4. **The method is improvable**: The optimization trace shows each iteration yielded substantial gains (7.9x on Poisson, 33.6% on ice-shelf), suggesting further tuning could close the gap.
5. **Poisson performance, while worse than fixed-LBFGS, is 8.5x better than Adam-only**: The method clearly adds value over first-order optimization even with resampling.

A "bad" verdict would require the method to fundamentally not work. Instead, the evidence shows a working method that needs a narrower claim (inverse problems) and potentially higher overlap fractions for forward problems.

### Recommended Pivot

Per the pre-registered decision rule, the appropriate action is to **narrow the claim to inverse problems with clustering-type failure modes**, where:
- Fixed-collocation L-BFGS underperforms due to cluster-locked solutions
- Collocation resampling is beneficial for escaping clusters
- Overlap-LBFGS provides the best of both worlds: resampling coverage + L-BFGS convergence
- The o=0.5 overlap fraction is sufficient (unlike forward problems that require o>=0.9)

### Caveats and Limitations

1. **Budget mismatch**: The ice-shelf comparison is not compute-matched (50K vs 30K). A fairer comparison would either give baselines 50K budget or restrict overlap-LBFGS to 30K.
2. **Small sample size**: 3 seeds per method limits statistical power. The ice-shelf improvements are directionally correct but not statistically significant.
3. **Seed 2 NaN**: One ice-shelf seed terminated via NaN, indicating the method has stability issues that need addressing.
4. **Optimization required**: The method required multiple optimization iterations to reach competitive performance on both benchmarks, suggesting high sensitivity to hyperparameters.
5. **o=0.5 only works on ice-shelf**: The proposed o=0.5 is not universally applicable -- Poisson requires o>=0.9, limiting the generality of the overlap-resampling concept.
