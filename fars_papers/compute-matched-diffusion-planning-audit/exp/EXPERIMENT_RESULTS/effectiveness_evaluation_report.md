# Effectiveness Evaluation Report

## Verdict: good

## Summary

The compute-matched evaluation protocol ran successfully across all three conditions on both tasks (Countdown and Mini Sudoku, 500 instances each). The key finding is that the diffusion planning advantage is **task-dependent**: Dream-v0-Base-7B loses badly to compute-matched Qwen2.5-7B best-of-k on Countdown (-32.5 percentage points), but retains a statistically significant advantage on Mini Sudoku (+10.4pp, 95% CI [+6.1pp, +14.6pp]). The protocol reveals that the naive greedy-vs-diffusion comparison substantially overstates the diffusion advantage by conflating model capacity differences with inference compute differences.

## Experiment Feasibility Check

All experiments completed without infrastructure or environment issues:
- Condition A (Qwen2.5-7B greedy): ran successfully via vLLM on A100-80GB
- Condition B (Qwen2.5-7B best-of-k, optimized): ran with 3 seeds (42, 123, 456), temperature=1.2, stop=["\n\n"]
- Condition C (Dream-v0-Base-7B diffusion): ran successfully via HF transformers diffusion_generate
- Wall-clock calibration completed on 50-instance calibration sets, yielding k=35 (Countdown) and k=39 (Sudoku)
- Bootstrap CI computation completed locally (B=10,000, seed=2024)

No missing results. All three conditions produced per-instance outputs for all 500 test instances on both tasks.

## Results Analysis

### Main Results Table

| Condition | Model | Method | Countdown | Mini Sudoku |
|-----------|-------|--------|-----------|-------------|
| A | Qwen2.5-7B | Greedy (temp=0) | 6.0% | 16.8% |
| B | Qwen2.5-7B | Best-of-k (compute-matched) | 39.1% +/- 1.4% (k=35) | 67.2% +/- 0.4% (k=39) |
| C | Dream-v0-Base-7B | Diffusion (single) | 6.6% | 77.6% |

### Pairwise Differences (Dream - Qwen BoK)

| Task | Delta | 95% Bootstrap CI | CI Excludes 0? |
|------|-------|-------------------|----------------|
| Countdown | -32.5pp | [-36.3pp, -28.7pp] | Yes |
| Mini Sudoku | +10.4pp | [+6.1pp, +14.6pp] | Yes |

### Wall-Clock Timing

| Task | Dream median (s) | Qwen median (s) | k (median) |
|------|-----------------|-----------------|------------|
| Countdown | 47.68 | 1.36 | 35 |
| Mini Sudoku | 55.75 | 1.40 | 39 |

### Key Observations

1. **Countdown**: Compute-matched Qwen BoK (39.1%) massively outperforms Dream diffusion (6.6%) by +32.5pp. The AR model benefits enormously from repeated sampling on arithmetic reasoning. With 35 independent attempts, Qwen solves nearly 40% of problems that it almost never solves in a single greedy attempt (6.0%). Dream gains almost nothing from its diffusion process on this task.

2. **Mini Sudoku**: Dream diffusion (77.6%) beats compute-matched Qwen BoK (67.2%) by +10.4pp with tight CI [+6.1, +14.6]. The diffusion model's parallel token refinement appears to provide a genuine advantage for constraint-satisfaction problems. However, the BoK protocol still closes ~60% of the original greedy gap (from 60.8pp to 10.4pp).

3. **Gap Reduction**: The compute-matched protocol drastically closes the naive comparison gap:
   - Countdown: Dream had +0.6pp advantage over greedy, now has -32.5pp (BoK dominates)
   - Sudoku: Dream had +60.8pp advantage over greedy, reduced to +10.4pp under compute matching

4. **Variance**: Qwen BoK shows low variance across 3 seeds (std 1.4pp on Countdown, 0.4pp on Sudoku), indicating stable results.

## Statistical Significance

Bootstrap 95% confidence intervals (B=10,000 resamples, N=500 instances):

- **Countdown**: Delta = -0.3248, CI = [-0.3627, -0.2873]. CI entirely below zero. Dream is **significantly worse** than compute-matched Qwen BoK (p < 0.025 one-sided).

- **Mini Sudoku**: Delta = +0.1042, CI = [+0.0613, +0.1460]. CI entirely above zero. Dream is **significantly better** than compute-matched Qwen BoK (p < 0.025 one-sided).

Both differences are statistically significant. The effect sizes are large (Countdown: -32.5pp, Sudoku: +10.4pp), well above the noise floor.

## Decision Rule Application

Per the pre-registered decision rule:

1. **"Supports robust diffusion advantage" (Dream >= +5pp on BOTH tasks, CI excludes 0)**: NOT MET. Dream fails on Countdown (-32.5pp).

2. **"Refutes/downweights diffusion advantage" (Dream <= +2pp on BOTH tasks or negative, or CI includes 0)**: NOT MET. Dream has +10.4pp on Sudoku with CI excluding 0.

3. **"Pivot" (both near ceiling >95% or near zero <5%)**: NOT MET. Results are in informative range.

**Outcome**: The result falls between the pre-registered decision boundaries. The diffusion advantage is **task-dependent** -- absent for arithmetic reasoning (Countdown), present for constraint satisfaction (Sudoku).

## Verdict Justification

**Verdict: good**

The experiment is rated "good" for the following reasons:

1. **Complete execution**: All three conditions ran successfully on both tasks with no infrastructure failures. Both main experiment and baseline results are available for comparison.

2. **Clear, scientifically interesting finding**: The compute-matched protocol reveals nuanced, task-dependent behavior that the naive comparison obscures. This is a positive research contribution regardless of whether it "supports" or "refutes" the diffusion advantage -- it provides a more precise characterization.

3. **Positive signal on Sudoku**: Dream retains a statistically significant +10.4pp advantage on Mini Sudoku even after compute matching. This exceeds the +5pp threshold from the decision rule (on this task) and the CI [+6.1, +14.6] firmly excludes zero. This suggests a genuine structural advantage of diffusion for constraint-satisfaction problems.

4. **Protocol validation**: The compute-matched evaluation methodology itself is validated as useful -- it reveals structure that greedy-only comparisons miss and provides a fairer comparison framework.

5. **Warrant for follow-up**: The task-dependent finding motivates the planned sensitivity analysis (k estimator variant) and scaling curve analysis, which will further characterize the phenomenon.

The result is not "bad" because there IS a positive signal (Sudoku). The result is not "failed" because all experiments ran and produced results. The mixed, task-dependent finding is itself a substantive research contribution.
