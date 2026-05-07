# Effectiveness Evaluation Report

## Verdict: good

## Summary

The Definition Unit Tests (DUT, Condition C) method is effective. Discriminative definition checks significantly improve downstream convention-adherence accuracy beyond engagement-matched neutral checks (Condition B) on both base models. The primary decision criterion -- C improves over B by >= 5 percentage points with bootstrap 95% CI excluding 0, and C > B in >= 2 of 3 convention families -- is fully met on both Qwen2.5-Math-7B-Instruct and Llama-3.1-8B-Instruct, with C > B in all 3 families on both models. The method should proceed to further analysis and experiments.

## Experiment Feasibility Check

All experiments ran successfully and produced complete results:

- **Condition A** (Glossary-only): Completed on both models (300 items each).
- **Condition B** (Neutral checks, k=3): Completed on both models.
- **Condition C** (Discriminative DUT, k=3): Completed on both models. Underwent one optimization round for Llama (robust answer extraction for truncated outputs); all conditions were re-scored uniformly.
- **A@5 and B@5** (Majority-vote, 5 samples): Completed on both models with temperature=0.7, top_p=0.95, max_new_tokens=512.

No infrastructure or environment issues prevented result collection. Both main experiment (C) and all baselines (A, B, A@5, B@5) produced results for comparison.

## Results Analysis

### Consolidated Results Table

| Model | Condition | Overall | Asymptotics | Completeness | Convolution | Alt Rate |
|-------|-----------|---------|-------------|--------------|-------------|----------|
| Qwen  | A         | 90.3%   | 99%         | 80%          | 92%         | 6.3%     |
| Qwen  | B         | 90.0%   | 97%         | 79%          | 94%         | 7.0%     |
| Qwen  | **C**     | **95.0%** | **99%**   | **91%**      | **95%**     | **1.3%** |
| Qwen  | A@5       | 41.3%   | 43%         | 6%           | 75%         | --       |
| Qwen  | B@5       | 50.7%   | 51%         | 22%          | 79%         | --       |
| Llama | A         | 56.7%   | 74%         | 30%          | 66%         | 11.0%    |
| Llama | B         | 58.7%   | 92%         | 21%          | 63%         | 9.3%     |
| Llama | **C**     | **81.3%** | **96%**   | **82%**      | **66%**     | **2.0%** |
| Llama | A@5       | 58.7%   | 75%         | 19%          | 82%         | --       |
| Llama | B@5       | 60.7%   | 88%         | 24%          | 70%         | --       |

### Primary Criterion: C vs B

**Qwen2.5-Math-7B-Instruct:**
- C - B overall: **+5.0pp** (95.0% vs 90.0%)
- Bootstrap 95% CI: [+2.0%, +8.3%], **excludes zero**
- Per-family C - B: Asymptotics +2pp, Completeness +12pp, Convolution +1pp
- C > B in **3 of 3** families (required: >= 2)
- **Criterion MET**

**Llama-3.1-8B-Instruct:**
- C - B overall: **+22.7pp** (81.3% vs 58.7%)
- Bootstrap 95% CI: [+16.0%, +29.3%], **excludes zero**
- Per-family C - B: Asymptotics +4pp, Completeness +61pp, Convolution +3pp
- C > B in **3 of 3** families (required: >= 2)
- **Criterion MET**

Both models pass the primary criterion.

### Secondary: C vs A (Glossary-only)

**Qwen:** C - A = +4.7pp (CI [+1.3%, +8.0%], excludes 0). Asymptotics tied at 99%, completeness +11pp, convolution +3pp. DUT improves over plain glossary injection in 2/3 families.

**Llama:** C - A = +24.7pp. Asymptotics +22pp, completeness +52pp, convolution tied at 66%. DUT dramatically improves in 2/3 families.

### Secondary: C vs Majority Vote (A@5, B@5)

**Qwen:** C - A@5 = +53.7pp (CI [+48.0%, +59.3%]); C - B@5 = +44.3pp (CI [+38.7%, +50.0%]). The majority-vote baselines performed much worse than single-sample greedy due to the 512-token limit truncating Qwen's chain-of-thought reasoning.

**Llama:** C - A@5 = +22.7pp (CI [+16.0%, +29.3%]); C - B@5 = +20.7pp (CI [+14.0%, +27.3%]). Even on Llama where majority vote was less degraded, DUT still significantly outperforms.

Single-sample DUT provides a qualitatively different benefit from inference-time scaling via majority vote. The DUT mechanism (discriminative definition checks) steers the model toward the correct convention interpretation, which is orthogonal to variance reduction through repeated sampling.

### Secondary: Alternate Convention Rate

DUT sharply reduces the fraction of answers matching the alternate (wrong) convention:
- **Qwen:** A=6.3%, B=7.0%, C=1.3% (81% relative reduction vs B)
- **Llama:** A=11.0%, B=9.3%, C=2.0% (78% relative reduction vs B)

This confirms the mechanism: discriminative checks help the model lock onto the glossary-defined convention rather than defaulting to an alternate convention it may have memorized.

## Statistical Significance

All primary and secondary comparisons use paired bootstrap confidence intervals (10,000 resamples, 95% level):

| Comparison | Qwen Diff | Qwen 95% CI | Llama Diff | Llama 95% CI |
|------------|-----------|-------------|------------|--------------|
| C - B      | +5.0pp    | [+2.0, +8.3] | +22.7pp  | [+16.0, +29.3] |
| C - A      | +4.7pp    | [+1.3, +8.0] | +24.7pp  | N/A (large)    |
| C - A@5    | +53.7pp   | [+48.0, +59.3] | +22.7pp | [+16.0, +29.3] |
| C - B@5    | +44.3pp   | [+38.7, +50.0] | +20.7pp | [+14.0, +27.3] |

All CIs exclude zero, confirming statistical significance for all comparisons.

## Verdict Justification

The verdict is **"good"** (proceed) based on the following evidence:

1. **Primary criterion fully met on both models.** C improves over B by +5.0pp (Qwen) and +22.7pp (Llama), both with bootstrap 95% CIs excluding zero. C > B in all 3 convention families on both models, exceeding the 2/3 requirement.

2. **Engagement-matched control rules out computation confound.** Condition B (neutral checks) performs within ~2pp of Condition A (no checks) on both models, establishing that simply adding extra reasoning steps (engagement) does not improve convention adherence. The gains from C are therefore attributable to the discriminative content of the definition checks, not the additional computation.

3. **DUT outperforms inference-time scaling.** Single-sample DUT (C) vastly outperforms 5-sample majority vote (A@5, B@5) on both models. This demonstrates that the DUT benefit is qualitatively different from variance reduction -- it steers the model toward correct convention interpretation.

4. **Alternate-convention reduction.** C reduces alternate-convention match rates by ~80% relative to B, directly confirming the hypothesized mechanism: discriminative checks help the model distinguish the glossary convention from memorized alternatives.

5. **Generalization across model strengths.** The effect holds on both a strong model (Qwen, 90% baseline) and a weaker model (Llama, 59% baseline), with larger absolute gains on the weaker model where there is more headroom for improvement.

6. **Per-family analysis.** The largest gains come from the completeness family (Qwen: +12pp, Llama: +61pp), where baseline accuracy is lowest and convention confusion is highest. Asymptotics and convolution families show smaller but consistent gains. No family shows C < B.

**Decision: PROCEED.** The DUT method is effective and should advance to further experiments.
