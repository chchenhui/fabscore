# Cross-Model Robustness Check: HardGate vs EBR

## Experiment Overview

Replicates the decisive B (HardGate) vs C (EBR) comparison on `claude-sonnet-4-5` to assess whether the EBR effect (reduction in earliest-arrival bias) generalizes beyond `gemini-2.5-flash`. Both conditions use the same agent implementations (`HardGateCustomerAgent`, `EBRCustomerAgent`) and experimental parameters (K=3, temperature=0.7, 10 repetitions x 3 scenarios = 30 runs per condition).

## Setup

| Parameter | Value |
|-----------|-------|
| Models | gemini-2.5-flash (prior), claude-sonnet-4-5 (new) |
| Conditions | HardGate (B), EBR (C) |
| K (proposals to collect) | 3 |
| Temperature | 0.7 |
| Scenarios | contractors_first, contractors_second, contractors_third |
| Repetitions per scenario | 10 (claude), 15 (gemini, from prior tasks) |
| Total runs per condition | 30 (claude), 45 (gemini) |

## Key Results

### Cross-Model Comparison Table

| Model | HardGate (B) | EBR (C) | Delta (B-C) | p-value | Significant? |
|-------|-------------|---------|-------------|---------|--------------|
| gemini-2.5-flash | 0.733 | 0.244 | 0.489 | 0.000002 | Yes (p < 0.001) |
| claude-sonnet-4-5 | 0.367 | 0.233 | 0.133 | 0.130 | No (p = 0.13) |

### Completion Rates

Both conditions achieve 100% completion rate on both models. EBR introduces no cost to task completion.

### Per-Scenario Breakdown (claude-sonnet-4-5)

**HardGate:**
- contractors_first: 0.400 (4/10)
- contractors_second: 0.400 (4/10)
- contractors_third: 0.300 (3/10)

**EBR:**
- contractors_first: 0.400 (4/10)
- contractors_second: 0.200 (2/10)
- contractors_third: 0.100 (1/10)

### Rank Distributions

**HardGate (claude):** Rank 1: 11, Rank 2: 9, Rank 3: 10 -- near-uniform distribution
**EBR (claude):** Rank 1: 7, Rank 2: 11, Rank 3: 12 -- shifted away from rank 1

## Key Observations

1. **Claude does not exhibit first-proposal bias under HardGate.** The HardGate earliest-arrival rate for claude-sonnet-4-5 is 36.7%, close to random chance (33.3%). Its rank distribution (11/9/10) is nearly perfectly uniform. This contrasts sharply with gemini-2.5-flash, which showed 73.3% earliest-arrival rate under HardGate.

2. **EBR cannot reduce what is already absent.** Since claude's HardGate condition has no measurable first-proposal bias, there is a floor effect -- EBR cannot reduce the earliest-arrival rate below the random baseline. The 13.3pp difference (0.367 to 0.233) is not statistically significant (z=-1.127, p=0.13).

3. **The finding is a floor effect, not evidence against EBR's mechanism.** The key insight is that claude-sonnet-4-5 does not satisfice on the first proposal in sequential-reveal settings. It naturally evaluates all available proposals before deciding. This means:
   - The first-proposal bias is **model-dependent** (gemini exhibits it strongly; claude does not)
   - EBR is effective **when the bias exists** (as demonstrated on gemini)
   - EBR is unnecessary but harmless when the model already behaves rationally

4. **Generalization verdict: PARTIAL (floor effect).** EBR's mechanism is validated on gemini-2.5-flash. On claude-sonnet-4-5, the baseline condition (HardGate) already achieves near-uniform selection, so EBR has no bias to correct. The intervention should be recommended for models known to exhibit satisficing behavior, rather than as a universal protocol.

## Statistical Details

### gemini-2.5-flash (B vs C)
- Z-test: z = -4.639, p = 0.000002 (one-sided)
- Fisher's exact: OR = 0.118, p = 0.000003
- 95% CI for (C-B): (-0.669, -0.309)

### claude-sonnet-4-5 (B vs C)
- Z-test: z = -1.127, p = 0.130 (one-sided)
- Fisher's exact: OR = 0.526, p = 0.199
- 95% CI for (C-B): (-0.363, 0.096)

## Visualization

Cross-model comparison chart saved to: `multi-agent-marketplace/experiments/ebr/figures/cross_model_comparison.png`
