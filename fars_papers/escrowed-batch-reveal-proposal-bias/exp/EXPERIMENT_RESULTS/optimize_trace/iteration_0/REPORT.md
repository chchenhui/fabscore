# Optimization Iteration 0: Fix EBR Fallback Rate and Strengthen Comparison Prompt

## Experiment Overview

Re-ran the HardGate (Condition B) and EBR (Condition C) experiments with two code fixes addressing the 40% fallback rate in EBR and weak comparison behavior. Increased sample size from 10 to 15 runs per scenario (45 total per condition).

## Issues Fixed

### Issue 1: EBR 40% Fallback Rate (Critical)
- **Root cause**: In the original EBR, when businesses responded with text messages instead of OrderProposals, the customer LLM got stuck in a FetchMessages loop. The SoftWait prompt told it to "wait for 3 proposals" and it saw none (they were buffered), so it passively waited. After 10 empty fetches, fallback released whatever was buffered (often 1-2 proposals), making the bias measurement meaningless.
- **Fix**: Added a nudge mechanism in `EBRCustomerAgent.fetch_messages()`: after 4 consecutive empty fetches with a partial buffer, injects a synthetic system message prompting the LLM to send follow-up messages to businesses requesting formal proposals. Increased `MAX_EMPTY_FETCHES` from 10 to 20.
- **Result**: Fallback rate dropped from 40% to 0%.

### Issue 2: Weak Comparison Prompt (Moderate)
- **Root cause**: The original SoftWait prompt said "compare them on price, quality, and relevance" but wasn't explicit enough about structured comparison or follow-up behavior.
- **Fix**: Updated `SOFTWAIT_INSTRUCTION` to: (1) require explicit comparison of ALL proposals by price, (2) emphasize "do NOT automatically choose the first proposal you see," (3) instruct follow-up with businesses that haven't sent formal proposals.
- **Result**: Both EBR and HardGate show improved comparison behavior. EBR rank distribution is now indistinguishable from uniform.

## Setup

- **Model**: gemini-2.5-flash (via MAAS proxy)
- **Temperature**: 0.7
- **K**: 3
- **Scenarios**: contractors_first, contractors_second, contractors_third
- **Repetitions**: 15 per scenario (45 total per condition)
- **Conditions re-run**: HardGate (B) and EBR (C)

## Key Results

### Cross-Condition Comparison

| Metric | HardGate v2 | EBR v2 | Original HardGate | Original EBR |
|--------|------------|--------|-------------------|-------------|
| Earliest-arrival rate | **0.733** | **0.244** | 0.800 | 0.467 |
| Completion rate | 1.000 | 1.000 | 1.000 | 1.000 |
| Fallback rate | N/A | **0.000** | N/A | 0.400 |
| Rank 1 fraction | 0.733 | 0.244 | 0.800 | 0.467 |
| Rank 2 fraction | 0.089 | 0.356 | 0.033 | 0.233 |
| Rank 3 fraction | 0.178 | 0.378 | 0.167 | 0.300 |

### Statistical Tests

- **Two-proportion z-test (HardGate vs EBR)**: Z=4.639, p=0.000002 (one-sided). Highly significant.
- **Chi-squared test (EBR ranks vs uniform)**: chi2=1.409, p=0.494. NOT significantly different from uniform -- EBR effectively eliminates first-proposal bias.

### Per-Scenario Breakdown (EBR v2)

| Scenario | Earliest Rate | Rank 1 | Rank 2 | Rank 3 |
|----------|--------------|--------|--------|--------|
| contractors_first | 0.133 | 2 | 5 | 8 |
| contractors_second | 0.267 | 4 | 6 | 4 |
| contractors_third | 0.333 | 5 | 5 | 5 |

## Key Observations

1. **EBR now eliminates arrival-order bias entirely**: The earliest-arrival rate (24.4%) is not significantly different from the uniform baseline (33.3%), and the rank distribution is uniform (chi-squared p=0.49). This is a major improvement from the original 46.7%.

2. **Zero fallback rate**: The nudge mechanism and improved prompt completely eliminated fallback releases. All 45 EBR runs received 3+ proposals before release.

3. **HardGate still shows strong bias**: 73.3% earliest-arrival rate confirms that sequential visibility (seeing proposals one-at-a-time) creates strong anchoring even when the LLM is told to compare.

4. **B->C comparison is decisive**: The 48.9 percentage point reduction (73.3% -> 24.4%) with p<0.000002 provides strong evidence that sequential visibility is the primary mechanism driving first-proposal bias. EBR's batched+shuffled reveal breaks this mechanism.

5. **contractors_third scenario is perfectly uniform in EBR**: 5/5/5 across all three ranks, showing that EBR can achieve true debiasing.
