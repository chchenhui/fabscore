# Reveal-Position Bias Analysis for EBR (Condition C)

## Experiment Overview

This analysis examines whether Escrowed Batch Reveal (EBR, Condition C) truly eliminates ordering bias or merely shifts it from arrival-order to list-position (primacy) bias. When EBR reveals K=3 proposals simultaneously in a shuffled order, a new confound arises: the LLM may exhibit primacy bias toward the first proposal in the rendered list, regardless of arrival time.

## Setup

- **Model**: gemini-2.5-flash (temperature=0.7)
- **Data**: 45 EBR runs across 3 scenarios (contractors_first/second/third x 15 runs)
- **Valid runs**: 44 (1 excluded: paid proposal was outside the revealed batch of 3)
- **Data source**: EBR shuffle order logs from SQLite databases + payment actions
- **Analysis scripts**: `experiments/ebr/analysis/extract_reveal_order.py`, `experiments/ebr/analysis/reveal_order_analysis.py`

## Key Results

### Reveal-Position Selection Rates

| Reveal Position | Count (of 44) | Rate | 95% Wilson CI |
|----------------|---------------|------|---------------|
| 1st (first in shuffled list) | 32 | **72.7%** | (58.2%, 83.7%) |
| 2nd | 5 | 11.4% | (5.0%, 24.0%) |
| 3rd | 7 | 15.9% | (7.9%, 29.4%) |

### Statistical Tests

- **One-sided binomial test** (H0: p1 = 1/3, H1: p1 > 1/3): p = 1.1e-07 (highly significant)
- **Chi-squared vs uniform**: chi2 = 30.864, p < 1e-06 (distribution is far from uniform)

### Comparison: Arrival-Order vs. Reveal-Position Bias

| Metric | HardGate (B) | EBR (C) |
|--------|-------------|---------|
| 1st-arrival chosen rate | **73.3%** | 24.4% (near uniform) |
| 1st-reveal-position chosen rate | N/A | **72.7%** |
| Interpretation | Strong arrival-order bias | Arrival-order bias eliminated, but list-position bias persists |

## Key Observations

1. **EBR successfully eliminates arrival-order bias**: The arrival-rank distribution under EBR is approximately uniform (prior chi-squared p = 0.49), confirming that shuffling effectively decouples payment decisions from proposal arrival order.

2. **Bias has shifted, not disappeared**: The 1st proposal in the shuffled list is selected 72.7% of the time -- virtually identical to HardGate's 73.3% first-arrival selection rate. The LLM exhibits a fundamental primacy bias toward whatever proposal appears first in its context window, regardless of how that ordering was determined.

3. **This triggers the Pivot branch**: Per the pre-registered decision rule, a 1st-reveal-position rate >= 0.50 indicates that bias has shifted from arrival-order to list-position. Sequential visibility was indeed causal for arrival-order bias (the EBR mechanism works), but the underlying LLM primacy preference is a deeper confound.

4. **EBR still provides fairness value**: Even though list-position bias persists, EBR randomizes which proposal ends up in the privileged first position. Over many transactions, each business has an equal probability of being listed first, creating statistical fairness even if per-transaction bias exists.

## Recommendations

To fully eliminate ordering bias in individual transactions:

1. **Forced comparison step**: After revealing all proposals, require the LLM to explicitly compare each pair before making a final selection.
2. **Rotation aggregation**: Present proposals in multiple orderings across separate LLM calls, then aggregate the rankings.
3. **Independent scoring**: Have the LLM score each proposal independently (one at a time) and select the highest-scored proposal.
4. **Position-aware prompting**: Explicitly instruct the LLM to evaluate proposals regardless of their position in the list.

## Decision Rule Application

- 1st-reveal-position rate = 72.7% >= 0.50 threshold
- **Decision: PIVOT** -- Sequential visibility was causal for arrival-order bias, but generic primacy persists
- EBR provides statistical fairness (randomized first position) but not per-transaction debiasing
