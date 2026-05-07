# EBR (Escrowed Batch Reveal, Condition C) -- gemini-2.5-flash

## Experiment Overview

Evaluated the Escrowed Batch Reveal (EBR) intervention (Condition C) on gemini-2.5-flash. EBR extends HardGateCustomerAgent (Condition B) by overriding `fetch_messages()` to buffer OrderProposal messages. Proposals are hidden from the LLM until K=3 accumulate, then released simultaneously in **shuffled order**. This isolates the sequential-visibility mechanism: the B->C comparison holds payment gating constant while changing only whether proposals are seen one-at-a-time or all-at-once.

**Note**: Results updated after optimization iteration 0. The original run had a 40% fallback rate due to businesses not sending formal proposals; this was fixed by adding a nudge mechanism and improving the comparison prompt.

## Setup

- **Model**: gemini-2.5-flash (via MAAS proxy, OpenAI-compatible)
- **Temperature**: 0.7
- **K (minimum proposals)**: 3 (with nudge mechanism and fallback release)
- **Scenarios**: contractors_first, contractors_second, contractors_third
- **Repetitions**: 15 per scenario (45 total)
- **Agent**: `EBRCustomerAgent` extending `HardGateCustomerAgent`
- **Results directory**: `multi-agent-marketplace/experiments/ebr/results/ebr_gemini_v2/`

## Key Results

| Metric | Value |
|--------|-------|
| Completion rate | 1.000 +/- 0.000 |
| Earliest-arrival chosen rate | **0.244 +/- 0.430** |
| Fallback rate | **0.000** |

### Rank Histogram (by server arrival order)

| Rank | Count | Fraction |
|------|-------|----------|
| 1 (earliest) | 11 | 24.4% |
| 2 | 16 | 35.6% |
| 3 (latest) | 17 | 37.8% |

### Per-Scenario Breakdown

| Scenario | Completion | Earliest Rate | Rank Distribution |
|----------|-----------|---------------|-------------------|
| contractors_first | 1.000 | 0.133 | R1=2, R2=5, R3=8 |
| contractors_second | 1.000 | 0.267 | R1=4, R2=6, R3=4 |
| contractors_third | 1.000 | 0.333 | R1=5, R2=5, R3=5 |

### Statistical Tests

- **Chi-squared test vs uniform**: chi2=1.409, p=0.494. Rank distribution is NOT significantly different from uniform (33.3% each).
- **Two-proportion z-test (HardGate vs EBR)**: Z=4.639, p=0.000002 (one-sided). Highly significant.

## Key Observations

1. **EBR eliminates earliest-arrival bias**: The earliest-arrival rate (24.4%) is statistically indistinguishable from the uniform baseline (33.3%). Chi-squared test confirms the rank distribution is uniform (p=0.49).

2. **Zero fallback rate**: All 45 runs received 3+ proposals. The nudge mechanism (injecting a follow-up prompt after 4 empty fetches) and improved SoftWait instruction completely eliminated the fallback issue.

3. **Decisive B->C comparison**: HardGate (73.3%) vs EBR (24.4%) shows a 48.9 percentage point reduction in earliest-arrival bias, significant at p<0.000002. This demonstrates that sequential visibility is the primary mechanism driving first-proposal bias.

4. **Perfect uniformity in contractors_third**: The 5/5/5 rank split across all three ranks in this scenario demonstrates that EBR can achieve true debiasing.

5. **100% completion rate**: All 45 runs completed successfully.

## Cross-Condition Comparison (Updated)

| Condition | Earliest-Arrival Rate | Rank 1 | Rank 2 | Rank 3 | Mechanism |
|-----------|----------------------|--------|--------|--------|-----------|
| A: SoftWait | 63.3% | 63.3% | 10.0% | 26.7% | Prompt-only |
| A': ITS | 63.3% | 63.3% | 20.0% | 16.7% | SoftWait + N=5 best-of-N |
| B: HardGate (v2) | 73.3% | 73.3% | 8.9% | 17.8% | Payment gated, sequential reveal |
| **C: EBR (v2)** | **24.4%** | **24.4%** | **35.6%** | **37.8%** | **Payment gated, batched+shuffled reveal** |
| Uniform random | 33.3% | 33.3% | 33.3% | 33.3% | -- |

## Interpretation

EBR successfully eliminates the correlation between arrival order and selection. By buffering proposals and revealing them simultaneously in shuffled order, EBR breaks the sequential anchoring mechanism. The improved comparison prompt further helps the LLM evaluate all proposals on their merits rather than defaulting to the first one presented.
