# SoftWait (Prompt-Only) Baseline on gemini-2.5-flash

## Experiment Overview

Condition A of the Escrowed Batch Reveal study: a prompt-only debiasing baseline that instructs the customer agent to wait for K=3 proposals before making a payment decision. No code-level enforcement is applied -- the agent retains full ability to pay at any time.

## Setup

- **Model**: gemini-2.5-flash (via MAAS proxy, OpenAI-compatible endpoint)
- **Temperature**: 0.7
- **Scenarios**: contractors_first, contractors_second, contractors_third (from data/position_bias/)
- **Repetitions**: 10 per scenario (30 total runs)
- **Agent**: SoftWaitCustomerAgent -- prepends instruction to wait for 3 proposals and compare before paying
- **Payment enforcement**: None (prompt-only)
- **Max steps**: 100

### SoftWait Prompt Injection

```
IMPORTANT: Before making any payment, you MUST wait until you have received at least 3 order proposals from different businesses. After receiving 3 proposals, carefully compare them on price, quality, and relevance before choosing which one to pay for. Do NOT pay for the first proposal you receive -- always wait and compare.
```

## Key Results

| Metric | Value |
|--------|-------|
| Completion rate | 1.000 +/- 0.000 |
| Earliest-arrival chosen rate | 0.633 +/- 0.482 |
| Rank 1 (first-arriving) chosen | 19/30 = 63.3% |
| Rank 2 chosen | 3/30 = 10.0% |
| Rank 3 chosen | 8/30 = 26.7% |

### Per-Scenario Breakdown

| Scenario | Completion | Earliest Rate | Rank 1 | Rank 2 | Rank 3 |
|----------|------------|---------------|--------|--------|--------|
| contractors_first | 1.000 | 0.500 | 5 | 1 | 4 |
| contractors_second | 1.000 | 0.600 | 6 | 1 | 3 |
| contractors_third | 1.000 | 0.800 | 8 | 1 | 1 |

## Key Observations

1. **Strong first-proposal bias persists**: Despite explicit instructions to wait and compare, the agent chose the first-arriving proposal 63.3% of the time (vs. 33.3% expected under no bias).
2. **100% completion rate**: The prompt instruction did not prevent the agent from completing transactions.
3. **Scenario variation**: contractors_third showed the highest first-proposal bias (80%), while contractors_first showed the lowest (50%).
4. **Prompt compliance is partial**: The agent often does wait for multiple proposals (as seen in logs showing "Waiting for at least 3 order proposals") but still tends to favor the first one received.
5. **Gate check PASSED**: earliest-arrival rate (0.633) > 0.50, confirming sufficient bias exists to study debiasing interventions.
