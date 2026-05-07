# Inference-Time Scaling (ITS) Baseline on gemini-2.5-flash

## Experiment Overview

Condition A' (InferScale): SoftWait prompt + best-of-N=5 sampling at payment time. When the customer agent decides to pay, the LLM call is sampled 5 times and the sample selecting the lowest-price proposal is chosen. Non-payment actions use single-sample behavior.

This tests whether increased compute at the payment decision point can reduce first-proposal bias without protocol-level changes.

## Setup

- **Model**: gemini-2.5-flash (via MAAS proxy, OpenAI-compatible)
- **Temperature**: 0.7
- **N samples at payment**: 5
- **SoftWait K**: 3 (prompt instructs to wait for 3 proposals)
- **Scenarios**: contractors_first, contractors_second, contractors_third (3 scenarios x 10 reps = 30 runs)
- **Agent**: `ITSCustomerAgent` (extends `SoftWaitCustomerAgent`)
- **Implementation**: `experiments/ebr/agents/its_agent.py`

## Key Results

| Metric | ITS (this) | SoftWait (Cond. A) |
|--------|-----------|-------------------|
| Completion rate | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| Earliest-arrival chosen rate | 0.633 +/- 0.482 | 0.633 +/- 0.482 |

### Rank Histogram

| Rank | ITS | SoftWait |
|------|-----|----------|
| 1 (earliest) | 63.3% (19/30) | 63.3% (19/30) |
| 2 | 20.0% (6/30) | 10.0% (3/30) |
| 3 (latest) | 16.7% (5/30) | 26.7% (8/30) |

### Per-Scenario Breakdown

| Scenario | ITS earliest rate | SoftWait earliest rate |
|----------|------------------|----------------------|
| contractors_first | 0.20 | 0.50 |
| contractors_second | 0.70 | 0.60 |
| contractors_third | 1.00 | 0.80 |

## Key Observations

1. **Aggregate earliest-arrival rate is identical** (0.633) between ITS and SoftWait -- inference-time scaling at payment did not reduce first-proposal bias overall.

2. **Divergent per-scenario effects**: ITS dramatically reduced bias in `contractors_first` (0.50 -> 0.20) but worsened it in `contractors_third` (0.80 -> 1.00). This is because `contractors_third` is the scenario where the cheapest contractor arrives last -- when ITS selects the lowest-price option among those *already visible*, it may still pick the first-arrived proposal if it happens to be the cheapest one seen so far.

3. **ITS shifts rank-2 vs rank-3 distribution**: ITS chose rank-2 proposals more often (20% vs 10%) and rank-3 less often (16.7% vs 26.7%), suggesting the best-of-N mechanism tends to optimize price among early-arriving proposals rather than waiting for later ones.

4. **Completion rate unaffected**: 100% completion in both conditions -- the ITS mechanism does not degrade task success.

5. **Conclusion**: Inference-time scaling at payment is insufficient to reduce first-proposal bias. The bias originates from the agent's decision to pay *before* all proposals arrive, which N=5 sampling at that decision point cannot fix. Protocol-level interventions (HardGate, EBR) that change *when* the agent can pay are needed.
