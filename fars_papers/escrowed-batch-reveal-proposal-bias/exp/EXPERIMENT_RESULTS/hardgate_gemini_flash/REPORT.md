# HardGate (Condition B) Baseline -- gemini-2.5-flash

## Experiment Overview

Evaluated the HardGate payment-gating baseline (Condition B) on gemini-2.5-flash. HardGate enforces a code-level constraint: the customer agent's `pay` action is blocked (returns `ACTION_UNAVAILABLE`) until `proposal_storage.count_proposals() >= K` (K=3). This ensures the agent has received at least 3 proposals before it can pay. Proposals are still revealed sequentially through `check_messages`. The same SoftWait prompt augmentation (Condition A) is used to avoid prompt confounds.

**Note**: Results updated after optimization iteration 0 with improved comparison prompt and increased sample size (15 reps per scenario).

## Setup

- **Model**: gemini-2.5-flash (via MAAS proxy, OpenAI-compatible)
- **Temperature**: 0.7
- **K (minimum proposals)**: 3
- **Scenarios**: contractors_first, contractors_second, contractors_third
- **Repetitions**: 15 per scenario (45 total)
- **Agent**: `HardGateCustomerAgent` extending `SoftWaitCustomerAgent`
- **Results directory**: `multi-agent-marketplace/experiments/ebr/results/hardgate_gemini_v2/`

## Key Results

| Metric | Value |
|--------|-------|
| Completion rate | 1.000 +/- 0.000 |
| Earliest-arrival chosen rate | **0.733 +/- 0.442** |

### Rank Histogram (Aggregate)

| Rank | Count | Fraction |
|------|-------|----------|
| 1 (earliest) | 33 | 73.3% |
| 2 | 4 | 8.9% |
| 3 (latest) | 8 | 17.8% |

### Per-Scenario Breakdown

| Scenario | Completion | Earliest Rate | Rank Distribution |
|----------|-----------|---------------|-------------------|
| contractors_first | 1.000 | 0.800 | R1=12, R2=3 |
| contractors_second | 1.000 | 0.467 | R1=7, R2=1, R3=7 |
| contractors_third | 1.000 | 0.933 | R1=14, R3=1 |

## Key Observations

1. **HardGate maintains strong first-proposal bias**: The earliest-arrival rate is 73.3%, confirming that sequential proposal reveal creates strong anchoring even when the LLM is explicitly told to compare all options.

2. **Sequential reveal drives the bias**: Despite having seen all 3 proposals before payment (enforced by code), the agent overwhelmingly selects the first-arriving proposal. The sequential `check_messages` loop means the first proposal enters the context earliest and is anchored upon.

3. **Decisive comparison with EBR**: HardGate (73.3%) vs EBR (24.4%) with Z=4.639, p=0.000002 provides the strongest evidence that sequential visibility is causal.

4. **contractors_third most extreme**: 93.3% earliest-arrival rate, showing the strongest anchoring bias.

## Comparison with Other Conditions

| Condition | Earliest-Arrival Rate | Mechanism |
|-----------|----------------------|-----------|
| A: SoftWait | 63.3% | Prompt-only |
| A': ITS | 63.3% | SoftWait + N=5 best-of-N |
| **B: HardGate (v2)** | **73.3%** | **Payment gated, sequential reveal** |
| C: EBR (v2) | 24.4% | Payment gated, batched reveal |
