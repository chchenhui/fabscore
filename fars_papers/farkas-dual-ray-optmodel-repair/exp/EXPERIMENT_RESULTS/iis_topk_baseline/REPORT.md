# IIS-TopK Repair Baseline (Condition A)

## Experiment Overview

Evaluate the IIS-TopK repair baseline on the 31 infeasible instances from attempt-0. For each infeasible instance, compute the IIS on the LP relaxation using HiGHS, sort by constraint name alphabetically, truncate to top K=5, and provide as feedback in a repair prompt. The LLM generates a corrected `.lp` file in one repair attempt.

## Setup

- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Decoding**: Greedy (temperature=0.0)
- **Benchmark**: MAMO-Optimization (863 instances: 652 EasyLP + 211 ComplexLP)
- **Infeasible subset**: 31 instances (9 EasyLP + 22 ComplexLP)
- **K**: 5 (max IIS constraints shown)
- **Solver**: HiGHS (presolve=off, iis_strategy=2)
- **GPU**: 1x GPU via TrainService

## Key Results

| Metric | Value |
|--------|-------|
| Repair successes | 1/31 (3.23%) |
| Repair success (on repairable) | 1/31 (3.23%) |
| Pass@1 after <=2 attempts (Overall) | 58.17% (502/863) |
| Pass@1 after <=2 attempts (EasyLP) | 71.32% (465/652) |
| Pass@1 after <=2 attempts (ComplexLP) | 17.54% (37/211) |

### Comparison with Attempt-0

| Metric | Attempt-0 | After IIS-TopK Repair | Delta |
|--------|-----------|----------------------|-------|
| Overall pass@1 | 58.05% | 58.17% | +0.12pp |
| EasyLP pass@1 | 71.17% | 71.32% | +0.15pp |
| ComplexLP pass@1 | 17.54% | 17.54% | +0.00pp |

### Repair Status Distribution

| Status | Count |
|--------|-------|
| fail-infeasible | 22 |
| fail-wrong-objective | 4 |
| fail-error | 4 |
| pass | 1 |

### IIS Size Statistics

- Min: 1, Max: 33, Mean: 7.0
- Truncated (IIS > K=5): 16/31 (51.6%)
- The single successful repair (EasyLP_425) had IIS size 2 (not truncated)

## Key Observations

1. IIS-TopK repair is largely ineffective: only 1 out of 31 infeasible instances was repaired successfully.
2. The repair mostly fails to resolve infeasibility (22/31 remain infeasible after repair).
3. 16 out of 31 instances (51.6%) have IIS > K=5, meaning the feedback is truncated. This is the regime where dual-ray ranking could potentially help by selecting more informative constraints.
4. The single success was a non-truncated EasyLP instance, suggesting the model struggles with complex infeasibility patterns even with full IIS feedback.
5. No ComplexLP instances were repaired, consistent with the difficulty of these problems.
