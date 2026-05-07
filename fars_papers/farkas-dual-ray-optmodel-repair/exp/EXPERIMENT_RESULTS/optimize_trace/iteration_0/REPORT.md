# Optimization Iteration 0: Stronger Models + Higher N + Iterative Best-of-N

## Experiment Overview

Optimized the DualRay repair method with three major improvements:
1. **Stronger models**: Upgraded from Qwen2.5-7B-Instruct to Qwen2.5-72B-Instruct and Qwen3-32B
2. **Higher N=16**: Increased sampling from 6 to 16 candidates (1 greedy + 16 sampled at temp=0.7)
3. **Iterative repair (2 rounds)**: Re-diagnoses failed instances after Round 1 and retries with updated feedback

Retained from previous optimization: Condition C (weighted Farkas feedback), enhanced repair prompt, K=10.

## Setup

- **Models**: Qwen2.5-72B-Instruct (tp=4, 4 GPUs) and Qwen3-32B (tp=2, 2 GPUs)
- **Infeasible subset**: 31 instances (9 EasyLP, 22 ComplexLP) from attempt-0
- **K**: 10, **N**: 16 per round, **Rounds**: 2
- **Temperature**: 0.7, **Seed**: 42 (round 1), 1042 (round 2)
- **Round 2 logic**: For still-infeasible instances, re-diagnose and retry. For wrong-objective instances, fall back to attempt-0 feedback with a note about the previous failure.

## Key Results

### Repair Success Rate Comparison

| Method | Model | Repair (on 31) | Pass@1 Overall | Pass@1 EasyLP | Pass@1 ComplexLP |
|--------|-------|----------------|----------------|---------------|-----------------|
| IIS-TopK K=5 greedy (A) | 7B | 1/31 (3.23%) | 58.17% (502/863) | 71.32% (465/652) | 17.54% (37/211) |
| DualRay+Wt K=5 greedy (C) | 7B | 1/31 (3.23%) | 58.17% (502/863) | 71.32% (465/652) | 17.54% (37/211) |
| DualRay+Wt K=10 best-of-6 (C+) | 7B | 3/31 (9.68%) | 58.40% (504/863) | 71.63% (467/652) | 17.54% (37/211) |
| DualRay+Wt K=10 N=16 R=2 | Qwen3-32B | 6/31 (19.35%) | 58.75% (507/863) | 71.78% (468/652) | 18.48% (39/211) |
| **DualRay+Wt K=10 N=16 R=2** | **72B** | **7/31 (22.58%)** | **58.86% (508/863)** | **71.78% (468/652)** | **18.96% (40/211)** |

### Successfully Repaired Instances (72B)

| Instance | Round | Passed/Total | Notes |
|----------|-------|-------------|-------|
| EasyLP_246 | 1 | 17/17 | 100% success rate; was 0/6 with 7B |
| EasyLP_425 | 1 | 17/17 | 100% success rate |
| EasyLP_620 | 1 | 8/17 | 47% success rate |
| EasyLP_645 | 1 | 6/17 | 35% success rate |
| ComplexLP_62 | 1 | 2/17 | First ComplexLP repair! |
| ComplexLP_111 | 2 | 6/34 | Repaired in Round 2 (6/17 in R2) |
| ComplexLP_86 | 2 | 1/34 | Repaired in Round 2 (1/17 in R2) |

### Qwen3-32B Unique Repairs

| Instance | Round | Passed/Total | Not repaired by 72B |
|----------|-------|-------------|---------------------|
| EasyLP_99 | 1 | 5/17 | 72B: 0/34 still infeasible |
| EasyLP_289 | 1 | 4/17 | 72B: 0/34 wrong-objective |
| ComplexLP_211 | 2 | 1/34 | 72B: 0/34 error |

### Union of Repairs (Both Models Combined)

If we could cherry-pick best results per instance: **10/31 (32.26%)**
- 72B unique: EasyLP_246, EasyLP_620, ComplexLP_62, ComplexLP_86 (4 instances)
- Qwen3-32B unique: EasyLP_99, EasyLP_289, ComplexLP_211 (3 instances)
- Both: EasyLP_425, EasyLP_645, ComplexLP_111 (3 instances)

## Key Observations

1. **Model capacity is the dominant factor**: 72B achieves 7/31 vs 3/31 with 7B (2.3x improvement). EasyLP_246 went from 0/6 pass (7B) to 17/17 pass (72B).
2. **ComplexLP repairs now possible**: 72B repaired 3 ComplexLP instances (ComplexLP_62, 111, 86). Qwen3-32B also repaired ComplexLP_111 and ComplexLP_211. With 7B, zero ComplexLP repairs were achieved.
3. **Iterative repair (Round 2) adds value**: Both models gained additional repairs in Round 2 (72B: ComplexLP_111, ComplexLP_86; Qwen3-32B: ComplexLP_111, ComplexLP_211).
4. **Different models repair different instances**: 72B and Qwen3-32B have complementary strengths. The union covers 10/31 instances.
5. **High feasibility rate with 72B**: Many wrong-objective instances show 100% feasibility (17/17 or 34/34 feasible), indicating the 72B model reliably resolves infeasibility -- the bottleneck shifts to getting the objective correct.
