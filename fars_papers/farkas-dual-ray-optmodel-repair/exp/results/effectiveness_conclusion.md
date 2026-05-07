# DualRayRank Effectiveness Conclusion

## 1. Primary Hypothesis Test: DualRay-TopK (B) vs IIS-TopK (A) on Truncation Regime

### 1.1 Practical Relevance Gate

**Result: FAILED (practical relevance refuted)**

The truncation regime (infeasible instances where IIS size > K=5) contains only **16 out of 31** infeasible instances. The 31 infeasible instances represent 3.59% of the 863-instance benchmark. The truncation regime thus covers only 16 instances -- well below the 50-instance threshold required for the practical relevance gate.

This means that even if dual-ray ranking produced a perfect repair on every truncated instance, the maximum possible end-to-end improvement would be 16/863 = 1.85 percentage points. In practice, the infeasibility rate is too low on this benchmark for the truncation bottleneck to be a meaningful factor.

### 1.2 B vs A Repair Success Rate (Truncation Regime)

Under controlled conditions (same model Qwen2.5-7B, same K=5, greedy decoding):

| Condition | Repair on Truncation (16) | Repair on All (31) |
|-----------|--------------------------|-------------------|
| IIS-TopK (A) | 0/16 (0.0%) | 1/31 (3.23%) |
| DualRay-TopK (B) | 0/16 (0.0%) | 1/31 (3.23%) |

**Result: Core claim REFUTED.** DualRay-TopK (B) shows zero improvement over IIS-TopK (A) on the truncation regime. Both methods achieve 0% repair rate on truncated instances and repair the identical single instance (EasyLP_425, iis_size=2, non-truncated).

### 1.3 End-to-End Impact

Both (A) and (B) yield identical Pass@1-after-<=2-attempts: **58.17%** overall. Delta = 0.00pp.

**Result: No end-to-end impact.**

### Primary Hypothesis Verdict: **REFUTED**

---

## 2. Secondary Comparisons

### 2.1 Condition C vs B (Weights vs Names Only)

Under K=5 greedy (controlled):
- C = B = 1/31 (3.23%). Adding normalized Farkas multiplier weights provides zero benefit.

Under K=10 best-of-6 (7B):
- C+ = 3/31 (9.68%) vs A+ (IIS K=10 best-of-6) = 2/31 (6.45%). Marginal +1 instance (EasyLP_620).
- However, this comparison confounds K increase (5->10), sampling (greedy->best-of-6), and prompt changes.

**Conclusion:** Under controlled conditions, weights add nothing. Under optimized conditions, there is a marginal signal (+1 instance) but heavily confounded.

### 2.2 Repair Methods vs Best-of-2

| Method | Pass@1 Overall |
|--------|---------------|
| Attempt-0 (no repair) | 58.05% |
| IIS-TopK (A) K=5 greedy 7B | 58.17% (+0.12pp) |
| DualRay-TopK (B) K=5 greedy 7B | 58.17% (+0.12pp) |
| DualRay+Wt (C+) K=10 best-of-6 7B | 58.40% (+0.35pp) |
| DualRay+Wt (C+) K=10 N=16 R=2 72B | 58.86% (+0.81pp) |
| **Best-of-2** | **65.12% (+7.07pp)** |

**Result: Best-of-2 dominates all repair methods by a large margin (~6pp).** Simple inference scaling is far more effective than solver-feedback-guided repair. Even the most optimized repair condition (72B, K=10, N=16, 2 rounds) achieves only 58.86%, which is 6.26pp below Best-of-2 (65.12%).

This is particularly notable because the Best-of-2 baseline uses the same 7B model and only 2 samples, while the best repair method uses a 72B model and up to 34 samples (17 per round x 2 rounds) plus solver overhead.

### 2.3 EasyLP vs ComplexLP

Under the original controlled conditions (7B, K=5, greedy), zero ComplexLP instances were repaired by any method. All 16 truncation regime instances are ComplexLP, and all remain unrepaired.

Under optimized conditions (72B, K=10, N=16, R=2), 3 ComplexLP instances were repaired:
- ComplexLP_62 (iis_size=4, not truncated)
- ComplexLP_111 (iis_size=1, not truncated)
- ComplexLP_86 (iis_size=6, truncated) -- the only truncation regime repair across all experiments

This suggests that model capacity (72B vs 7B) is the dominant factor for ComplexLP repair, not the feedback method.

---

## 3. Overall Assessment

### Primary Hypothesis: **REFUTED**

DualRay-TopK does not outperform IIS-TopK on the truncation regime under controlled comparison. The evidence is unambiguous:
- 0/16 vs 0/16 on the truncation regime
- 1/31 vs 1/31 overall (same instance repaired)
- 0.00pp difference in end-to-end Pass@1

### Practical Relevance: **WEAK**

Only 16 instances fall in the truncation regime (< 50-instance gate). The infeasibility rate itself is low (3.59%), making the truncation bottleneck a minor factor on this benchmark.

### Repair vs Inference Scaling: **Repair loses**

Best-of-2 (65.12%) outperforms all repair methods (max 58.86%) by ~6pp, using a weaker model and fewer compute resources.

### Positive Signals (from Optimization)

The optimization phase revealed that model capacity is the dominant factor. With 72B + high N + iterative repair, 7/31 instances can be repaired (22.58%). However:
1. These improvements are confounded with model size, K, N, and iterative rounds.
2. Only 1/16 truncation regime instances was repaired even under these conditions.
3. The 72B repair pipeline still underperforms 7B Best-of-2 by 6pp.

### Verdict

The DualRayRank method -- ranking IIS constraints by Farkas dual ray multipliers -- does not provide meaningful improvement over standard IIS-TopK feedback for LLM-based LP repair. The core hypothesis that dual-ray-based ranking helps in the truncation regime is refuted by the data. Solver feedback repair in general is dominated by simple inference scaling on this benchmark.
