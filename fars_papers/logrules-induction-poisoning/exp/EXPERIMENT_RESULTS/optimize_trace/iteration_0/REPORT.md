# Optimization Iteration 0: Extend C1 and C2 to k={1,3,5,7} and Fix Baseline

## Experiment Overview

This optimization addresses issues in both the C1 poisoning evaluation and the C2 defense evaluation:

1. **C1**: Extended attack budgets from k={1,3} to k={1,3,5,7} and fixed C0 baseline mismatch bug
2. **C2**: Extended defense evaluation to k={1,3,5,7} to match the optimized C1

## Issues Fixed

### Issue 1: Insufficient Attack Budget in C1 and C2

Both C1 and C2 evaluations only tested k={1,3} (10-30% poisoning). Phase-0v2 showed k=5-7 (50-70%) are much more effective. Extended both to k={1,3,5,7}.

### Issue 2: C0 Baseline Mismatch in C1 Evaluation

`evaluate_c1.py` compared C1 predictions against `c0_clean` baseline (different pipeline session). Fixed to use `phase0v2_clean` (same session).

### Issue 3: C2 Missing Higher Attack Budgets

`evaluate_c2.py` only tested k={1,3}, missing the most effective attack configs. Extended to k={1,3,5,7}.

## Results

### C1 (Attack) Improvement

- **Before**: 0/18 configs with mean PA drop >= 5pp on 2/3 datasets
- **After**: 6/12 payload-k configs pass. Best: D_k7 on 3/3 datasets (BGL=9.9%, Linux=12.4%, HDFS=15.1%)

### C2 (Defense) Improvement

- **Before**: 21/54 (38.9%) configs trigger R_safe fallback, limited recovery evidence
- **After**: 46/108 (42.6%) configs trigger R_safe fallback
- **Key new finding**: D_k7 on Linux triggers R_safe in 3/3 seeds with 55.8% PA recovery

### Best Config: D_k7 Full Pipeline (C0 -> C1 -> C2)

| Dataset | C0 PA | C1 PA (attack) | C2 PA (defense) | C1 Drop | C2 Recovery |
|---------|-------|-----------------|-----------------|---------|-------------|
| BGL | 0.348 | 0.249 | 0.249 | 9.9% | 0% (undetected) |
| Linux | 0.173 | 0.050 | 0.105 | 12.4% | 55.8% |
| HDFS | 0.151 | 0.000 | 0.000 | 15.1% | 0% (R_safe=0) |

## Key Observations

1. Defense is most effective where the attack is strongest (Linux at k=5,7)
2. Higher k makes both the attack more effective AND the defense more likely to detect it
3. R_safe's low baseline PA limits recovery ceiling
4. No new GPU computation needed -- all artifacts reused from Phase-0v2
