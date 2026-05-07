# Delta Ablation: Admission Control Margin Parameter (delta=0 vs delta=2)

## Experiment Overview

Ablates the admission control margin parameter delta to understand its impact on the defense's false acceptance rate and overall effectiveness. Compares delta=0 (no margin; R_gen selected if canary PA >= R_safe canary PA) versus the pre-registered delta=2 (R_gen must beat R_safe by >2pp on canary PA).

## Setup

- **Datasets**: BGL, Linux, HDFS
- **Seeds**: 42, 123, 456
- **Payloads**: D, E, F (all qualifying payloads)
- **Attack budgets**: k in {1, 3}
- **Total configs per delta**: 3 payloads x 2 k-values x 3 datasets x 3 seeds = 54
- **Delta semantics**: delta=0 uses >= (equal or better), delta=2 uses strict > (must beat by >2pp)
- **No new parsing required**: Reuses canary PA values and test predictions from C2 defense

## Key Results

### Overall Admission Decisions

| Delta | R_gen Deployed (False Acceptance) | R_safe Fallback | Mean Test PA | Mean PA Recovery |
|-------|-----------------------------------|-----------------|-------------|-----------------|
| 0     | 46/54 (85.2%)                     | 8/54 (14.8%)    | 0.2029      | 5.6%            |
| 2     | 33/54 (61.1%)                     | 21/54 (38.9%)   | 0.2008      | 6.1%            |

### Per-Dataset Breakdown

| Dataset | Delta | R_gen Rate | Mean PA | Mean Recovery |
|---------|-------|-----------|---------|---------------|
| BGL     | 0     | 18/18 (100%) | 0.3421 | 0.0% |
| BGL     | 2     | 18/18 (100%) | 0.3421 | 0.0% |
| Linux   | 0     | 10/18 (55.6%) | 0.1676 | 16.7% |
| Linux   | 2     | 8/18 (44.4%) | 0.1676 | 18.4% |
| HDFS    | 0     | 18/18 (100%) | 0.0989 | 0.0% |
| HDFS    | 2     | 7/18 (38.9%) | 0.0929 | 0.0% |

### Decision Flips (delta=0 vs delta=2)

13 configs flipped from r_safe (delta=2) to r_gen (delta=0). No configs flipped in the opposite direction.

| Config | r_gen_canary | r_safe_canary | delta0 PA | delta2 PA | PA Change |
|--------|-------------|---------------|-----------|-----------|-----------|
| D/k1/HDFS/seed_42 | 0.00 | 0.00 | 0.000 | 0.000 | +0.000 |
| D/k1/HDFS/seed_123 | 0.02 | 0.00 | 0.055 | 0.000 | +0.055 |
| D/k3/HDFS/seed_42 | 0.00 | 0.00 | 0.000 | 0.000 | +0.000 |
| D/k3/HDFS/seed_123 | 0.00 | 0.00 | 0.000 | 0.000 | +0.000 |
| E/k1/HDFS/seed_42 | 0.00 | 0.00 | 0.001 | 0.000 | +0.001 |
| E/k1/HDFS/seed_123 | 0.00 | 0.00 | 0.000 | 0.000 | +0.000 |
| E/k1/HDFS/seed_456 | 0.00 | 0.00 | 0.000 | 0.000 | +0.000 |
| E/k3/HDFS/seed_123 | 0.00 | 0.00 | 0.000 | 0.000 | +0.000 |
| F/k1/Linux/seed_123 | 0.16 | 0.14 | 0.110 | 0.105 | +0.005 |
| F/k1/HDFS/seed_42 | 0.00 | 0.00 | 0.004 | 0.000 | +0.004 |
| F/k1/HDFS/seed_123 | 0.00 | 0.00 | 0.049 | 0.000 | +0.049 |
| F/k3/Linux/seed_123 | 0.14 | 0.14 | 0.100 | 0.105 | -0.005 |
| F/k3/HDFS/seed_123 | 0.00 | 0.00 | 0.000 | 0.000 | +0.000 |

### False Acceptance Rate Comparison

- **delta=0**: 85.2% (46/54 configs deploy poisoned R_gen)
- **delta=2**: 61.1% (33/54 configs deploy poisoned R_gen)
- **Difference**: delta=2 reduces false acceptance by 24.1 percentage points (13 additional configs blocked)

## Key Observations

1. **Delta=0 dramatically increases false acceptance**: With no margin, 85.2% of poisoned rule sets pass the admission gate vs 61.1% at delta=2. The 2pp margin blocks 13 additional poisoned configs.

2. **Most flips occur on HDFS where both R_gen and R_safe have 0% canary PA**: 10 of 13 flips are on HDFS where both rule sets achieve PA=0 on the canary set. With delta=0 and >= semantics, ties go to R_gen (deploying the poisoned rules). With delta=2, ties go to R_safe (safe fallback). This is the primary mechanism by which delta=2 provides protection.

3. **BGL is unaffected by delta choice**: All 18 BGL configs select R_gen under both delta values because poisoned BGL rules maintain high canary PA (well above R_safe). The margin parameter has no effect here.

4. **Linux shows marginal difference**: Only 2 configs flip (F/k1/seed_123 and F/k3/seed_123), with negligible PA impact (~0.5pp). The Linux defense is primarily driven by cases where R_gen canary PA is well below R_safe, which are caught by both delta values.

5. **Overall PA and recovery are nearly identical**: Mean PA is 0.2029 (delta=0) vs 0.2008 (delta=2), and mean recovery is 5.6% vs 6.1%. The higher false acceptance rate of delta=0 does not translate to significantly worse outcomes because the flipped HDFS configs already have ~0% PA under both R_gen and R_safe (since R_safe also gets 0% on HDFS).

6. **Delta=2 provides conservative but justified protection**: While the practical impact is small for this specific R_safe, the 2pp margin correctly prevents deploying poisoned rules in ambiguous cases (especially ties). This is a sound design choice for production deployment where R_safe quality may vary.

7. **Delta=0 would NOT have led to significantly worse outcomes**: The 13 flipped configs show negligible aggregate PA change. However, this is specific to the current R_safe (which has 0% HDFS PA). With a better R_safe, the margin could matter more.
