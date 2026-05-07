# FIR-CUSUM Optimization Report (Iteration 0)

## Experiment Overview

Optimized the CUSUM-epsilon controller by replacing the full reset (S_t=0) after rollback with a Fast Initial Response (FIR) partial reset (S_t = h * reset_fraction). This addresses the "reset blind spot" where the original CUSUM repeatedly accepted perturbed/corrupted steps because it had to re-accumulate evidence from scratch after each rollback.

## Issue Diagnosed

**Reset Blind Spot (Critical)**: After each rollback triggered by S_t > h, the original implementation reset S_t to 0. During ongoing perturbation, the CUSUM needed ~3 steps to climb back from 0 to h=14 before re-triggering. This caused an oscillating accept/reject pattern where ~6/10 perturbed steps were accepted (vs Or-epsilon's ~3.9/10). Each accepted perturbed step corrupted the model state further, causing escalating probe loss that persisted long after the perturbation window ended.

## Fix Applied

1. **FIR partial reset**: After rollback, set S_t = h * 0.5 instead of 0. This retains evidence of recent instability so the CUSUM re-triggers within 1-2 anomalous steps.
2. **S_0 = 0 at initialization**: The FIR head start is only applied after rollback, not at training start, to avoid early false alarms during the naturally volatile early training phase.
3. **Re-calibrated h**: With FIR reset in the calibration simulation, h=18.0 gives exact alarm rate=0.002 matching the target p_0.

## Setup

- Same model/data/perturbation as main experiment (ResNet-18/CIFAR-10)
- CUSUMController with h=18.0, k=0.5, reset_fraction=0.5, alpha=0.1
- mu_0=-0.0413, sigma_0=0.2260 (unchanged from calibration)
- 20 seeds x 3 conditions (step, ramp, nominal)

## Key Results

| Metric | Old CUSUM (h=14, S_t=0) | FIR-CUSUM (h=18, S_t=h/2) | Or-epsilon | Improvement |
|--------|------------------------|---------------------------|------------|-------------|
| Peak Excess (step) | 5.40 +/- 1.72 | 3.47 +/- 0.81 | 1.85 +/- 0.92 | -36% |
| Peak Excess (ramp) | 5.62 +/- 1.03 | 3.65 +/- 0.75 | 2.06 +/- 0.70 | -35% |
| Excess AUC (step) | 275.26 +/- 144.39 | 182.57 +/- 83.21 | 124.08 +/- 65.77 | -34% |
| Excess AUC (ramp) | 271.37 +/- 109.07 | 211.95 +/- 68.58 | 114.60 +/- 51.35 | -22% |
| Nominal rate | 0.0022 | 0.0020 | 0.0016 | exact match |
| False rollbacks (step) | 25.9 | 37.0 | 87.5 | +43% |
| False rollbacks (ramp) | 26.7 | 67.4 | 86.5 | +152% |

## Rate Matching

| Check | Value | Threshold | Status |
|-------|-------|-----------|--------|
| FIR-CUSUM vs p_0 | 0.0% relative error | 20% | PASS |
| Or vs p_0 | 20.0% relative error | 20% | PASS |
| FIR-CUSUM vs Or | 25.0% relative diff | 20% | MARGINAL FAIL |

## Key Observations

1. FIR partial reset substantially reduces peak excess loss (36% for step, 35% for ramp) by preventing the oscillating accept/reject pattern during perturbation.
2. Excess AUC also improves (34% for step, 22% for ramp), reflecting faster recovery after perturbation.
3. Nominal rollback rate exactly matches p_0=0.002, validating the calibration with S_0=0 initialization.
4. False rollbacks outside the perturbation window increase (37 vs 26 for step), because the FIR reset makes the CUSUM more aggressive after any rollback. However, the count remains well below Or-epsilon's 87.5.
5. The CUSUM still has higher peak excess and AUC than Or-epsilon, due to the inherent ~2-3 step detection delay of the sequential test vs Or-epsilon's single-step threshold.
