# FIR-CUSUM-epsilon Experiment Report

## Experiment Overview

Implemented and evaluated the FIR-CUSUM-epsilon (Fast Initial Response CUSUM) one-sided sequential test rollback controller on ResNet-18/CIFAR-10. The controller accumulates evidence of sustained upward drift in standardized innovations: `S_t = max(0, S_{t-1} + r_t - k)`, triggering rollback when `S_t > h`. The key innovation over the original CUSUM is the FIR partial reset: after rollback, `S_t` is set to `h * reset_fraction` instead of 0, enabling rapid re-triggering during sustained perturbation.

## Setup

- **Model**: ResNet-18 on CIFAR-10
- **Controller**: FIR-CUSUM-epsilon (one-sided CUSUM with partial reset)
- **Calibration**: h=18.0, calibrated with FIR reset dynamics on nominal traces
  - Exact match: alarm rate = 0.002 at h=18.0 (target p_0=0.002)
  - mu_0 = -0.0413, sigma_0 = 0.2260
  - k = 0.5, reset_fraction = 0.5
  - S_0 = 0 (no FIR head start at initialization)
  - S_reset = h * 0.5 = 9.0 (FIR partial reset after rollback)
- **EMA smoothing**: alpha = 0.1
- **Training**: 250 steps, AdamW (lr=1e-3, weight_decay=0.01), batch_size=128
- **Perturbation**: gradient amplification zeta=300, window at steps 120-129
- **Probe**: 16-sample fixed subset from CIFAR-10 test set
- **Seeds**: 20 seeds per condition
- **Conditions**: step perturbation, ramp perturbation, nominal (no perturbation)

## Key Results

| Metric | FIR-CUSUM (Step) | FIR-CUSUM (Ramp) | Or-eps (Step) | Or-eps (Ramp) | No-ctrl (Step) | No-ctrl (Ramp) |
|--------|-----------------|-----------------|---------------|---------------|----------------|----------------|
| Peak Excess Loss | 3.47 +/- 0.81 | 3.65 +/- 0.75 | 1.85 +/- 0.92 | 2.06 +/- 0.70 | 16.94 +/- 8.77 | 22.42 +/- 15.00 |
| Excess AUC | 182.57 +/- 83.21 | 211.95 +/- 68.59 | 124.08 +/- 65.77 | 114.60 +/- 51.35 | 284.87 +/- 150.07 | 409.36 +/- 195.26 |
| False Rollbacks | 37.0 +/- 38.0 | 67.4 +/- 44.1 | 87.5 +/- 44.6 | 86.5 +/- 40.0 | 0 | 0 |
| Detection Delay | -18.25 +/- 44.4 | -22.05 +/- 47.7 | -24.80 +/- 48.0 | -18.60 +/- 43.9 | N/A | N/A |

**Nominal rollback fraction**: 0.0020 +/- 0.0033 (target: 0.002, exact match)

## Improvement vs Original CUSUM (h=14, S_t=0 reset)

| Metric | Original CUSUM | FIR-CUSUM | Change |
|--------|---------------|-----------|--------|
| Peak Excess (step) | 5.40 +/- 1.72 | 3.47 +/- 0.81 | -36% |
| Peak Excess (ramp) | 5.62 +/- 1.03 | 3.65 +/- 0.75 | -35% |
| Excess AUC (step) | 275.26 +/- 144.39 | 182.57 +/- 83.21 | -34% |
| Excess AUC (ramp) | 271.37 +/- 109.07 | 211.95 +/- 68.59 | -22% |
| Nominal rate | 0.0022 | 0.0020 | exact match |

## Key Observations

1. **FIR partial reset eliminates the reset blind spot**: The original CUSUM reset S_t to 0 after rollback, requiring ~3 steps to re-trigger during ongoing perturbation. FIR resets to h/2=9.0, enabling re-triggering within 1-2 anomalous steps.

2. **Peak excess loss reduced by ~35%** vs original CUSUM for both perturbation types, narrowing the gap with Or-epsilon (ratio 1.88x vs 2.92x previously for step perturbation).

3. **Excess AUC reduced by 22-34%** vs original CUSUM, indicating faster recovery after perturbation.

4. **Exact nominal rate matching**: With S_0=0 initialization and calibrated h=18.0, the nominal rollback fraction exactly matches the target p_0=0.002.

5. **False rollback trade-off**: FIR-CUSUM has more false rollbacks outside the perturbation window than the original CUSUM (37 vs 26 for step), because the partial reset makes it more aggressive after any rollback. However, it still maintains fewer false rollbacks than Or-epsilon (37 vs 87.5 for step).

6. **CUSUM still has inherent detection delay**: Even with FIR, the CUSUM needs 2-3 steps to first trigger (accumulating from S_0=0), giving Or-epsilon a fundamental advantage in peak excess loss for sudden perturbations. This trade-off is inherent to sequential testing.
