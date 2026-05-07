# No-Controller Baseline: ResNet-18 / CIFAR-10

## Experiment Overview

No-controller baseline establishing the unmitigated failure trajectory under gradient amplification perturbations. ResNet-18 trained on CIFAR-10 for 250 steps with AdamW, no accept/rollback mechanism. Three conditions: step perturbation (zeta=300 at steps 120-129), ramp perturbation (zeta ramping 1->300 at steps 120-129), and nominal (no perturbation).

## Setup

- **Model**: ResNet-18 (torchvision, from scratch, ~11M params)
- **Dataset**: CIFAR-10 (batch_size=128, standard normalization)
- **Optimizer**: AdamW (lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
- **Probe set**: 16 samples from CIFAR-10 test split, fixed per seed
- **Training steps**: 250
- **Seeds**: 20 (seeds 0-19)
- **Perturbation window**: Steps 120-129
- **EMA smoothing**: alpha=0.1 (computed but no rollback decisions made)
- **GPU**: 1x NVIDIA A100 via TrainService
- **WandB**: Logged offline, project=cusum-calibrated-rollback-controller

## Key Results

### Step Perturbation (zeta=300, constant at steps 120-129)

| Metric | Mean | Std |
|--------|------|-----|
| y_pre (baseline probe loss) | 1.4028 | 0.2738 |
| Peak excess loss | 16.9417 | 8.7685 |
| Excess AUC | 284.8689 | 150.0652 |
| False rollbacks outside window | 0.0 | 0.0 |
| Detection delay | N/A (no rollbacks) | - |

### Ramp Perturbation (zeta ramping 1->300 at steps 120-129)

| Metric | Mean | Std |
|--------|------|-----|
| y_pre (baseline probe loss) | 1.4175 | 0.2225 |
| Peak excess loss | 22.4195 | 15.0044 |
| Excess AUC | 409.3644 | 195.2601 |
| False rollbacks outside window | 0.0 | 0.0 |
| Detection delay | N/A (no rollbacks) | - |

### Nominal (no perturbation)

| Metric | Mean | Std |
|--------|------|-----|
| y_pre (baseline probe loss) | 1.4270 | 0.2404 |
| Nominal rollback fraction | 0.0 | 0.0 |

## Key Observations

1. **Severe unmitigated damage**: Without a controller, step perturbation causes peak excess loss of ~16.9 and ramp perturbation causes ~22.4, demonstrating the need for a rollback mechanism.
2. **Ramp > Step damage**: Ramp perturbation causes more severe damage than step, likely because the ramp reaches the same maximum amplification (300x) but also includes intermediate amplification levels that may compound differently.
3. **High variance across seeds**: Standard deviations are large relative to means (50-70% coefficient of variation), indicating seed-dependent initialization sensitivity.
4. **Nominal traces saved**: 20 nominal probe loss traces (shape [251]) saved for post-hoc calibration of Or-epsilon and CUSUM-epsilon controllers.
5. **No rollbacks by definition**: The no-controller condition always accepts, so rollback metrics are trivially zero.

## Files

- Aggregated metrics: `cusum_controller/results/no_controller/summary_metrics.json`
- Per-seed metrics: `cusum_controller/results/no_controller/per_seed_metrics.json`
- Full traces: `cusum_controller/results/no_controller/traces/*.npz`
- Nominal probe traces (for calibration): `cusum_controller/results/no_controller/nominal_probe_traces/seed*.npy`
