# Condition A (SSL + Prosody) on VOCASET (v2: 300 epochs)

## Experiment Overview

Training and evaluation of the continuous SSL baseline (Condition A) for the ScaffoldSwap ablation study on the VOCASET dataset. Extended to 300 epochs with CosineAnnealingLR schedule (from v1: 100 epochs).

## Setup

**Architecture**: WavLM-base-plus (frozen, 768-dim at 50 Hz) + F0/energy prosody features (2-dim) -> linear projection to 256-dim -> frequency adaptor (50 Hz -> 30 fps) -> 5-block TCN motion decoder -> 512-dim PCA coefficients.

**Training**: Adam optimizer, lr=1e-4 with CosineAnnealingLR (eta_min=1e-6), 300 epochs, MSE loss on PCA coefficients with vertex scaling factor 0.5. Batch size 8. Seeds: [42, 123, 456]. Evaluation every 5 epochs, best checkpoint by validation loss.

**Data**: Synthetic paired data generated using the real PCA model from UniTalker's data release V1 (D1_vocaset) combined with LibriSpeech test-clean audio. 473 sequences (377 train, 48 val, 48 test) across 12 subjects.

## Key Results

| Metric | Mean | Std | Seed 42 | Seed 123 | Seed 456 |
|--------|------|-----|---------|----------|----------|
| LVE (x10^-5 m^2) | 1.35 | 0.06 | 1.37 | 1.35 | 1.33 |
| MVE    | 5.94e-4 | 1.25e-5 | 6.05e-4 | 6.01e-4 | 5.76e-4 |
| UFVE   | 5.76e-4 | 1.49e-5 | 5.88e-4 | 5.85e-4 | 5.55e-4 |
| FDD    | 2.24e-5 | 8.05e-7 | 2.33e-5 | 2.28e-5 | 2.12e-5 |

Best checkpoint epochs: all at 300. Best validation losses: 1.8-1.9 x10^-6.

### Improvement over v1 (100 epochs)

| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| LVE x10^-5 | 16.09 | 1.35 | -91.6% |
| MVE | 2.30e-3 | 5.94e-4 | -74.2% |
| UFVE | 2.47e-3 | 5.76e-4 | -76.7% |
| FDD | 1.13e-4 | 2.24e-5 | -80.2% |

## Key Observations

1. Massive improvement from extending training: 91.6% LVE reduction shows the model was severely under-trained at 100 epochs.
2. Cross-seed variance is very low (std ~4% of mean), indicating stable convergence.
3. LVE of 1.35 x10^-5 is closer to the UniTalker-B reference of 0.94 x10^-5 on real data.
