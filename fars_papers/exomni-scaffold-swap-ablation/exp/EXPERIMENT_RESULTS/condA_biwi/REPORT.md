# Condition A (SSL + Prosody) Baseline on BIWI

## Experiment Overview

Training and evaluation of the continuous SSL baseline (Condition A) for the ScaffoldSwap ablation study on the BIWI dataset. This replicates the UniTalker-Base-[D0] audio frontend architecture (WavLM-base-plus) within a controlled framework.

## Setup

**Architecture**: WavLM-base-plus (frozen, 768-dim at 50 Hz) + F0/energy prosody features (2-dim) -> linear projection to 256-dim -> frequency adaptor (50 Hz -> 25 fps) -> 5-block TCN motion decoder -> 512-dim PCA coefficients.

**Training**: Adam optimizer, lr=1e-4 with CosineAnnealingLR (eta_min=1e-6), 300 epochs, MSE loss on PCA coefficients with BIWI vertex scaling factor 0.2. Batch size 8. Seeds: [42, 123, 456]. Evaluation every 5 epochs, best checkpoint by validation loss.

**Data**: Synthetic paired data generated using the real PCA model from UniTalker's data release V1 combined with LibriSpeech test-clean audio. 238 sequences (190 train, 24 val, 24 test).

**Model Parameters**: 1,398,400 trainable / 95,780,336 total (including frozen WavLM).

## Key Results

| Metric | Mean | Std | Seed 42 | Seed 123 | Seed 456 |
|--------|------|-----|---------|----------|----------|
| LVE    | 0.003771 | 0.000062 | 0.003771 | 0.003843 | 0.003699 |
| MVE    | 0.001348 | 0.000010 | 0.001345 | 0.001356 | 0.001343 |
| UFVE   | 0.001278 | 0.000012 | 0.001277 | 0.001290 | 0.001267 |
| FDD    | 5.88e-6  | 4.59e-7  | 5.39e-6  | 6.33e-6  | 5.91e-6  |

Best checkpoint epochs: all at 300. Best validation losses: ~4e-6.

## Key Observations

1. Significant improvement over v1 (100 epochs, no scheduler): LVE reduced 37.3% from 0.006015 to 0.003771.
2. All 3 seeds converged smoothly. Cross-seed variance reduced compared to v1.
3. Training speed: ~5 seconds per epoch on a single GPU, ~25 minutes per seed for 300 epochs.
