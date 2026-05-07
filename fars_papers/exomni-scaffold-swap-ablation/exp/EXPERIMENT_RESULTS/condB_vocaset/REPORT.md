# Condition B (Discrete Units + Prosody) on VOCASET (v2: 300 epochs)

## Experiment Overview

Trained and evaluated Condition B (HuBERT discrete speech units + prosody) on VOCASET across 3 seeds [42, 123, 456]. Extended to 300 epochs with CosineAnnealingLR schedule (from v1: 100 epochs).

## Setup

- **Frontend**: HuBERT-base-ls960 (frozen) -> k-means quantization (K=200, VOCASET-specific codebook) -> unit embedding (256-dim) -> prosody (F0+energy) concat -> linear projection
- **Decoder**: frequency adaptor (50Hz -> 30fps) -> TCN decoder (5 blocks, kernel=3) -> 512-dim PCA output
- **Training**: Adam optimizer, lr=1e-4, cosine annealing to 1e-6, 300 epochs, batch_size=8, eval every 5 epochs
- **Data**: VOCASET synthetic (377 train, 48 val, 48 test), 12 speakers, vertex_scaling=0.5

## Key Results

| Metric | Mean | Std | Seed 42 | Seed 123 | Seed 456 |
|--------|------|-----|---------|----------|----------|
| LVE x10^-5 | 0.75 | 0.03 | 0.76 | 0.77 | 0.71 |
| MVE | 3.99e-4 | 4.20e-6 | 4.04e-4 | 3.97e-4 | 3.95e-4 |
| UFVE | 3.43e-4 | 4.04e-6 | 3.49e-4 | 3.40e-4 | 3.40e-4 |
| FDD | 1.17e-5 | 5.41e-7 | 1.13e-5 | 1.24e-5 | 1.13e-5 |

Best epochs: 295-300. Best val losses: ~1.0e-6.

### Improvement over v1 (100 epochs)

| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| LVE x10^-5 | 12.55 | 0.75 | -94.0% |
| MVE | 2.13e-3 | 3.99e-4 | -81.3% |
| UFVE | 2.30e-3 | 3.43e-4 | -85.1% |
| FDD | 1.08e-4 | 1.17e-5 | -89.2% |

## Key Observations

1. Condition B outperforms Condition A on VOCASET: LVE 0.75 vs 1.35 (x10^-5), a 44% improvement. Consistent with BIWI.
2. LVE of 0.75 x10^-5 is below the UniTalker-B reference of 0.94 x10^-5, though on synthetic data.
3. Very low cross-seed variance (std ~3% of mean) indicates stable training.
4. 94% LVE reduction from v1 confirms the model was severely under-trained at 100 epochs.
