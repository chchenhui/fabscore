# Condition C (Phoneme + Timing + Prosody) on VOCASET

## Experiment Overview

First training and evaluation of Condition C (MFA phoneme alignments + timing + prosody) on VOCASET. This completes the three-condition ablation on the second dataset.

## Setup

- **Frontend**: MFA forced alignment -> phoneme ID embedding (67 ARPAbet symbols) + within-phoneme position + duration + prosody (F0+energy) -> linear projection to 256-dim
- **Decoder**: frequency adaptor (50Hz -> 30fps) -> TCN decoder (5 blocks, kernel=3) -> 512-dim PCA output
- **Training**: Adam optimizer, lr=1e-4, cosine annealing to 1e-6, 300 epochs, batch_size=8, eval every 5 epochs
- **Data**: VOCASET synthetic (377 train, 48 val, 48 test), 12 speakers, vertex_scaling=0.5
- **MFA alignments**: `scaffoldswap/data/vocaset/mfa_alignments.json` (473 sequences, all aligned)

## Key Results

| Metric | Mean | Std | Seed 42 | Seed 123 | Seed 456 |
|--------|------|-----|---------|----------|----------|
| LVE x10^-5 | 0.87 | 0.02 | 0.89 | 0.88 | 0.84 |
| MVE | 4.17e-4 | 2.15e-6 | 4.14e-4 | 4.19e-4 | 4.17e-4 |
| UFVE | 3.62e-4 | 3.65e-6 | 3.57e-4 | 3.65e-4 | 3.65e-4 |
| FDD | 1.44e-5 | 2.60e-7 | 1.48e-5 | 1.42e-5 | 1.42e-5 |

Best epochs: all at 300. Best val losses: ~1.1e-6.

## Key Observations

1. Condition C (LVE 0.87) falls between A (1.35) and B (0.75) on VOCASET, unlike BIWI where C was best.
2. Condition B (discrete units) is the best-performing condition on VOCASET, outperforming both C and A.
3. The ranking on VOCASET (B > C > A) differs slightly from BIWI (C > B > A), suggesting dataset-dependent advantages.
4. Very low cross-seed variance (std ~2% of mean) indicates highly stable training for the phoneme condition.
5. All conditions show a clear improvement hierarchy over continuous SSL features (Condition A).
