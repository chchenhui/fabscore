# Condition C (Phoneme + Timing + Prosody) on BIWI

## Experiment Overview

Condition C tests the hypothesis that explicit phoneme+timing features from forced alignment provide an effective temporal scaffold for facial animation, matching or exceeding discrete speech units (Condition B) and continuous SSL features (Condition A).

## Setup

- **Frontend**: MFA forced alignment -> phoneme ID embedding (67 ARPAbet symbols, nn.Embedding(67, 256)) + within-phoneme position p in [0,1] + phoneme duration d + F0/energy prosody -> Linear(260, 256)
- **Decoder**: FrequencyAdaptor (50Hz->25fps) -> TCN decoder (5 blocks, kernel_size=3) -> 512-dim PCA output
- **Training**: Adam optimizer, lr=1e-4 with CosineAnnealingLR (eta_min=1e-6), 300 epochs, batch_size=8, grad_clip=1.0, vertex_scaling=0.2
- **Eval every**: 5 epochs, best checkpoint by val_loss
- **Seeds**: [42, 123, 456]
- **Trainable params**: 1,284,992 (all trainable, no frozen encoder)

## Key Results

| Metric | Mean | Std | Seed 42 | Seed 123 | Seed 456 |
|--------|------|-----|---------|----------|----------|
| LVE | 0.003121 | 0.000023 | 0.003130 | 0.003145 | 0.003089 |
| MVE | 0.001219 | 0.000006 | 0.001219 | 0.001226 | 0.001211 |
| UFVE | 0.001101 | 0.000008 | 0.001101 | 0.001111 | 0.001092 |
| FDD | 4.71e-6  | 1.16e-7  | 4.77e-6  | 4.80e-6  | 4.54e-6  |

Best epochs: all at 300. Best val losses: ~4.0-4.2e-6.

## Comparison Across All Conditions (v2: 300 epochs + cosine LR)

| Metric | Cond A (SSL) | Cond B (Units) | Cond C (Phoneme) |
|--------|-------------|----------------|------------------|
| LVE | 0.003771 +/- 0.000062 | 0.003286 +/- 0.000039 | **0.003121 +/- 0.000023** |
| MVE | 0.001348 +/- 0.000010 | 0.001244 +/- 0.000008 | **0.001219 +/- 0.000006** |
| UFVE | 0.001278 +/- 0.000012 | 0.001138 +/- 0.000011 | **0.001101 +/- 0.000008** |
| FDD | 5.88e-6 +/- 4.59e-7 | **4.24e-6 +/- 1.78e-7** | 4.71e-6 +/- 1.16e-7 |

## Key Observations

1. **Condition C outperforms A and B on LVE, MVE, UFVE.** LVE: 17.2% better than A, 5.0% better than B. Supports the hypothesis that phoneme+timing conditioning is an effective temporal scaffold.
2. **Lowest variance across seeds** (LVE std = 0.000023), indicating stable training.
3. **FDD is slightly higher for C than B** (4.71e-6 vs 4.24e-6), suggesting discrete units preserve temporal dynamics marginally better.
4. **Fastest training**: ~0.3s/epoch vs ~3.4s (B) or ~5s (A), no pretrained encoder needed.
5. **All trainable**: 1.28M trainable parameters, more parameter-efficient than A/B.
