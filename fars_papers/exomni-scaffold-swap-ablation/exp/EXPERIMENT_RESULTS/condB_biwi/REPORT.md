# Condition B (HuBERT Discrete Units + Prosody) on BIWI

## Experiment Overview

Condition B tests the premise that discrete speech units provide an effective temporal scaffold for facial animation. Audio is processed through HuBERT-base-ls960 (frozen), quantized to K=200 discrete unit IDs via k-means, then mapped through a learnable embedding table before concatenation with prosody features.

## Setup

- **Frontend**: HuBERT-base-ls960 (frozen) -> last_hidden_state (768-dim, 50 Hz) -> MiniBatchKMeans (K=200) quantization -> nn.Embedding(200, 256) -> concat F0+energy (2-dim) -> Linear(258, 256)
- **K-means codebook**: Trained on all 190 BIWI training sequences, all 200 clusters non-empty
- **Decoder**: FrequencyAdaptor (50Hz->25fps) -> TCN decoder (5 blocks, kernel_size=3) -> 512-dim PCA output
- **Training**: Adam optimizer, lr=1e-4 with CosineAnnealingLR (eta_min=1e-6), 300 epochs, batch_size=8, grad_clip=1.0, vertex_scaling=0.2
- **Eval every**: 5 epochs, best checkpoint by val_loss
- **Seeds**: [42, 123, 456]
- **Trainable params**: 1,318,528 / 95,690,240 total

## Key Results

| Metric | Mean | Std | Seed 42 | Seed 123 | Seed 456 |
|--------|------|-----|---------|----------|----------|
| LVE | 0.003286 | 0.000039 | 0.003307 | 0.003319 | 0.003231 |
| MVE | 0.001244 | 0.000008 | 0.001247 | 0.001251 | 0.001232 |
| UFVE | 0.001138 | 0.000011 | 0.001143 | 0.001148 | 0.001123 |
| FDD | 4.24e-6  | 1.78e-7  | 4.32e-6  | 4.40e-6  | 3.99e-6  |

Best epochs: all at 300. Best val losses: ~3.8-3.9e-6.

## Key Observations

1. LVE improved 41.4% over v1 (100 epochs): 0.005605 -> 0.003286.
2. Condition B outperforms Condition A on all metrics (LVE: 0.003286 vs 0.003771).
3. Very low cross-seed variance, indicating stable and reproducible training.
