# HuBERT-Continuous Ablation on BIWI

## Experiment Overview

This ablation determines whether the benefit of Condition B (discrete units) comes from discretization itself or from the underlying HuBERT representation. A variant replaces k-means unit IDs with continuous HuBERT features (no quantization), keeping all else equal.

## Setup

- **Frontend**: HuBERT-base-ls960 (frozen) continuous features (768-dim, 50Hz) + prosody (F0, energy) -> linear projection to 256-dim
- **Difference from Condition A**: Uses HuBERT instead of WavLM as the SSL encoder
- **Difference from Condition B**: No k-means quantization; uses continuous HuBERT features directly
- **Dataset**: BIWI (synthetic, 190 train / 24 val / 24 test)
- **Training**: AdamW, lr=2e-4, weight_decay=1e-4, 600 epochs, cosine LR with 10-epoch warmup, batch_size=8
- **Seeds**: 42, 123, 456
- **Checkpoints**: `outputs/biwi/hubert_continuous/seed{42,123,456}/best_model.pt`

## Key Results

| Variant | Encoder | Discretization | LVE (mean +/- std) |
|---------|---------|---------------|-------------------|
| Condition A (SSL) | WavLM-base-plus | No | 9.00e-6 +/- 6.85e-7 |
| **HuBERT-continuous** | **HuBERT-base-ls960** | **No** | **9.49e-6 +/- 1.06e-7** |
| Condition B (Units) | HuBERT-base-ls960 | Yes (K=200) | 8.08e-6 +/- 8.05e-7 |

### Per-Seed LVE

| Seed | LVE | Best Epoch | Best Val Loss |
|------|-----|------------|--------------|
| 42 | 9.41e-6 | 595 | 3.49e-6 |
| 123 | 9.64e-6 | 590 | 3.53e-6 |
| 456 | 9.43e-6 | 590 | 3.49e-6 |

## Key Observations

1. **Discretization IS the driver**: Condition B (discrete, LVE=8.08e-6) outperforms HuBERT-continuous (LVE=9.49e-6) by ~15%. Since both use the same HuBERT encoder, the only difference is quantization via k-means. This confirms that discretization provides a beneficial information bottleneck.

2. **HuBERT vs WavLM (continuous)**: HuBERT-continuous (9.49e-6) performs slightly worse than Condition A / WavLM-continuous (9.00e-6). This suggests the HuBERT encoder itself is not inherently better than WavLM for this facial animation task when used in continuous form.

3. **Low seed variance**: HuBERT-continuous shows very low variance across seeds (std=1.06e-7), suggesting stable but consistently suboptimal performance.

4. **Implication**: The performance hierarchy is B (discrete) > A (WavLM continuous) > HuBERT-continuous, supporting the hypothesis that discretization acts as a beneficial information bottleneck that removes irrelevant speaker/acoustic variation while preserving the phonetic content needed for lip synchronization.
