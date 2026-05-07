# Ablation: Condition C Without Timing Features (BIWI)

## Experiment Overview

This ablation tests whether the explicit timing features (within-phoneme position `p in [0,1]` and phoneme duration `d`) contribute to Condition C's performance, or whether phoneme identity alone at frame-level resolution is sufficient.

**Variant "C w/o timing"**: Concatenates `[phoneme_embedding, F0, energy]` (omitting `p` and `d`) before the linear projection. The phoneme embedding is still looked up per 50 Hz frame from MFA alignment, so the model still knows which phoneme is active at each frame, but receives no sub-phoneme positional or duration information.

## Setup

- Dataset: BIWI (synthetic, 190 train / 24 val / 24 test)
- Seeds: 42, 123, 456
- Config: `scaffoldswap/configs/biwi_condC_no_timing_v3.yaml`
- Hyperparameters: identical to Condition C baseline (AdamW, lr=2e-4, 600 epochs, cosine LR, warmup=10)
- Change: `timing_dim: 0` in config (vs `timing_dim: 2` for full Condition C)
- Checkpoints: `scaffoldswap/outputs/biwi/condC_no_timing/seed{42,123,456}/best_model.pt`

## Key Results

| Variant | LVE (mean +/- std) | MVE | Best Epochs |
|---------|--------------------|----|-------------|
| C (full, with p & d) | 8.188e-6 +/- 3.43e-8 | 1.132e-3 | 600, 595, 580 |
| C w/o timing | 8.189e-6 +/- 5.29e-8 | 1.131e-3 | 595, 590, 600 |
| **Relative diff** | **+0.01%** | **-0.01%** | -- |

### Per-Seed LVE Breakdown

| Seed | C (full) | C w/o timing | Diff |
|------|----------|--------------|------|
| 42 | 8.224e-6 | 8.237e-6 | +0.16% |
| 123 | 8.199e-6 | 8.215e-6 | +0.20% |
| 456 | 8.142e-6 | 8.115e-6 | -0.33% |

## Key Observations

1. **Timing features have negligible impact**: The "C w/o timing" variant achieves virtually identical LVE to the full Condition C (8.189e-6 vs 8.188e-6, +0.01% relative difference). This difference is well within seed-to-seed variance.

2. **Phoneme identity alone is sufficient**: The frame-level phoneme ID embedding already encodes temporal structure implicitly -- the model knows which phoneme is active at each 50 Hz frame, and phoneme boundaries are naturally captured by the change in embedding. The explicit sub-phoneme position and duration features are redundant.

3. **Consistent across seeds**: All 3 seeds show the same pattern -- no seed shows a meaningful degradation from removing timing features.

4. **Conclusion**: The phoneme scaffold's effectiveness comes from the phoneme identity information at frame-level resolution, NOT from the explicit timing signals (position p, duration d). The MFA alignment already provides the critical temporal structure by assigning phoneme labels to individual frames.
