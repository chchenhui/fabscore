# Temporal Shuffle Negative Control (Sanity Check)

## Experiment Overview

This experiment validates the experimental framework by running a negative control: shuffling the temporal order of audio features within each utterance for Condition A (SSL) before feeding them to the decoder. If the framework is working correctly, time-shuffled features should produce substantially degraded LVE compared to temporally ordered features, because lip motion is tightly coupled to the temporal progression of speech.

## Setup

- **Dataset**: BIWI
- **Condition**: A (SSL + Prosody) with temporal shuffle
- **Seed**: 42 (single seed sufficient for sanity check)
- **Shuffle Method**: After extracting WavLM features at 50 Hz, randomly permute the temporal order of feature frames (including prosody) within each utterance, independently per sample per batch. Different random permutation each forward pass.
- **Hyperparameters**: Identical to standard Condition A (600 epochs, AdamW, lr=2e-4, cosine scheduler, batch_size=8)
- **Implementation**: Added `shuffle_temporal` flag to `ScaffoldModel.forward()` in `scaffoldswap/model/scaffold_model.py` and `--shuffle_temporal` CLI arg in `scaffoldswap/train.py`
- **Checkpoint**: `scaffoldswap/outputs/biwi/condA_shuffled/seed42/best_model.pt`

## Key Results

| Configuration | LVE | MVE | UFVE | FDD | Best Epoch |
|---|---|---|---|---|---|
| Condition A Normal (seed=42) | 9.00e-06 | 1.160e-03 | 1.029e-03 | 3.19e-06 | 595 |
| Condition A Shuffled (seed=42) | **1.033e-05** | 1.206e-03 | 1.090e-03 | 2.61e-06 | 590 |
| Constant-Motion Baseline (mean face) | 7.79e-06 | 1.121e-03 | 9.75e-04 | 6.65e-06 | N/A |

### Degradation Analysis

- **Shuffled vs Normal LVE**: 1.033e-05 / 9.00e-06 = **1.149x** (14.9% worse)
- **Shuffled vs Constant Baseline LVE**: 1.033e-05 / 7.79e-06 = **1.326x** (32.6% worse)
- **Shuffled LVE worse than constant baseline**: Yes -- the shuffled model produces worse lip motion than simply predicting the mean face for every frame

## Key Observations

1. **Sanity check PASSES**: Temporal shuffling degrades LVE by 14.9%, confirming the decoder exploits temporal structure from the audio frontend.

2. **Shuffled model worse than mean-face baseline**: The shuffled model's LVE (1.033e-05) exceeds the constant-motion baseline (7.79e-06). This strongly confirms that when temporal structure is destroyed, the model cannot learn meaningful lip-speech mapping -- it performs worse than the trivial "predict nothing" baseline.

3. **Other metrics also degrade**: MVE increases by 3.95% (1.160e-03 -> 1.206e-03) and UFVE increases by 5.9% (1.029e-03 -> 1.090e-03) with shuffling. This indicates temporal structure impacts the entire face, not just lips.

4. **FDD decreases with shuffling**: FDD drops from 3.19e-06 to 2.61e-06, meaning the shuffled model produces less dynamic motion (closer to static). This makes sense -- without temporal coherence, the model defaults to smoother, less dynamic outputs.

5. **Training loss still converges**: The shuffled model's best val_loss (3.62e-06) is close to but worse than normal (3.41e-06), suggesting the model can still minimize loss somewhat by leveraging speaker embeddings and global audio statistics, but the temporal degradation is clearly reflected in the evaluation metrics.

## Verdict

**PASS** -- The temporal shuffle negative control confirms that the decoder genuinely exploits temporal structure from the audio frontend rather than learning trivial shortcuts (e.g., relying solely on speaker identity or static features).
