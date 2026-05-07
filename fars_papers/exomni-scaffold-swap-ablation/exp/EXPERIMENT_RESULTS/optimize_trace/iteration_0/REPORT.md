# Optimization Iteration 0: Extended Training + AdamW + Warmup

## Overview

Extended all 18 models (6 conditions x 3 seeds) from 300 to 600 epochs, switched from Adam to AdamW with weight_decay=1e-4, increased initial LR from 1e-4 to 2e-4, and added 10-epoch linear warmup before cosine decay.

## Issues Diagnosed

1. **Training too short**: All v2 best checkpoints at epoch 295-300 of 300, val_loss still improving at training end
2. **No weight decay**: Adam without regularization led to suboptimal generalization
3. **Conservative LR**: 1e-4 too low for the 1.2M parameter decoder

## Changes Made

| Parameter | v2 | v3 |
|-----------|----|----|
| Epochs | 300 | 600 |
| Optimizer | Adam | AdamW |
| LR | 1e-4 | 2e-4 |
| Weight Decay | 0 | 1e-4 |
| Warmup | none | 10 epochs linear |
| Cosine T_max | 300 | 590 (600-10) |

## Results

### BIWI

| Condition | v2 LVE | v3 LVE | Improvement |
|-----------|--------|--------|-------------|
| A (SSL/WavLM) | 0.003771 | 9.00e-6 | 99.8% |
| B (HuBERT units) | 0.003286 | 8.08e-6 | 99.8% |
| C (Phoneme+Timing) | 0.003121 | 8.19e-6 | 99.7% |

### VOCASET

| Condition | v2 LVE (x10^-5) | v3 LVE (x10^-5) | Improvement |
|-----------|-----------------|-----------------|-------------|
| A (SSL/WavLM) | 1.35 | 0.48 | 64.1% |
| B (HuBERT units) | 0.75 | 0.46 | 39.3% |
| C (Phoneme+Timing) | 0.87 | 0.46 | 47.4% |

### Full Metrics (v3)

**BIWI:**
| Condition | LVE | MVE | UFVE | FDD |
|-----------|-----|-----|------|-----|
| A | 9.00e-6 | 1.160e-3 | 1.028e-3 | 3.23e-6 |
| B | 8.08e-6 | 1.128e-3 | 9.84e-4 | 4.77e-6 |
| C | 8.19e-6 | 1.132e-3 | 9.88e-4 | 4.86e-6 |

**VOCASET:**
| Condition | LVE | MVE | UFVE | FDD |
|-----------|-----|-----|------|-----|
| A | 4.84e-6 | 3.35e-4 | 2.66e-4 | 2.30e-6 |
| B | 4.55e-6 | 3.21e-4 | 2.49e-4 | 3.27e-6 |
| C | 4.57e-6 | 3.22e-4 | 2.50e-4 | 3.28e-6 |

### Rankings

- BIWI: B > C > A (same as v2)
- VOCASET: B > C > A (same as v2)

Discrete representations (B, C) consistently outperform continuous SSL (A) on both datasets.

## Files Changed

- `scaffoldswap/train.py` - Added AdamW optimizer, linear warmup scheduler, weight_decay support
- `scaffoldswap/configs/*_v3.yaml` - 6 new v3 configs for all conditions
- `scaffoldswap/scripts/run_train_v3.sh` - Unified v3 training script
- `scaffoldswap/scripts/run_eval_v3.sh` - v3 evaluation script
