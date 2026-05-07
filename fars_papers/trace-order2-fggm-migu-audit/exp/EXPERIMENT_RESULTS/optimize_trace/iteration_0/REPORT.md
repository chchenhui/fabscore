# Optimization Iteration 0: FGGM on TRACE Order 2 (3 Seeds)

## Overview

The Main Experiment ("FGGM on TRACE Order 2, 3 Seeds") had never been executed -- the config file was a placeholder. This optimization populated the configs, ran training for 3 seeds (42, 123, 456) with 8 GPUs each, and evaluated all checkpoints.

## Issues Found and Fixed

1. **Missing Config (Critical)**: `audit/configs/fggm_order2.yaml` was a placeholder with no actual configuration. Created 3 seed-specific configs (`fggm_order2_seed{42,123,456}.yaml`) with Order 2 task ordering, identical hyperparameters to the validated default order config.

2. **Missing Scripts**: No training or evaluation scripts existed for Order 2. Created `run_fggm_order2_train.sh` and `run_fggm_order2_eval.sh` parameterized by seed.

## Results

| Seed | TRACE-OP | BWT |
|------|----------|-----|
| 42 | 41.14 | -1.11 |
| 123 | 41.85 | -4.16 |
| 456 | 39.33 | -4.96 |
| **Mean** | **40.77 +/- 1.06** | **-3.41 +/- 1.66** |

## Comparison with Default Order

| Method | Order | TRACE-OP | BWT |
|--------|-------|----------|-----|
| FGGM | Default (seed=42) | 45.84 | -8.52 |
| FGGM | Order 2 (3 seeds) | 40.77 | -3.41 |

## Success Assessment

The experiment was successfully executed and produced valid results. The FGGM implementation (validated on default order with TRACE-OP=45.84 vs published 46.00) was correctly applied to Order 2 with consistent mask statistics (30%/70% split) across all seeds and tasks.
