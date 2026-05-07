# MIGU on TRACE Order 2 (3 Seeds)

## Experiment Overview

MIGU (MIxed GradUal masking) evaluated on TRACE benchmark Order 2 with Qwen2-1.5B across 3 seeds (42, 123, 456). This establishes MIGU's performance under the alternative task order for comparison against FGGM and SFT on Order 2. MIGU recomputes gradient masks from each batch's activations (not from per-task Fisher estimates), so it may be less sensitive to task ordering than FGGM.

## Setup

- **Model**: Qwen2-1.5B (BF16)
- **Task Order (Order 2)**: NumGLUE-cm -> NumGLUE-ds -> FOMC -> 20Minuten -> C-STANCE -> Py150 -> MeetingBank -> ScienceQA
- **Seeds**: 42, 123, 456
- **Per-task Epochs**: NumGLUE-cm(5), NumGLUE-ds(5), FOMC(3), 20Minuten(7), C-STANCE(5), Py150(5), MeetingBank(7), ScienceQA(3)
- **Optimizer**: AdamW, LR=1e-5, constant with warmup, weight_decay=0
- **Batch Size**: 128 (16 per device x 8 GPUs)
- **Precision**: BF16, DeepSpeed ZeRO Stage 2
- **MIGU threshold ratio**: T=0.7 (mask top 70% of activations by magnitude)
- **Evaluation**: vLLM with data-parallel-size=8, temperature=0.1

## Key Results

### Aggregate Metrics (3 Seeds)

| Metric | Mean | Std |
|--------|------|-----|
| **TRACE-OP** | **43.72** | 0.13 |
| **BWT** | **-1.07** | 0.65 |

### Per-Seed Results

| Seed | TRACE-OP | BWT |
|------|----------|-----|
| 42 | 43.80 | -0.94 |
| 123 | 43.53 | -1.92 |
| 456 | 43.84 | -0.36 |

### Comparison with SFT and FGGM on Order 2

| Method | Order | TRACE-OP | BWT |
|--------|-------|----------|-----|
| **MIGU** | **Order 2** | **43.72 +/- 0.13** | **-1.07 +/- 0.65** |
| FGGM | Order 2 | 40.77 +/- 1.06 | -3.41 +/- 1.66 |
| SFT | Order 2 | 39.82 +/- 0.47 | -5.30 +/- 0.59 |

### Comparison Across Orders

| Method | Order | TRACE-OP | BWT |
|--------|-------|----------|-----|
| MIGU | Default | 47.43 | -8.05 |
| **MIGU** | **Order 2** | **43.72** | **-1.07** |
| FGGM | Default | 45.84 | -8.52 |
| FGGM | Order 2 | 40.77 | -3.41 |
| SFT | Default | 49.31 | -34.25 |
| SFT | Order 2 | 39.82 | -5.30 |

### OP Per Step (Mean Across Seeds)

| Step | Task | OP_t |
|------|------|------|
| 1 | NumGLUE-cm | 32.51 |
| 2 | NumGLUE-ds | 46.06 |
| 3 | FOMC | 46.97 |
| 4 | 20Minuten | 43.71 |
| 5 | C-STANCE | 43.01 |
| 6 | Py150 | 44.72 |
| 7 | MeetingBank | 43.34 |
| 8 | ScienceQA | 49.48 |

### Final Row Performance (Mean Across Seeds, After All 8 Tasks)

| Task | Score | Std |
|------|-------|-----|
| NumGLUE-cm | 33.74 | 1.16 |
| NumGLUE-ds | 53.03 | 0.52 |
| FOMC | 53.97 | 1.48 |
| 20Minuten | 39.23 | 0.34 |
| C-STANCE | 46.78 | 7.47 |
| Py150 | 55.43 | 0.65 |
| MeetingBank | 32.61 | 1.34 |
| ScienceQA | 81.05 | 0.32 |

### Diagonal Scores (Mean Across Seeds)

| Task | Score | Std |
|------|-------|-----|
| NumGLUE-cm | 32.51 | 0.58 |
| NumGLUE-ds | 53.85 | 1.76 |
| FOMC | 51.14 | 1.24 |
| 20Minuten | 39.90 | 0.45 |
| C-STANCE | 52.40 | 0.83 |
| Py150 | 56.70 | 0.50 |
| MeetingBank | 35.81 | 0.10 |
| ScienceQA | 81.05 | 0.32 |

## Key Observations

1. **MIGU outperforms FGGM and SFT on Order 2**: MIGU TRACE-OP (43.72) is 2.95 points higher than FGGM (40.77) and 3.90 points higher than SFT (39.82). This is a notable advantage, consistent with MIGU's batch-level gradient masking being more adaptive than FGGM's fixed per-task Fisher masks.

2. **MIGU has the best BWT on Order 2**: MIGU BWT (-1.07) is substantially better than FGGM (-3.41) and SFT (-5.30). MIGU suffers the least forgetting under this task order.

3. **Extremely low TRACE-OP variance**: MIGU's TRACE-OP std of 0.13 is the smallest across all methods (FGGM: 1.06, SFT: 0.47), indicating MIGU is highly stable across seeds on Order 2.

4. **MIGU TRACE-OP drops less between orders**: MIGU drops 3.71 points from default (47.43) to Order 2 (43.72), while FGGM drops 5.07 points (45.84 -> 40.77). This supports the hypothesis that MIGU's batch-level masking is less sensitive to task ordering than FGGM's pre-computed Fisher masks.

5. **BWT improvement from default to Order 2**: All methods show improved BWT on Order 2 (MIGU: -8.05 -> -1.07, FGGM: -8.52 -> -3.41, SFT: -34.25 -> -5.30). This is driven by Order 2 placing ScienceQA last (short 3-epoch training, high accuracy), whereas the default order places 20Minuten last.

6. **C-STANCE high variance**: C-STANCE final row std=7.47 for MIGU (vs FGGM=5.78, SFT=3.20). Seed 123 shows only 36.25 while seed 456 shows 51.30, a 15-point spread. This task is consistently the most variable across seeds and methods.

7. **General ability evaluation**: Skipped for consistency with prior experiments (SFT default, MIGU default, FGGM default, FGGM Order 2, SFT Order 2 all skipped general eval).

## Files

- Configs: `audit/configs/migu_order2_seed{42,123,456}.yaml`
- Training script: `audit/scripts/run_migu_order2_train.sh`
- Evaluation script: `audit/scripts/run_migu_order2_eval.sh`
- Checkpoints: `audit/results/migu_order2_seed{42,123,456}/checkpoints/{0-7}/`
- Eval results: `audit/results/migu_order2_seed{42,123,456}/eval/`
- Aggregated summary: `audit/results/migu_order2_summary.json`
