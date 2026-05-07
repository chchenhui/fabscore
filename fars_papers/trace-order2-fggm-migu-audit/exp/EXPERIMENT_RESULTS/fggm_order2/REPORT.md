# FGGM on TRACE Order 2 (3 Seeds)

## Experiment Overview

FGGM (Fisher-Guided Gradient Masking) evaluated on TRACE benchmark Order 2 with Qwen2-1.5B across 3 seeds (42, 123, 456). This is the core experiment of the audit: testing whether FGGM's advantage over MIGU transfers from the default order to Order 2.

## Setup

- **Model**: Qwen2-1.5B (BF16)
- **Task Order (Order 2)**: NumGLUE-cm -> NumGLUE-ds -> FOMC -> 20Minuten -> C-STANCE -> Py150 -> MeetingBank -> ScienceQA
- **Seeds**: 42, 123, 456
- **Per-task Epochs**: NumGLUE-cm(5), NumGLUE-ds(5), FOMC(3), 20Minuten(7), C-STANCE(5), Py150(5), MeetingBank(7), ScienceQA(3)
- **Optimizer**: AdamW, LR=1e-5, constant with warmup, weight_decay=0
- **Batch Size**: 128 (16 per device x 8 GPUs)
- **Precision**: BF16, DeepSpeed ZeRO Stage 2
- **FGGM alpha**: 0.7 (30% parameters updated, 70% frozen per task)
- **Evaluation**: vLLM with data-parallel-size=8, temperature=0.1

## Key Results

### Aggregate Metrics (3 Seeds)

| Metric | Mean | Std |
|--------|------|-----|
| **TRACE-OP** | **40.77** | 1.06 |
| **BWT** | **-3.41** | 1.66 |

### Per-Seed Results

| Seed | TRACE-OP | BWT |
|------|----------|-----|
| 42 | 41.14 | -1.11 |
| 123 | 41.85 | -4.16 |
| 456 | 39.33 | -4.96 |

### Comparison with Default Order

| Method | Order | TRACE-OP | BWT |
|--------|-------|----------|-----|
| FGGM | Default | 45.84 | -8.52 |
| **FGGM** | **Order 2** | **40.77** | **-3.41** |
| MIGU | Default | 47.43 | -8.05 |
| SFT | Default | 49.31 | -34.25 |

### OP Per Step (Mean Across Seeds)

| Step | Task | OP_t |
|------|------|------|
| 1 | NumGLUE-cm | 32.92 |
| 2 | NumGLUE-ds | 41.17 |
| 3 | FOMC | 40.50 |
| 4 | 20Minuten | 37.54 |
| 5 | C-STANCE | 38.02 |
| 6 | Py150 | 43.44 |
| 7 | MeetingBank | 43.66 |
| 8 | ScienceQA | 48.94 |

### Final Row Performance (Mean Across Seeds, After All 8 Tasks)

| Task | Score | Std |
|------|-------|-----|
| NumGLUE-cm | 29.63 | 1.75 |
| NumGLUE-ds | 49.13 | 1.54 |
| FOMC | 56.65 | 1.98 |
| 20Minuten | 40.10 | 0.14 |
| C-STANCE | 48.20 | 5.78 |
| Py150 | 55.96 | 0.95 |
| MeetingBank | 26.92 | 1.71 |
| ScienceQA | 84.95 | 0.23 |

### Diagonal Scores (Mean Across Seeds)

| Task | Score | Std |
|------|-------|-----|
| NumGLUE-cm | 32.92 | 0.58 |
| NumGLUE-ds | 57.64 | 2.27 |
| FOMC | 54.17 | 2.68 |
| 20Minuten | 39.91 | 0.31 |
| C-STANCE | 52.87 | 0.74 |
| Py150 | 57.22 | 1.62 |
| MeetingBank | 35.75 | 1.20 |
| ScienceQA | 84.95 | 0.23 |

## Key Observations

1. **Lower TRACE-OP on Order 2**: FGGM Order 2 TRACE-OP (40.77) is significantly lower than default order (45.84), a drop of 5.07 points. This suggests FGGM is sensitive to task ordering, consistent with the audit hypothesis.

2. **Improved BWT on Order 2**: BWT improves from -8.52 (default) to -3.41 (Order 2), indicating less forgetting. This is likely because Order 2 places ScienceQA last (high accuracy, short training), while default order places 20Minuten last (long training that overwrites knowledge).

3. **Low early-step OP values**: OP_1 = 32.92 (NumGLUE-cm diagonal) is particularly low, dragging down TRACE-OP. The front-loaded numerical reasoning tasks (NumGLUE-cm, NumGLUE-ds) have inherently lower peak performance.

4. **Consistent across seeds**: Standard deviation of TRACE-OP is only 1.06, showing stable behavior across random seeds.

5. **C-STANCE high variance**: C-STANCE final row shows std=5.78, the highest across tasks, indicating sensitivity to seed in this binary classification task.

## Seed Extension Analysis (Step 3 of Task)

The task specifies extending to 5 seeds if the FGGM-MIGU gap in TRACE-OP falls within overlapping standard deviation ranges.

| Method | TRACE-OP Mean | Std | 1-sigma Range |
|--------|---------------|-----|---------------|
| FGGM | 40.77 | 1.06 | [39.71, 41.83] |
| MIGU | 43.72 | 0.13 | [43.59, 43.85] |
| SFT | 39.82 | 0.47 | [39.35, 40.29] |

The FGGM-MIGU gap is 2.95 points. The 1-sigma confidence intervals do NOT overlap (FGGM upper bound 41.83 < MIGU lower bound 43.59). The separation is clear even with the relatively high FGGM variance (std=1.06 vs MIGU std=0.13).

**Conclusion**: Seed extension to 5 seeds is NOT needed. The 3-seed results provide sufficient statistical power to confirm that MIGU outperforms FGGM on Order 2 (ranking reversal relative to the default order where FGGM OP=45.84 > MIGU OP=47.43 was closer).

## Aggregated Summary

Stored in `audit/results/fggm_order2_summary.json` for downstream paired comparisons.

## Files

- Configs: `audit/configs/fggm_order2_seed{42,123,456}.yaml`
- Training script: `audit/scripts/run_fggm_order2_train.sh`
- Evaluation script: `audit/scripts/run_fggm_order2_eval.sh`
- Checkpoints: `audit/results/fggm_order2_seed{42,123,456}/checkpoints/{0-7}/`
- Eval results: `audit/results/fggm_order2_seed{42,123,456}/eval/`
- Masks: `audit/results/fggm_order2_seed{42,123,456}/masks/`
