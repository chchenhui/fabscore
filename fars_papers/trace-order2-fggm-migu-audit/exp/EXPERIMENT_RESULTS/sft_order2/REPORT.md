# SFT Baseline on TRACE Order 2 (3 Seeds)

## Experiment Overview

Unconstrained sequential fine-tuning (SFT) evaluated on TRACE benchmark Order 2 with Qwen2-1.5B across 3 seeds (42, 123, 456). This establishes the SFT baseline under the alternative task ordering for comparison against MIGU and FGGM on Order 2.

## Setup

- **Model**: Qwen2-1.5B (BF16)
- **Task Order (Order 2)**: NumGLUE-cm -> NumGLUE-ds -> FOMC -> 20Minuten -> C-STANCE -> Py150 -> MeetingBank -> ScienceQA
- **Seeds**: 42, 123, 456
- **Per-task Epochs**: NumGLUE-cm(5), NumGLUE-ds(5), FOMC(3), 20Minuten(7), C-STANCE(5), Py150(5), MeetingBank(7), ScienceQA(3)
- **Optimizer**: AdamW, LR=1e-5, constant with warmup, weight_decay=0
- **Batch Size**: 128 (16 per device x 8 GPUs)
- **Precision**: BF16, DeepSpeed ZeRO Stage 2
- **Evaluation**: vLLM with data-parallel-size=8, temperature=0.1

## Key Results

### Aggregate Metrics (3 Seeds)

| Metric | Mean | Std |
|--------|------|-----|
| **TRACE-OP** | **39.82** | 0.47 |
| **BWT** | **-5.30** | 0.59 |

### Per-Seed Results

| Seed | TRACE-OP | BWT |
|------|----------|-----|
| 42 | 40.31 | -4.48 |
| 123 | 39.20 | -5.86 |
| 456 | 39.97 | -5.56 |

### Comparison with Default Order and Other Methods

| Method | Order | TRACE-OP | BWT |
|--------|-------|----------|-----|
| **SFT** | **Order 2** | **39.82 +/- 0.47** | **-5.30 +/- 0.59** |
| SFT | Default | 49.31 | -34.25 |
| FGGM | Order 2 | 40.77 +/- 1.06 | -3.41 +/- 1.66 |
| FGGM | Default | 45.84 | -8.52 |
| MIGU | Default | 47.43 | -8.05 |

### OP Per Step (Mean Across Seeds)

| Step | Task | OP_t |
|------|------|------|
| 1 | NumGLUE-cm | 32.92 |
| 2 | NumGLUE-ds | 39.73 |
| 3 | FOMC | 39.83 |
| 4 | 20Minuten | 35.76 |
| 5 | C-STANCE | 37.05 |
| 6 | Py150 | 43.48 |
| 7 | MeetingBank | 42.29 |
| 8 | ScienceQA | 47.53 |

### Final Row Performance (Mean Across Seeds, After All 8 Tasks)

| Task | Score | Std |
|------|-------|-----|
| NumGLUE-cm | 22.63 | 1.54 |
| NumGLUE-ds | 45.64 | 0.52 |
| FOMC | 56.32 | 2.05 |
| 20Minuten | 38.59 | 0.55 |
| C-STANCE | 50.63 | 3.20 |
| Py150 | 54.67 | 2.68 |
| MeetingBank | 26.26 | 1.32 |
| ScienceQA | 85.53 | 0.26 |

### Diagonal Scores (Mean Across Seeds)

| Task | Score | Std |
|------|-------|-----|
| NumGLUE-cm | 32.92 | 0.58 |
| NumGLUE-ds | 58.05 | 0.81 |
| FOMC | 55.85 | 3.33 |
| 20Minuten | 39.42 | 0.41 |
| C-STANCE | 52.28 | 0.69 |
| Py150 | 57.79 | 0.20 |
| MeetingBank | 35.52 | 0.83 |
| ScienceQA | 85.53 | 0.26 |

## Key Observations

1. **Dramatically improved BWT on Order 2 vs Default**: SFT Order 2 BWT (-5.30) is vastly better than SFT Default BWT (-34.25). This is because Order 2 places ScienceQA last (short 3-epoch training, high accuracy), while the default order places 20Minuten last (long 7-epoch training that overwrites prior knowledge).

2. **Lower TRACE-OP on Order 2**: SFT Order 2 TRACE-OP (39.82) is 9.49 points lower than SFT Default (49.31). The front-loaded numerical reasoning tasks (NumGLUE-cm OP_1=32.92) drag down the average, and significant forgetting occurs in middle tasks.

3. **SFT vs FGGM on Order 2**: SFT TRACE-OP (39.82) is close to FGGM (40.77), only 0.95 points apart. However, FGGM has better BWT (-3.41 vs -5.30), indicating FGGM provides moderate forgetting protection even on this order.

4. **High consistency across seeds**: TRACE-OP std is only 0.47, even lower than FGGM's 1.06, indicating SFT results are highly reproducible.

5. **C-STANCE highest variance**: Final row C-STANCE shows std=3.20, the highest across tasks, consistent with FGGM Order 2 observations.

6. **General ability evaluation**: Skipped for consistency with prior experiments (SFT default, MIGU default, FGGM default, FGGM Order 2 all skipped general eval).

## Files

- Configs: `audit/configs/sft_order2_seed{42,123,456}.yaml`
- Training script: `audit/scripts/run_sft_order2_train.sh`
- Evaluation script: `audit/scripts/run_sft_order2_eval.sh`
- Checkpoints: `audit/results/sft_order2_seed{42,123,456}/checkpoints/{0-7}/`
- Eval results: `audit/results/sft_order2_seed{42,123,456}/eval/`
- Aggregated summary: `audit/results/sft_order2_summary.json`
