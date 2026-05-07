# SFT Baseline on TRACE Default Order (Qwen2-1.5B)

## Experiment Overview

Sequential Fine-Tuning (SFT) baseline on the TRACE benchmark default task order using Qwen2-1.5B. SFT trains the model on each task sequentially with standard cross-entropy loss — no gradient masking, replay, or regularization. This serves as the simplest continual learning baseline and validates the shared training and evaluation pipeline for all subsequent experiments.

## Setup

- **Model**: Qwen2-1.5B (BF16)
- **Task Order**: C-STANCE -> FOMC -> MeetingBank -> Py150 -> ScienceQA -> NumGLUE-cm -> NumGLUE-ds -> 20Minuten
- **Per-task Epochs**: [5, 3, 7, 5, 3, 5, 5, 7]
- **Optimizer**: AdamW, LR=1e-5, constant schedule with warmup, weight_decay=0
- **Batch Size**: 128 (16 per device x 8 GPUs)
- **Precision**: BF16 mixed precision
- **DeepSpeed**: ZeRO Stage 2
- **Gradient Checkpointing**: Enabled
- **Seed**: 42
- **GPUs**: 8x A100-80GB
- **Evaluation**: vLLM with temperature=0.1, data-parallel-size=2

## Key Results

| Metric | Our Result | Published (FGGM Table 1) | Diff |
|--------|-----------|-------------------------|------|
| **TRACE-OP** | **49.31** | 49.22 | +0.09 |
| BWT | -34.25 | N/A | - |
| OP_T (final row avg) | 24.10 | N/A | - |
| Diagonal avg | 54.07 | N/A | - |

**Validation: PASS** (49.31 is within 47.22-51.22 tolerance of published 49.22)

### Performance Matrix (8x8)

Each row = checkpoint after training on task t (0-indexed). Each column = evaluation on that task.

| Checkpoint | C-STANCE | FOMC | MeetingBank | Py150 | ScienceQA | NumGLUE-cm | NumGLUE-ds | 20Minuten |
|-----------|----------|------|-------------|-------|-----------|------------|------------|-----------|
| 0 (C-STANCE) | **50.10** | - | - | - | - | - | - | - |
| 1 (FOMC) | 47.35 | **66.33** | - | - | - | - | - | - |
| 2 (MeetingBank) | 47.75 | 62.30 | **39.73** | - | - | - | - | - |
| 3 (Py150) | 47.50 | 59.68 | 37.90 | **57.98** | - | - | - | - |
| 4 (ScienceQA) | 48.40 | 59.88 | 38.40 | 54.61 | **86.30** | - | - | - |
| 5 (NumGLUE-cm) | 47.30 | 61.69 | 39.40 | 56.76 | 80.50 | **34.57** | - | - |
| 6 (NumGLUE-ds) | 48.15 | 62.50 | 39.35 | 57.05 | 75.10 | 23.46 | **57.54** | - |
| 7 (20Minuten) | 42.95 | 22.58 | 18.85 | 6.83 | 61.60 | 0.00 | 0.00 | **40.03** |

### Per-Step OP Values (TRACE-OP = average of these)

| Step | OP_t |
|------|------|
| 1 | 50.10 |
| 2 | 56.84 |
| 3 | 49.93 |
| 4 | 50.76 |
| 5 | 57.52 |
| 6 | 53.37 |
| 7 | 51.88 |
| 8 | 24.10 |
| **Avg (TRACE-OP)** | **49.31** |

## Key Observations

1. **Pipeline validated**: TRACE-OP=49.31 matches published 49.22 within tolerance, confirming correctness of data loading, training, and evaluation pipelines.

2. **Catastrophic forgetting on final task (20Minuten)**: After training on 20Minuten (a text simplification task with 7 epochs), performance on FOMC drops from 62.50 to 22.58, Py150 from 57.05 to 6.83, and both NumGLUE tasks collapse to 0.00. This is consistent with 20Minuten being a long-training, fundamentally different task (German text simplification) that overwrites learned representations.

3. **Diagonal scores are strong**: Average diagonal performance is 54.07, showing the model successfully learns each task when freshly trained. The forgetting (BWT=-34.25) is the primary challenge.

4. **TRACE-OP definition**: TRACE-OP is NOT the final row average (OP_T=24.10). It is the average of all intermediate OP_t values (t=1..T). This weights early-task retention heavily, where forgetting is less severe.

5. **General ability evaluation**: Skipped due to cluster resource constraints. TRACE-OP validation is the primary criterion for pipeline correctness.

## Technical Notes

- **Token-merge fix**: Qwen2 tokenizer merges `:` with following characters (e.g., `Stance:C` -> tokens `[St, ance, :C]`). During vLLM inference, standalone `Stance:` tokenizes differently. Fixed by detecting merge conflicts and dropping the trailing token, then stripping the regenerated prefix from predictions.

- **First-character extraction**: Classification tasks (C-STANCE, FOMC) produce verbose predictions after ScienceQA training teaches the model to explain answers. Fixed by extracting only the first character for classification metrics.
