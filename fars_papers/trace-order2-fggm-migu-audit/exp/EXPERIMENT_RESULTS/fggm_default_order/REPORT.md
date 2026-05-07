# FGGM (Fisher-Guided Gradient Masking) - TRACE Default Order Sanity Check

## Experiment Overview

Implemented FGGM from the algorithm description (no public code available) and validated on TRACE's default task order with Qwen2-1.5B. FGGM computes a diagonal Fisher Information Matrix per task to build a binary gradient mask that restricts parameter updates to the most Fisher-important subset.

## Setup

- **Model**: Qwen2-1.5B
- **Task order**: C-STANCE -> FOMC -> MeetingBank -> Py150 -> ScienceQA -> NumGLUE-cm -> NumGLUE-ds -> 20Minuten
- **Seed**: 42
- **Epochs**: [5, 3, 7, 5, 3, 5, 5, 7]
- **Optimizer**: AdamW (lr=1e-5, constant schedule, weight_decay=0)
- **Batch size**: 128 (8 GPUs x 16)
- **FGGM alpha**: 0.7 (30% of parameters updated per task, 70% frozen)
- **DeepSpeed**: ZeRO stage 2, bf16
- **Fisher computation**: model_engine.backward() only (no optimizer step), gradient hooks capture squared gradients for FIM

## Key Results

| Metric | Value | Published | Tolerance | Status |
|--------|-------|-----------|-----------|--------|
| **TRACE-OP** | **45.84** | 46.00 | ±2.0 | **PASS** |
| BWT | -8.52 | - | - | - |

### Comparison with Other Methods

| Method | TRACE-OP | BWT | Published OP |
|--------|----------|-----|-------------|
| SFT | 49.31 | -34.25 | 49.22 |
| MIGU | 47.43 | -8.05 | 44.08 |
| **FGGM** | **45.84** | **-8.52** | **46.00** |

### Performance Matrix (final row = after all 8 tasks)

| Task | C-STANCE | FOMC | MeetingBank | Py150 | ScienceQA | NumGLUE-cm | NumGLUE-ds | 20Minuten |
|------|----------|------|-------------|-------|-----------|------------|------------|-----------|
| After task 7 | 49.35 | 49.80 | 35.75 | 57.38 | 72.20 | 23.46 | 39.08 | 39.70 |
| Diagonal | 50.60 | 59.48 | 39.30 | 59.80 | 84.45 | 35.80 | 57.23 | 39.70 |

## Key Observations

1. **TRACE-OP validation passed**: 45.84 is within [44.00, 48.00], confirming correct FGGM implementation.

2. **Forgetting mitigation**: FGGM BWT=-8.52 vs SFT BWT=-34.25 shows that gradient masking substantially reduces catastrophic forgetting, comparable to MIGU (BWT=-8.05).

3. **Plasticity-stability trade-off**: FGGM TRACE-OP (45.84) < SFT (49.31), reflecting reduced plasticity from masking 70% of parameters per task.

4. **Critical implementation details**:
   - Fisher computation must NOT call model_engine.step() to avoid optimizer state corruption
   - Input-dimension Aggregation: for weight W in R^{D_out x D_in}, sum Fisher across dim=1 per output neuron
   - Bias parameters use raw Fisher values (no aggregation)
   - Quantile threshold at alpha=0.7 (70th percentile), mask=1 for scores above threshold
   - Task 0 has no masking (no prior knowledge to protect)
   - Binary masks saved per task for downstream overlap analysis

5. **Mask statistics**: Consistent 30.0% updated / 70.0% frozen across all tasks and all 8 GPU ranks.
