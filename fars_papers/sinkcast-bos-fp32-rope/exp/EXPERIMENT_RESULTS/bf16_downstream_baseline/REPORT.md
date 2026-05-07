# BF16 FlashAttention Downstream Position-Shift Baseline

## Experiment Overview

Evaluate standard BF16 FlashAttention on downstream benchmarks (RULER, LongBench) under position-shift conditions to measure accuracy degradation caused by BF16 RoPE shift-invariance breakdown. Each sample is evaluated twice: at default position offset (delta=0) and with M=4096 masked prefix tokens (delta=M).

## Setup

- **Models**: Llama-3.1-8B, Mistral-7B-v0.3 (BF16, FlashAttention-2)
- **RULER tasks**: NIAH-Single, NIAH-Multikey, Variable Tracing, QA (SQuAD-based); seq lengths 4096, 8192; 50 samples/task/length
- **LongBench tasks**: NarrativeQA (F1), HotpotQA (F1), GovReport (ROUGE-L), TREC (classification acc), PassageRetrieval-en (retrieval acc); 200 samples/task
- **Shift protocol**: Prepend M=4096 masked tokens (attention_mask=0), position_ids start at M for real tokens
- **Inference**: Greedy decoding, batch_size=1, deterministic (seed=42)
- **GPU**: 1x A100-80GB per model

## Key Results

### RULER Position-Shift

| Model | Seq Len | Task | Acc(d=0) | Acc(d=M) | Drop |
|-------|---------|------|----------|----------|------|
| Llama-3.1-8B | 4096 | niah_single | 100.0 | 100.0 | 0.0 |
| Llama-3.1-8B | 4096 | niah_multikey | 100.0 | 100.0 | 0.0 |
| Llama-3.1-8B | 4096 | variable_tracing | 2.0 | 2.0 | 0.0 |
| Llama-3.1-8B | 4096 | qa | 9.8 | 10.6 | -0.9 |
| Llama-3.1-8B | 4096 | **overall** | **52.9** | **53.2** | **-0.2** |
| Llama-3.1-8B | 8192 | niah_single | 100.0 | 100.0 | 0.0 |
| Llama-3.1-8B | 8192 | niah_multikey | 100.0 | 100.0 | 0.0 |
| Llama-3.1-8B | 8192 | variable_tracing | 6.0 | 6.0 | 0.0 |
| Llama-3.1-8B | 8192 | qa | 11.6 | 9.9 | 1.7 |
| Llama-3.1-8B | 8192 | **overall** | **54.4** | **54.0** | **0.4** |
| Mistral-7B-v0.3 | 4096 | niah_single | 54.0 | 52.0 | 2.0 |
| Mistral-7B-v0.3 | 4096 | niah_multikey | 36.0 | 40.0 | -4.0 |
| Mistral-7B-v0.3 | 4096 | variable_tracing | 0.0 | 0.0 | 0.0 |
| Mistral-7B-v0.3 | 4096 | qa | 13.9 | 15.6 | -1.6 |
| Mistral-7B-v0.3 | 4096 | **overall** | **26.0** | **26.9** | **-0.9** |
| Mistral-7B-v0.3 | 8192 | niah_single | 34.0 | 36.0 | -2.0 |
| Mistral-7B-v0.3 | 8192 | niah_multikey | 48.0 | 48.0 | 0.0 |
| Mistral-7B-v0.3 | 8192 | variable_tracing | 0.0 | 0.0 | 0.0 |
| Mistral-7B-v0.3 | 8192 | qa | 28.7 | 31.1 | -2.4 |
| Mistral-7B-v0.3 | 8192 | **overall** | **27.7** | **28.8** | **-1.1** |

### LongBench Position-Shift

| Model | Task | Metric | Acc(d=0) | Acc(d=M) | Drop |
|-------|------|--------|----------|----------|------|
| Llama-3.1-8B | narrativeqa | F1 | 23.7 | 24.5 | -0.9 |
| Llama-3.1-8B | hotpotqa | F1 | 11.8 | 11.9 | -0.2 |
| Llama-3.1-8B | gov_report | ROUGE-L | 9.4 | 9.8 | -0.4 |
| Llama-3.1-8B | trec | Acc | 67.5 | 67.5 | 0.0 |
| Llama-3.1-8B | passage_retrieval_en | Acc | 35.0 | 35.5 | -0.5 |
| Llama-3.1-8B | **overall** | **avg** | **29.5** | **29.9** | **-0.4** |
| Mistral-7B-v0.3 | narrativeqa | F1 | 17.0 | 16.9 | 0.1 |
| Mistral-7B-v0.3 | hotpotqa | F1 | 14.5 | 14.2 | 0.3 |
| Mistral-7B-v0.3 | gov_report | ROUGE-L | 1.9 | 1.7 | 0.2 |
| Mistral-7B-v0.3 | trec | Acc | 71.0 | 71.5 | -0.5 |
| Mistral-7B-v0.3 | passage_retrieval_en | Acc | 22.0 | 22.0 | 0.0 |
| Mistral-7B-v0.3 | **overall** | **avg** | **25.3** | **25.3** | **0.0** |

## Key Observations

1. **Position-shift drop is minimal on downstream tasks.** Across both models, both benchmarks, and all tasks, the accuracy drop under M=4096 shift is consistently within noise range (typically <1 point, often negative meaning shifted is slightly better).

2. **Llama-3.1-8B shows no meaningful degradation.** NIAH tasks remain at 100%, and LongBench tasks show slight improvements under shift (avg drop = -0.4). The only positive drop is QA at seq_len=8192 (+1.7), likely noise.

3. **Mistral-7B-v0.3 shows no meaningful degradation.** Despite lower baseline accuracy on RULER tasks (NIAH ~50% vs Llama's 100%), the shift-induced drop is within noise. LongBench avg drop is essentially 0.0.

4. **Variable tracing accuracy is near-zero for both models**, indicating these base models struggle with the task regardless of position shift.

5. **The downstream signal of BF16 shift-error is much weaker than the microbenchmark signal.** While microbenchmarks show measurable D_logit differences at the attention level, these do not translate to meaningful accuracy drops on task-level metrics. This may limit the practical relevance of SinkCast's correction for downstream accuracy.
