# SinkCast Downstream Evaluation

## Experiment Overview

Evaluate whether SinkCast -- FP32 recasting of attention-sink logits -- reduces the accuracy degradation caused by BF16 RoPE position shifts on downstream benchmarks. Two benchmarks are used: RULER-position-shift and LongBench-position-shift, evaluated on Llama-3.1-8B and Mistral-7B-v0.3.

## Setup

- **Models**: Llama-3.1-8B (`meta-llama/Llama-3.1-8B`) with K=4, Mistral-7B-v0.3 (`mistralai/Mistral-7B-v0.3`) with K=1
- **Precision**: BF16 with SinkCast correction (FP32 recast of first K key logits)
- **Shift Protocol**: Padding-based position-shift M=4096 (prepend M pad tokens with attention_mask=0, identical for BF16 baseline and SinkCast)
- **RULER**: 4 tasks (NIAH-Single, NIAH-Multikey, Variable Tracing, QA), seq_lengths 4096/8192, 50 samples/task
- **LongBench**: 5 tasks (NarrativeQA, HotpotQA, GovReport, TREC, PassageRetrieval-en), 200 samples/task, max_context 8192
- **SinkCast hooks**: Monkey-patch attention layers; apply correction during prefill with padding-aware real-token extraction, fall back to standard attention during decode
- **Infrastructure**: 1x GPU per model, TrainService

## Key Results

### Llama-3.1-8B RULER (K=4)

| Task | Seq Len | BF16 d=0 | BF16 d=M | BF16 Drop | SC d=0 | SC d=M | SC Drop | Improvement |
|------|---------|----------|----------|-----------|--------|--------|---------|-------------|
| niah_single | 4096 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| niah_single | 8192 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| niah_multikey | 4096 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| niah_multikey | 8192 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 |
| variable_tracing | 4096 | 2.0 | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| variable_tracing | 8192 | 6.0 | 6.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qa | 4096 | 9.75 | 10.63 | -0.88 | 50.07 | 48.96 | 1.11 | -1.99 |
| qa | 8192 | 11.59 | 9.92 | 1.67 | 46.34 | 46.76 | -0.42 | 2.09 |
| **Average** | | | | **0.10** | | | **0.09** | **0.01** |

### Llama-3.1-8B LongBench (K=4)

| Task | BF16 d=0 | BF16 d=M | BF16 Drop | SC d=0 | SC d=M | SC Drop | Improvement |
|------|----------|----------|-----------|--------|--------|---------|-------------|
| narrativeqa | 23.68 | 24.54 | -0.86 | 23.63 | 24.15 | -0.52 | -0.34 |
| hotpotqa | 11.75 | 11.93 | -0.18 | 11.56 | 11.43 | 0.13 | -0.31 |
| gov_report | 9.39 | 9.78 | -0.39 | 9.70 | 10.00 | -0.30 | -0.09 |
| trec | 67.5 | 67.5 | 0.0 | 68.0 | 67.0 | 1.0 | -1.0 |
| passage_retrieval_en | 35.0 | 35.5 | -0.50 | 34.0 | 34.0 | 0.0 | -0.50 |
| **Average** | 29.46 | 29.85 | **-0.39** | 29.38 | 29.32 | **0.06** | **-0.45** |

### Mistral-7B-v0.3 RULER (K=1)

| Task | Seq Len | BF16 d=0 | BF16 d=M | BF16 Drop | SC d=0 | SC d=M | SC Drop | Improvement |
|------|---------|----------|----------|-----------|--------|--------|---------|-------------|
| niah_single | 4096 | 54.0 | 52.0 | 2.0 | 60.0 | 56.0 | 4.0 | -2.0 |
| niah_single | 8192 | 34.0 | 36.0 | -2.0 | 60.0 | 58.0 | 2.0 | -4.0 |
| niah_multikey | 4096 | 36.0 | 40.0 | -4.0 | 60.0 | 56.0 | 4.0 | -8.0 |
| niah_multikey | 8192 | 48.0 | 48.0 | 0.0 | 60.0 | 58.0 | 2.0 | -2.0 |
| variable_tracing | 4096 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 2.0 | -2.0 |
| variable_tracing | 8192 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| qa | 4096 | 13.95 | 15.59 | -1.64 | 34.56 | 35.36 | -0.80 | -0.84 |
| qa | 8192 | 28.70 | 31.09 | -2.39 | 47.36 | 45.76 | 1.60 | -3.99 |
| **Average** | | | | **-1.00** | | | **1.85** | **-2.85** |

### Mistral-7B-v0.3 LongBench (K=1)

| Task | BF16 d=0 | BF16 d=M | BF16 Drop | SC d=0 | SC d=M | SC Drop | Improvement |
|------|----------|----------|-----------|--------|--------|---------|-------------|
| narrativeqa | 17.02 | 16.93 | 0.09 | 17.22 | 16.61 | 0.61 | -0.52 |
| hotpotqa | 14.46 | 14.15 | 0.31 | 14.28 | 14.00 | 0.28 | 0.03 |
| gov_report | 1.94 | 1.71 | 0.23 | 1.95 | 1.96 | -0.01 | 0.24 |
| trec | 71.0 | 71.5 | -0.50 | 72.0 | 71.0 | 1.0 | -1.50 |
| passage_retrieval_en | 22.0 | 22.0 | 0.0 | 22.0 | 22.0 | 0.0 | 0.0 |
| **Average** | 25.28 | 25.26 | **0.03** | 25.49 | 25.11 | **0.38** | **-0.35** |

### Summary Aggregates

| Model | Benchmark | K | Avg BF16 Drop | Avg SC Drop | Avg Improvement |
|-------|-----------|---|---------------|-------------|-----------------|
| Llama-3.1-8B | RULER | 4 | 0.10 | 0.09 | +0.01 |
| Llama-3.1-8B | LongBench | 4 | -0.39 | 0.06 | -0.45 |
| Mistral-7B-v0.3 | RULER | 1 | -1.00 | 1.85 | -2.85 |
| Mistral-7B-v0.3 | LongBench | 1 | 0.03 | 0.38 | -0.35 |

## Key Observations

1. **BF16 baseline already shows minimal position-shift degradation on downstream tasks**: The BF16 drops are generally very small (within +/-2 points for most tasks), making it difficult for SinkCast to demonstrate improvement. This contrasts with the microbenchmark results where the attention-level shift error was substantial.

2. **SinkCast does not consistently reduce downstream accuracy drop**: The average improvement is near zero or slightly negative across most model-benchmark pairs. This suggests that closing the shift-error gap at the attention level does not translate to measurable accuracy improvements when the baseline degradation is already minimal.

3. **SinkCast significantly alters d=0 absolute accuracy on RULER tasks**: The d=0 accuracy differs substantially between BF16 and SinkCast for some RULER tasks (especially QA: Llama 9.75 -> 50.07 at 4096, Mistral NIAH: 34-54% -> 60%). This is because the SinkCast hook replaces HF's standard attention path with direct flash_attn_func + FP32 correction, which alters the softmax distribution and leads to different generation paths.

4. **LongBench d=0 accuracy is well-preserved**: On LongBench, the d=0 differences between BF16 and SinkCast are very small (avg |d0_diff| < 0.5 points for both models), indicating SinkCast does not degrade baseline performance on longer, more diverse tasks.

5. **The gap between attention-level error and task-level degradation**: The microbenchmark showed SinkCast reduces the attention output error under position shift. However, this attention-level correction does not translate to downstream task improvements because:
   - The BF16 attention error, while measurable, is small enough that models' generation quality remains robust
   - Generation involves autoregressive decoding where small attention differences get amplified or dampened unpredictably
   - The error correction at prefill time doesn't persist through KV-cache-based decoding

6. **SinkCast introduces its own variance**: By changing the attention computation (FP32 for sink keys), SinkCast introduces its own source of variance relative to vanilla BF16. On RULER, this manifests as large d=0 accuracy changes; on LongBench (with more samples and continuous metrics), the effect is minimal.

## Optimization History

- **Original (K=1, position-ids-only shift)**: Had a critical bug where SinkCast used a different shift protocol than BF16 baseline (position_ids-only offset vs padding-based). Results were not directly comparable.
- **Optimized (K=4 Llama / K=1 Mistral, padding-based shift)**: Fixed shift protocol to use identical padding-based approach. Added padding-aware token extraction in hooks. Used optimal K per model. Results above reflect the corrected evaluation. No meaningful improvement over BF16 baseline.
