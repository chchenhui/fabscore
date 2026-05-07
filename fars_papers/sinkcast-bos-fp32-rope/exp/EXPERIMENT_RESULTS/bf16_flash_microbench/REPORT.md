# BF16 FlashAttention Shift Microbenchmark Baseline

## Experiment Overview

Measures how standard BF16 FlashAttention breaks RoPE shift-invariance on Llama-3.1-8B and Mistral-7B-v0.3. Identical token sequences are evaluated at different global position offsets to quantify the shift-error magnitude.

## Setup

- **Models**: meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-v0.3 (loaded in BF16 with flash_attention_2)
- **Sequence lengths**: 512, 1024, 2048, 4096
- **Shift pairs** (delta1, delta2): (0,16), (0,256), (0,4096)
- **Key indices** for D_logit: j in {0, 1, 2, 8, 64}
- **Determinism**: model.eval(), seed=42, greedy decoding, dropout=0
- **GPU**: 1x A100-SXM4-80GB per model
- **Input**: Fixed natural-language text about AI history, repeated/truncated to target seq_len, with BOS token at position 0

## Key Results

### D_logit(j) Distribution -- Llama-3.1-8B

| seq_len | shift | D_logit(0) | D_logit(1) | D_logit(2) | D_logit(8) | D_logit(64) | j0_fraction |
|---------|-------|------------|------------|------------|------------|-------------|-------------|
| 512 | (0,16) | 3.57 | 17.25 | 16.20 | 17.94 | 19.06 | 4.8% |
| 512 | (0,256) | 3.76 | 17.53 | 17.12 | 16.77 | 17.81 | 5.1% |
| 512 | (0,4096) | 4.01 | 18.33 | 19.96 | 20.87 | 19.79 | 4.8% |
| 1024 | (0,16) | 3.51 | 18.39 | 16.98 | 15.28 | 17.11 | 4.9% |
| 1024 | (0,256) | 3.64 | 19.15 | 17.45 | 16.74 | 19.12 | 4.8% |
| 1024 | (0,4096) | 4.08 | 22.50 | 19.44 | 17.21 | 20.02 | 4.9% |
| 2048 | (0,16) | 3.76 | 19.52 | 18.41 | 17.13 | 20.03 | 4.8% |
| 2048 | (0,256) | 3.97 | 21.01 | 19.16 | 17.63 | 21.54 | 4.8% |
| 2048 | (0,4096) | 4.48 | 22.57 | 20.49 | 19.45 | 22.65 | 5.0% |
| 4096 | (0,16) | 3.83 | 20.07 | 18.91 | 18.36 | 22.25 | 4.6% |
| 4096 | (0,256) | 4.05 | 19.94 | 19.67 | 18.55 | 20.79 | 4.9% |
| 4096 | (0,4096) | 4.56 | 23.41 | 21.87 | 20.32 | 23.92 | 4.8% |

### D_logit(j) Distribution -- Mistral-7B-v0.3

| seq_len | shift | D_logit(0) | D_logit(1) | D_logit(2) | D_logit(8) | D_logit(64) | j0_fraction |
|---------|-------|------------|------------|------------|------------|-------------|-------------|
| 512 | (0,16) | 2.93 | 8.13 | 8.05 | 9.10 | 8.75 | 7.9% |
| 512 | (0,256) | 3.30 | 8.16 | 8.00 | 9.94 | 9.47 | 8.5% |
| 512 | (0,4096) | 3.35 | 8.27 | 8.34 | 9.39 | 9.17 | 8.7% |
| 1024 | (0,16) | 2.82 | 8.06 | 7.67 | 8.25 | 7.92 | 8.1% |
| 1024 | (0,256) | 3.02 | 8.12 | 7.75 | 8.90 | 9.52 | 8.1% |
| 1024 | (0,4096) | 3.34 | 8.58 | 8.94 | 9.16 | 8.78 | 8.6% |
| 2048 | (0,16) | 3.31 | 9.31 | 9.42 | 10.74 | 9.47 | 7.8% |
| 2048 | (0,256) | 3.42 | 9.84 | 9.59 | 10.21 | 10.01 | 8.0% |
| 2048 | (0,4096) | 3.84 | 10.14 | 10.39 | 10.72 | 9.78 | 8.5% |
| 4096 | (0,16) | 3.43 | 10.20 | 9.89 | 10.49 | 9.55 | 7.9% |
| 4096 | (0,256) | 3.70 | 10.18 | 10.41 | 10.79 | 10.79 | 8.1% |
| 4096 | (0,4096) | 4.08 | 10.86 | 11.37 | 11.89 | 10.57 | 8.4% |

### Output-Logit Drift

| Model | seq_len | shift | max_drift | mean_drift |
|-------|---------|-------|-----------|------------|
| Llama-3.1-8B | 512 | (0,16) | 1.11 | 0.0738 |
| Llama-3.1-8B | 512 | (0,256) | 1.63 | 0.0758 |
| Llama-3.1-8B | 512 | (0,4096) | 1.44 | 0.0793 |
| Llama-3.1-8B | 1024 | (0,16) | 1.57 | 0.0659 |
| Llama-3.1-8B | 1024 | (0,256) | 1.77 | 0.0682 |
| Llama-3.1-8B | 1024 | (0,4096) | 1.76 | 0.0737 |
| Llama-3.1-8B | 2048 | (0,16) | 5.36 | 0.0523 |
| Llama-3.1-8B | 2048 | (0,256) | 6.13 | 0.0529 |
| Llama-3.1-8B | 2048 | (0,4096) | 5.25 | 0.0579 |
| Llama-3.1-8B | 4096 | (0,16) | 4.56 | 0.0427 |
| Llama-3.1-8B | 4096 | (0,256) | 5.63 | 0.0439 |
| Llama-3.1-8B | 4096 | (0,4096) | 7.43 | 0.0484 |
| Mistral-7B-v0.3 | 512 | (0,16) | 0.32 | 0.0201 |
| Mistral-7B-v0.3 | 512 | (0,256) | 0.66 | 0.0217 |
| Mistral-7B-v0.3 | 512 | (0,4096) | 0.84 | 0.0212 |
| Mistral-7B-v0.3 | 1024 | (0,16) | 1.28 | 0.0185 |
| Mistral-7B-v0.3 | 1024 | (0,256) | 0.63 | 0.0191 |
| Mistral-7B-v0.3 | 1024 | (0,4096) | 1.63 | 0.0210 |
| Mistral-7B-v0.3 | 2048 | (0,16) | 6.38 | 0.0190 |
| Mistral-7B-v0.3 | 2048 | (0,256) | 5.63 | 0.0189 |
| Mistral-7B-v0.3 | 2048 | (0,4096) | 3.89 | 0.0202 |
| Mistral-7B-v0.3 | 4096 | (0,16) | 3.02 | 0.0185 |
| Mistral-7B-v0.3 | 4096 | (0,256) | 2.45 | 0.0189 |
| Mistral-7B-v0.3 | 4096 | (0,4096) | 2.29 | 0.0204 |

## Key Observations

1. **D_logit(0) does NOT dominate the raw logit shift-error**. Across all configurations:
   - Llama-3.1-8B: j0_fraction ranges from 4.6% to 5.1% (consistently ~5%)
   - Mistral-7B-v0.3: j0_fraction ranges from 7.8% to 8.7% (consistently ~8%)
   - D_logit(0) is actually the *smallest* among the 5 sampled key indices, typically 3-5x smaller than D_logit for other keys

2. **D_logit increases with shift magnitude**: Larger position offsets (0,4096) produce slightly higher D_logit values than small shifts (0,16), consistent with Wang et al.'s finding that BF16 rounding error grows with absolute position.

3. **Output-logit drift is measurable but moderate**:
   - Llama-3.1-8B: max_drift ranges from 1.1 to 7.4, mean_drift from 0.04 to 0.08
   - Mistral-7B-v0.3: max_drift ranges from 0.3 to 6.4, mean_drift from 0.02 to 0.02
   - Max drift tends to increase with sequence length

4. **Mistral shows lower overall shift-error**: Mistral's D_logit values are roughly half of Llama's, and mean_drift is about 3x smaller.

5. **Note on j=0 localization**: The raw D_logit metric measures absolute pre-softmax logit differences. Wang et al.'s finding about j=0 dominance is about *attention weight* (post-softmax) differences, which is amplified by softmax when the sink key has high attention weight. The raw logit error at j=0 being smaller while still dominating post-softmax is consistent if the sink logit is near the max logit (softmax amplification). This distinction is important for interpreting SinkCast's effectiveness.
