# YaRN-Llama-2-7b-64k: All Conditions Evaluation

## Experiment Overview

Evaluate the Debiased One-Pass Attention Sorting method on a second model (NousResearch/Yarn-Llama-2-7b-64k) with stronger recency bias. The debiased method was optimized for YaRN using full-sort by divisive debiased scores.

## Setup

- **Model**: NousResearch/Yarn-Llama-2-7b-64k (YaRN position encoding, not instruction-tuned)
- **Benchmark**: SynthWiki (madlibs1.csv, 200 examples per seed)
- **Context Length**: ~28K tokens of junk context
- **Seeds**: 42, 123, 456
- **Prompt Type**: wizard (non-instruct model convention)
- **Decoding**: Greedy
- **Debiasing**: alpha=0.005, B=40, mean aggregation, divisive mode, full-sort by debiased scores

## Key Results

| Condition | Seed 42 | Seed 123 | Seed 456 | Mean | Std | Prefills/Query |
|-----------|---------|----------|----------|------|-----|----------------|
| No Sorting | 36.0% | 33.5% | 38.0% | 35.83% | 2.25% | 1 |
| Attn Sort k=1 | 47.0% | 45.0% | 49.5% | 47.17% | 2.25% | 2 |
| **Debiased k=1** | **57.5%** | **53.5%** | **56.5%** | **55.83%** | **2.08%** | **2** |
| Attn Sort k=5 | 71.5% | 73.5% | 67.0% | 70.67% | 3.33% | 6 |

### Regime Check
- k=5 (70.67%) - No Sorting (35.83%) = **34.84pp >= 3.0pp** -- PASS

### Debiased k=1 vs Uncalibrated k=1
- **Wins: 3/3 seeds** (all seeds show improvement)
- Per-seed improvements: +10.5pp, +8.5pp, +7.0pp
- Mean improvement: **+8.67pp**

### Debiased k=1 vs k=5
- k=5 still outperforms debiased k=1 (70.67% vs 55.83%)
- Gap: -14.84pp (reduced from -22.83pp with previous minimal-swap method)
- Debiased k=1 closes 37% of the k=1-to-k=5 gap using only 2 prefill passes vs 6

## Cross-Model Comparison

| Model | No Sort | k=1 | Debiased k=1 | k=5 | Debiased Gain |
|-------|---------|-----|-------------|------|---------------|
| LLaMA-2-7B-32K-Instruct | 72.83% | 94.83% | 94.83% | 95.50% | +0.00pp |
| YaRN-Llama-2-7b-64k | 35.83% | 47.17% | 55.83% | 70.67% | **+8.67pp** |

## Key Observations

1. **Full-sort debiasing is transformative on high-bias models**: On YaRN, full-sort by debiased scores yields +8.67pp over k=1, closing 37% of the k=1-to-k=5 gap. This is because the gold doc rarely ranks highest in raw attention (9.5% of the time) due to extreme recency bias.

2. **Divisive debiasing corrects multiplicative bias**: YaRN's recency bias is multiplicative (last doc gets ~18x middle position attention). Divisive mode (a/bias) correctly normalizes this, unlike additive mode (a-bias) which under-corrects the tail.

3. **Strategy must be adaptive**: On LLaMA-2-32K-Instruct where baseline is already 94.83%, minimal-swap preserves the good raw ordering. On YaRN where baseline is 47.17%, full-sort is needed to overcome the severe position bias.

4. **Mean gold doc rank improvement**: Debiased scoring moves the gold doc an average of +14.5 positions closer to the end (best position), from rank 126.6 to 141.1 out of ~166 documents.
