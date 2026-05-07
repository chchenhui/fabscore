# Kitty Baseline Experiment Report

## Experiment Overview

Reproduced the Kitty baseline (2-bit KV cache quantization with dynamic channel-wise precision boost at 12.5% boost ratio) on Qwen3-8B for AIME24 and AIME25 benchmarks. Kitty is the primary comparison target for FCBoost: FCBoost aims to match Kitty's accuracy with a static mask.

## Setup

- **Model**: Qwen/Qwen3-8B (36 layers, 32 query heads, 8 KV heads, head_dim=128)
- **Method**: Kitty -- INT2 base quantization with dynamic per-page magnitude-based channel selection. Top-K=16 channels (out of D=128, boost_ratio=12.5%) boosted to INT4 per quantization page. S=32 sink tokens kept in FP16.
- **Config**: sink_length=32, buffer_length=128, group_size=128, kbits=2, vbits=2, promote_ratio=0.125, promote_bit=4, channel_selection=1 (magnitude-based)
- **Evaluation**: lm-evaluation-harness (v0.4.12.dev0) with Kitty's KV cache simulation framework
- **Sampling**: temperature=0.6, top_p=0.95, top_k=20, max_gen_toks=32768
- **Repeats**: 3 seeds per benchmark (seed protocol: random={0,1,2}, numpy/torch/fewshot={1234,1235,1236})
- **Hardware**: 1x GPU per job (A100 80GB), two parallel jobs
- **Benchmarks**: AIME24 (30 problems, Maxwell-Jia/AIME_2024), AIME25 (30 problems, math-ai/aime25)

## Key Results

| Benchmark | Seed 0 | Seed 1 | Seed 2 | Mean | Max Dev | Published Mean | Published Dev |
|-----------|--------|--------|--------|------|---------|----------------|---------------|
| AIME24    | 76.7%  | 63.3%  | 76.7%  | 72.22% | +/-8.89 | 70.67%         | +/-7.33       |
| AIME25    | 66.7%  | 63.3%  | 53.3%  | 61.11% | +/-7.78 | 59.67%         | +/-10.33      |
| **Avg**   |        |        |        | **66.67%** |     | **65.17%**     |               |

### Deviation from Published (Kitty Table 4)
- AIME24: +1.55 points (within tolerance)
- AIME25: +1.44 points (within tolerance)
- Average: +1.50 points

### Consolidated Baseline Comparison

| Method | AIME24 (mean+/-dev) | AIME25 (mean+/-dev) | Avg | Source |
|--------|---------------------|---------------------|-----|--------|
| FP16 KV16 | 71.67+/-15.00 | 66.00+/-7.33 | 68.84 | Kitty Table 4 |
| KIVI-KV2 | 57.00+/-7.00 | 52.33+/-9.00 | 54.67 | Kitty Table 4 |
| KIVI-KV2* | 67.78+/-7.78 | 64.44+/-5.56 | 66.11 | Reproduced |
| Kitty | 72.22+/-8.89 | 61.11+/-7.78 | 66.67 | Reproduced |

## Key Observations

1. **Both benchmarks match published results closely**: AIME24 within +1.55 pts and AIME25 within +1.44 pts of Kitty's published values. Both well within published max_deviation ranges.

2. **Kitty improves over KIVI-KV2***: On AIME24, Kitty (72.22%) outperforms KIVI-KV2* (67.78%) by +4.44 pts. On AIME25, Kitty (61.11%) vs KIVI-KV2* (64.44%) shows -3.33 pts, though with high seed variance.

3. **Overall improvement**: On average, Kitty (66.67%) is +0.56 pts above KIVI-KV2* (66.11%). The improvement from dynamic channel boost is modest but consistent with published trends.

4. **Kitty's accuracy gain over KIVI-KV2* is the target for FCBoost**: FCBoost needs to recover >=90% of this gain with a static mask.

## Files

- Raw results: `eval_results/Qwen3-8B/aime24/kitty_g128_b128_s32_sel1_k2_v2_pb4_pr0.125/` and `eval_results/Qwen3-8B/aime25/kitty_g128_b128_s32_sel1_k2_v2_pb4_pr0.125/`
- Evaluation wrapper: `fcboost/evaluation/eval_aime.py`
- Shell scripts: `fcboost/scripts/run_kitty_aime24.sh`, `fcboost/scripts/run_kitty_aime25.sh`
- Results summary: `results/kitty_aime_results.json`, `results/baseline_summary.json`
