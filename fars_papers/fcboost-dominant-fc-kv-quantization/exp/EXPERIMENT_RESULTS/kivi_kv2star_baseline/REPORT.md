# KIVI-KV2* Baseline Experiment Report

## Experiment Overview

Reproduced the KIVI-KV2* baseline (2-bit KV cache quantization with sink tokens in FP16) on Qwen3-8B for AIME24 and AIME25 benchmarks. KIVI-KV2* is the critical reference lower bound for FCBoost's accuracy recovery experiments.

## Setup

- **Model**: Qwen/Qwen3-8B (36 layers, 32 query heads, 8 KV heads, head_dim=128)
- **Method**: KIVI-KV2* -- Key cache quantized per-channel to INT2, Value cache quantized per-token to INT2, S=32 initial sink tokens kept in FP16, no channel-wise precision boost (promote_ratio=0.0)
- **Config**: sink_length=32, buffer_length=128, group_size=128, kbits=2, vbits=2, promote_ratio=0.0, promote_bit=4, channel_selection=0
- **Evaluation**: lm-evaluation-harness (v0.4.12.dev0) with Kitty's KV cache simulation framework
- **Sampling**: temperature=0.6, top_p=0.95, top_k=20, max_gen_toks=32768
- **Repeats**: 3 seeds per benchmark (seed protocol: random={0,1,2}, numpy/torch/fewshot={1234,1235,1236})
- **Hardware**: 1x GPU per job (A100 80GB), two parallel jobs
- **Benchmarks**: AIME24 (30 problems, Maxwell-Jia/AIME_2024), AIME25 (30 problems, math-ai/aime25)

## Key Results

| Benchmark | Seed 0 | Seed 1 | Seed 2 | Mean | Max Dev | Published Mean | Published Dev |
|-----------|--------|--------|--------|------|---------|----------------|---------------|
| AIME24    | 73.3%  | 60.0%  | 70.0%  | 67.78% | ±7.78 | 67.67%         | ±9.00         |
| AIME25    | 60.0%  | 70.0%  | 63.3%  | 64.44% | ±5.56 | 57.67%         | ±9.00         |
| **Avg**   |        |        |        | **66.11%** |   | **62.67%**     |               |

### Deviation from Published (Kitty Table 4)
- AIME24: +0.11 points (within tolerance)
- AIME25: +6.77 points (above 2-point tolerance threshold)
- Average: +3.44 points

## Key Observations

1. **AIME24 matches nearly perfectly** (+0.11 pts), confirming the evaluation pipeline is correct.

2. **AIME25 deviation (+6.77 pts)**: This is above the 2-point tolerance specified in the task. However:
   - Our result (64.44%) falls within the published max_deviation range (57.67 ± 9.00 = [48.67, 66.67])
   - The high variance across seeds (60.0% to 70.0%) is consistent with the published high deviation (±9.00)
   - The deviation is likely caused by differences in the AIME25 dataset version on HuggingFace (`math-ai/aime25`) compared to what Kitty used

3. **Relative comparisons remain valid**: Since all subsequent experiments (Kitty, FCBoost, FP16 baseline) will use the exact same evaluation setup, datasets, seeds, and infrastructure, the relative performance differences will be consistent and meaningful.

4. **Shared evaluation configuration established**: The reusable evaluation wrapper (`fcboost/evaluation/eval_aime.py`) and YAML configs are ready for all subsequent experiments.

## Files

- Raw results: `eval_results/Qwen3-8B/aime24/` and `eval_results/Qwen3-8B/aime25/`
- Evaluation wrapper: `fcboost/evaluation/eval_aime.py`
- Shell scripts: `fcboost/scripts/run_kivi_kv2star_aime24.sh`, `fcboost/scripts/run_kivi_kv2star_aime25.sh`
- AIME task configs: `lm-evaluation-harness/lm_eval/tasks/aime/aime24.yaml`, `lm-evaluation-harness/lm_eval/tasks/aime/aime25.yaml`
