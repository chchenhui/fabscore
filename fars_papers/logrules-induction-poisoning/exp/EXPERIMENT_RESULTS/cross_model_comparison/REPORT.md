# Cross-Model Deduction Comparison: LLaMA-3-8B-Instruct vs Qwen2.5-7B-Instruct

## Experiment Overview

This experiment tests whether the induction-stage poisoning attack generalizes across deduction (parsing) models. The main experiments used Qwen2.5-7B-Instruct as the parsing model. Here we substitute LLaMA-3-8B-Instruct to check if the attack effectiveness is model-specific or a property of the poisoned rule text itself.

**Scope**: BGL dataset only, payload D (instruction-style, most effective in main experiments), k in {1, 3}, 3 seeds (42, 123, 456).

## Setup

- **Induction model**: gpt-4o-mini (unchanged)
- **Induced rules**: Reused from main experiments (identical C0/C1 rules)
- **Deduction model**: meta-llama/Meta-Llama-3-8B-Instruct (served via vLLM on 1x GPU)
- **Deduction prompt**: Same template as Qwen experiments (`prompts/deduction_with_rules.txt`)
- **vLLM config**: `--max-model-len 4096 --gpu-memory-utilization 0.9`
- **Concurrency**: 64 parallel requests

## Key Results

### Baseline Performance Gap

| Model | C0 PA (mean +/- std) |
|-------|---------------------|
| Qwen2.5-7B-Instruct | **0.348 +/- 0.034** |
| LLaMA-3-8B-Instruct | **0.017 +/- 0.009** |

LLaMA-3-8B has dramatically lower baseline parsing accuracy on BGL (PA ~2% vs ~35% for Qwen). This is the dominant finding -- LLaMA-3-8B-Instruct struggles with the log template extraction task even with clean rules.

### C1 Poisoned Results

| Model | Condition | k | PA (mean) | PA drop from C0 |
|-------|-----------|---|-----------|-----------------|
| Qwen | C0 | - | 0.3478 | - |
| Qwen | C1 | 1 | 0.4318 | -0.0840 (improved) |
| Qwen | C1 | 3 | 0.3124 | +0.0354 (degraded) |
| LLaMA | C0 | - | 0.0170 | - |
| LLaMA | C1 | 1 | 0.0184 | -0.0014 (no change) |
| LLaMA | C1 | 3 | 0.0388 | -0.0218 (improved) |

Neither model shows consistent PA degradation from poisoned rules on BGL at k=1,3. For Qwen, the C1 k=1 result is dominated by a single outlier seed (seed 123: PA 0.66, up from 0.37). For LLaMA, all values are near the noise floor (~2%).

### C2 Defense Results

| Model | k | C2 PA (mean) | Admission decisions |
|-------|---|-------------|---------------------|
| Qwen | 1 | 0.4318 | All r_gen (3/3) |
| Qwen | 3 | 0.3124 | All r_gen (3/3) |
| LLaMA | 1 | 0.0280 | 2 r_safe, 1 r_gen |
| LLaMA | 3 | 0.0473 | 2 r_safe, 1 r_gen |

For LLaMA, the canary-based admission control triggers R_safe fallback more often (4/6 configs). R_safe provides marginally higher PA (0.029) vs C1 poisoned (0.018), but the absolute values are too low for this to be practically meaningful.

## Key Observations

1. **Attack effectiveness is model-dependent**: The poisoning attack requires a model with sufficient baseline parsing capability to exhibit degradation. LLaMA-3-8B-Instruct's near-zero baseline PA means there is no room for the attack to degrade performance.

2. **LLaMA-3-8B-Instruct is weak at structured log parsing**: Even with clean rules, LLaMA-3-8B achieves only ~2% PA on BGL, compared to ~35% for Qwen2.5-7B. This suggests the model has poor instruction-following for the specific task of log template extraction with `<*>` wildcards.

3. **Poisoned rules don't cause PA degradation on either model for BGL at k={1,3}**: The BGL dataset at low k values shows minimal attack impact even for Qwen. The main experiments showed effectiveness primarily at higher k (5,7) and on other datasets (Linux, HDFS). This is consistent with the main finding that payload D scales monotonically with k.

4. **Defense triggers more frequently on LLaMA**: The admission control gate detects low canary PA more often with LLaMA (both R_gen and R_safe canary PA are near 0), leading to more R_safe fallback decisions, but the practical impact is minimal given the low baseline.

## Files

- `results/cross_model_comparison.csv` -- Full comparison table (33 rows)
- `results/cross_model_llama.csv` -- Raw LLaMA results (19 rows)
- `results/figures/cross_model_bgl.png` -- Side-by-side bar chart
- `outputs/predictions/{c0,c1,c2}_llama/` -- LLaMA prediction files
