# Optimization Iteration 1: Phantom Tagging + Self-Consistency

## Experiment Overview

This iteration combines two complementary improvements to Condition C (EACP):

1. **Phantom entity tagging**: Annotating inventory entries that never appear in the anonymized text with `[not in text]`, reducing confusion from large inventories.
2. **Self-consistency decoding**: Generating 5 samples at temperature=0.7 and selecting the answer via majority vote, preferring entity IDs over UNKNOWN.

## Setup

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Dataset**: ConFiQA-MC (6,000 full, 1,500 subset)
- **Decoding**: Self-consistency with n=5, temperature=0.7, majority vote
- **Infrastructure**: 2x GPU via TrainService, vLLM batch inference with tensor_parallel_size=2
- **Code changes**: `eacp/prompts/condition_c.py` (phantom tagging), `eacp/scripts/run_selfconsistency.py` (new script)

## Key Results

### Full ConFiQA-MC (6,000 examples)

| Method | Pc | Po | MR | EM |
|--------|---:|---:|---:|---:|
| Original C (greedy) | 68.98 | 12.38 | 15.22 | 75.68 |
| Tags-only (greedy) | 71.92 | 10.18 | 12.40 | 78.57 |
| **Tags + SC5** | **74.75** | **10.77** | **12.59** | **81.80** |
| Delta (vs original) | **+5.77** | **-1.61** | **-2.63** | **+6.12** |

### 1,500-Subset

| Method | Pc | Po | MR | EM |
|--------|---:|---:|---:|---:|
| Original C (greedy) | 68.07 | 12.20 | 15.20 | 75.07 |
| Tags-only (greedy) | 70.13 | 10.27 | 12.77 | 76.73 |
| **Tags + SC5** | **72.87** | **11.13** | **13.25** | **80.00** |
| Delta (vs original) | **+4.80** | **-1.07** | **-1.95** | **+4.93** |

## Key Observations

1. **Phantom tagging alone provides +2.94 Pc**: Annotating unused inventory entries helps the model focus on entities actually present in the text, reducing false UNKNOWN responses.

2. **Self-consistency adds another +2.83 Pc on top**: The majority vote mechanism converts borderline UNKNOWN/wrong responses into correct entity IDs by exploiting the stochasticity of sampling.

3. **Combined effect is +5.77 Pc**: The two optimizations are complementary -- tagging improves the base signal quality while SC5 aggregates over diverse samples.

4. **Po decreases by 1.61 points**: Less parametric recall, meaning the model is relying more on the context.

5. **EM improves by 6.12 points**: From 75.68 to 81.80, indicating substantially cleaner and more accurate outputs.

6. **MR decreases by 2.63 points**: The ratio of parametric-to-total correct answers drops from 15.22% to 12.59%.

7. **The two optimizations are methodologically simple and training-free**: No model modification, no fine-tuning. Phantom tagging is a prompt change; self-consistency is a standard inference-time technique.

## Output Files

- Full 6000 SC results: `eacp/outputs/C_llama31_8b_confiqa_mc_sc_full.jsonl`
- Full 6000 SC metrics: `eacp/outputs/C_llama31_8b_confiqa_mc_sc_full_metrics.json`
- 1500 subset SC results: `eacp/outputs/C_llama31_8b_confiqa_mc_1500_sc_full.jsonl`
- 1500 subset SC metrics: `eacp/outputs/C_llama31_8b_confiqa_mc_1500_sc_full_metrics.json`
- Full 6000 tags-only greedy: `eacp/outputs/C_llama31_8b_confiqa_mc.jsonl`
- 1500 subset tags-only greedy: `eacp/outputs/C_llama31_8b_confiqa_mc_1500.jsonl`
