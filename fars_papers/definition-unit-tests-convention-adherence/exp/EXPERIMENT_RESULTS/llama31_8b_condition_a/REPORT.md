# Condition A (Glossary-Only) -- Llama-3.1-8B-Instruct

## Experiment Overview

Evaluate the glossary-only prompting baseline (Condition A) on Llama-3.1-8B-Instruct across ErdosConventionsBench (300 items, 3 families). The model receives only the glossary snippet and the main question without any preceding checks. Uses chat-format inference with structured system/user messages (V3 prompt design). This provides a contrasting test bed to the math-specialized Qwen2.5-Math-7B-Instruct.

## Setup

- **Model**: meta-llama/Llama-3.1-8B-Instruct
- **Condition**: A (glossary-only, no checks)
- **Benchmark**: ErdosConventionsBench (300 items: 100 asymptotics, 100 completeness, 100 convolution)
- **Decoding**: Greedy (temperature=0), max_tokens=2048
- **Prompt version**: V3 (chat format with glossary-grounding system message)
- **Inference**: vLLM batched chat inference via TrainService (1x GPU)

## Key Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **56.7%** (170/300) |
| Asymptotics | 74.0% (74/100) |
| Completeness | 30.0% (30/100) |
| Convolution | 66.0% (66/100) |
| Alternate convention match rate | 11.0% (33/300) |

Note: Uses robust answer extraction (boxed + heuristic fallbacks for truncated outputs).

## Comparison with Qwen2.5-Math-7B-Instruct (Condition A)

| Metric | Qwen2.5-Math-7B | Llama-3.1-8B |
|--------|-----------------|--------------|
| Overall | 90.3% | 56.7% |
| Asymptotics | 99.0% | 74.0% |
| Completeness | 80.0% | 30.0% |
| Convolution | 92.0% | 66.0% |
| Alt rate | 6.3% | 11.0% |

## Key Observations

1. **Substantially lower baseline**: Llama-3.1-8B-Instruct achieves only 56.7% overall on Condition A, compared to 90.3% for the math-specialized Qwen model. This confirms that convention adherence is strongly model-dependent.

2. **Completeness is hardest**: Only 30% accuracy on the completeness family ("sufficiently large integers" vs "all positive integers"), indicating this general-purpose model struggles most with this subtle convention distinction.

3. **Asymptotics relatively easier**: 74% accuracy on asymptotics (O/o notation), still the easiest family but far below Qwen's near-ceiling 99%.

4. **Higher alternate rate**: 9.7% of predictions match the alternate convention (vs 6.3% for Qwen), suggesting Llama is more prone to defaulting to its parametric priors rather than following the glossary.

5. **Room for DUT improvement**: The low baseline provides substantial headroom to measure whether discriminative definition unit tests (Condition C) can improve adherence for a general-purpose model.

## Files

- Raw outputs: `dut_project/outputs/llama31_8b/condition_a.jsonl`
- Scoring results: `dut_project/results/llama31_8b/condition_a_results.json`
- Inference script: `dut_project/scripts/run_llama31_condition_a.sh`
- Scoring script: `dut_project/scripts/score_llama31_condition_a.py`
