# Condition A (Glossary-Only) -- Qwen2.5-Math-7B-Instruct

## Experiment Overview

Evaluate the glossary-only prompting baseline (Condition A) on Qwen2.5-Math-7B-Instruct across ErdosConventionsBench (300 items, 3 families). The model receives only the glossary snippet and the main question without any preceding checks. Uses chat-format inference with structured system/user messages (V3 prompt design).

## Setup

- **Model**: Qwen/Qwen2.5-Math-7B-Instruct
- **Condition**: A (glossary-only, no checks)
- **Benchmark**: ErdosConventionsBench (300 items: 100 asymptotics, 100 completeness, 100 convolution)
- **Decoding**: Greedy (temperature=0), max_tokens=2048
- **Prompt version**: V3 (chat format with glossary-grounding system message)
- **Inference**: vLLM batched chat inference via TrainService (1x GPU)

## Key Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **90.3%** (271/300) |
| Asymptotics | 99.0% (99/100) |
| Completeness | 80.0% (80/100) |
| Convolution | 92.0% (92/100) |
| Alternate convention match rate | 6.3% (19/300) |

## Key Observations

1. **Strong baseline**: The chat-format prompt with glossary-grounding system message achieves 90.3% overall, a substantial improvement over the original text-format prompt (56.3%).

2. **Near-ceiling on asymptotics**: 99% accuracy, indicating the model's parametric knowledge aligns well with the glossary's O/o notation convention.

3. **Moderate on completeness**: 80% accuracy. The "sufficiently large integers" vs "all positive integers" distinction remains the hardest family.

4. **Strong on convolution**: 92% accuracy, greatly improved from the original 23% due to the chat format and clearer instructions.

## Files

- Raw outputs: `dut_project/outputs/qwen25_math_7b_v3/condition_a.jsonl`
- Inference script: `dut_project/scripts/run_optimized_v3.sh`
