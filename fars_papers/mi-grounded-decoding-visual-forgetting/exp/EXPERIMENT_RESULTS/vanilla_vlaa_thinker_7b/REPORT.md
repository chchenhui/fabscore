# Vanilla Reasoning Baseline: VLAA-Thinker-7B

## Experiment Overview

Reproduce the vanilla reasoning baseline on VLAA-Thinker-7B with greedy decoding on MMStar and HallusionBench. This establishes the shared inference and evaluation pipeline for all subsequent experiments in the MI-grounded decoding project.

## Setup

- **Model**: `UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B` (bfloat16, Flash Attention 2)
- **System Prompt**: VLAA-Thinker thinking prompt ("You are VL-Thinking...") which triggers `<think>...</think><answer>...</answer>` format
- **Decoding**: Greedy (`do_sample=False`), `max_new_tokens=1000`
- **Hardware**: 4x A100-80GB GPUs, data-parallel sharding (each GPU processes 1/4 of dataset)
- **Answer Extraction**: Regex-based extraction from `<answer>` blocks; A/B/C/D for MMStar, Yes/No for HallusionBench

## Key Results

| Benchmark | Metric | Our Result | Published (VAPO paper) | Delta |
|-----------|--------|-----------|----------------------|-------|
| MMStar | Accuracy | **62.13%** | 49.7% | +12.4pp |
| HallusionBench | aAcc | **66.25%** | 54.7% | +11.6pp |
| HallusionBench (VD) | Accuracy | 60.24% | - | - |
| HallusionBench (VS) | Accuracy | 72.86% | - | - |

## Key Observations

### Discrepancy with Published Results

Our reproduced results are ~12pp above the published values from the VAPO paper (arXiv 2509.25848). After extensive investigation, the discrepancy is attributed to differences in evaluation methodology:

1. **Prompt Format**: The VAPO paper uses VLMEvalKit, which reformats MCQ questions with its own template (e.g., adds "Please select the correct answer from the options above." suffix, formats options as "A. text" instead of "A: text"). Our pipeline uses raw questions from HuggingFace datasets.

2. **Answer Extraction**: VLMEvalKit uses a multi-stage extraction: (a) option matching on last 5 tokens of the response, (b) regex character extraction, (c) GPT-based fallback for ambiguous cases. Our pipeline extracts from `<answer>` blocks using regex patterns. Simulating VLMEvalKit-style extraction on our outputs yields ~55-60% on MMStar (closer but still above 49.7%).

3. **HallusionBench Evaluation**: The official HallusionBench evaluation and VLMEvalKit use GPT-4 as a judge to determine if a Yes/No answer is correct, which is substantially more strict than regex-based Yes/No extraction.

4. **The published 49.7% is NOT from the original VLAA-Thinker paper**: The original VLAA-Thinker paper (arXiv 2504.11468) does not evaluate on MMStar or HallusionBench. The 49.7% and 54.7% were reproduced by the VAPO paper authors using VLMEvalKit.

### Internal Consistency

Despite the absolute value discrepancy, our pipeline is **internally consistent** and suitable for comparing decoding strategies (vanilla vs. visual replay vs. MI decoding). All experiments use the same:
- Model loading and prompting
- Dataset loading and question formatting
- Answer extraction logic
- Metric computation

The relative differences between methods are what matter for this project, not exact match with published absolute values.

## Runtime

- MMStar (1500 items, 4 GPUs): ~43 minutes
- HallusionBench (1129 items, 4 GPUs): ~35 minutes

## Files

- Raw predictions: `mi_decoding/outputs/vanilla_VLAA-Thinker-Qwen2.5VL-7B/{mmstar,hallusionbench}/`
- Results JSONs: `mi_decoding/results/vanilla_VLAA-Thinker-Qwen2.5VL-7B_{mmstar,hallusionbench}.json`
- Inference script: `mi_decoding/scripts/run_vanilla_baseline.py`
- Evaluation script: `mi_decoding/scripts/merge_and_evaluate.py`
- Shell launcher: `mi_decoding/scripts/run_vanilla_full.sh`
