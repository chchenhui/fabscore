# Qwen2.5-7B Greedy Baseline on Countdown and Mini Sudoku

## Experiment Overview

Evaluated Qwen2.5-7B (base model) with greedy decoding (temperature=0, single sample) on procedurally generated Countdown and 4x4 Mini Sudoku instances from Reasoning Gym. This establishes Condition A -- the standard AR single-sample baseline.

## Setup

- **Model**: Qwen/Qwen2.5-7B (base, 7.62B params)
- **Inference**: vLLM with bfloat16, greedy decoding (temperature=0)
- **Prompting**: 8-shot few-shot prompts (exemplars from seed=7777)
- **max_tokens**: 64 for both tasks
- **Stop sequence**: `\n\n`
- **Hardware**: 1x A100-SXM4-80GB (via TrainService)

### Datasets
| Dataset | Size | Seed | Source |
|---------|------|------|--------|
| Countdown test | 500 | 2024 | reasoning_gym.create_dataset('countdown') |
| Countdown cal | 50 | 9999 | reasoning_gym.create_dataset('countdown') |
| Mini Sudoku test | 500 | 2024 | reasoning_gym.create_dataset('mini_sudoku') |
| Mini Sudoku cal | 50 | 9999 | reasoning_gym.create_dataset('mini_sudoku') |

### Evaluation
Binary accuracy using Reasoning Gym's `dataset.score_answer()` -- score=1.0 counted as correct, anything else as incorrect.

## Key Results

| Task | Method | Accuracy | Correct | Total |
|------|--------|----------|---------|-------|
| Countdown | Qwen2.5-7B greedy | **6.00%** | 30 | 500 |
| Mini Sudoku | Qwen2.5-7B greedy | **16.80%** | 84 | 500 |

## Key Observations

1. Qwen2.5-7B base model with greedy decoding performs poorly on both procedural reasoning tasks, consistent with expectations for a non-finetuned base model on structured problems.
2. Countdown accuracy (6%) is notably lower than Mini Sudoku (16.8%), suggesting that exact arithmetic/expression composition is harder for the base model than grid completion.
3. These numbers serve as the lower-bound baseline (Condition A) against which Dream diffusion (Condition C) and compute-matched best-of-k (Condition B) will be compared.
4. Inference was very fast (~2-3 seconds total for 500 instances per task) using vLLM batch processing.
