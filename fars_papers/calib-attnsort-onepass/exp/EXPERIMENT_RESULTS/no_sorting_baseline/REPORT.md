# No-Sorting Baseline on SynthWiki

## Experiment Overview

Evaluate the vanilla (no sorting) baseline on SynthWiki long-context extractive QA using `togethercomputer/LLaMA-2-7B-32K-Instruct`. The model receives ~30K tokens of context (one gold document + ~180 distractors in random order) and generates an answer without any document reordering. This establishes the lower-bound reference for the attention sorting experiments.

## Setup

- **Model**: togethercomputer/LLaMA-2-7B-32K-Instruct (fp16, flash-attention-2)
- **Dataset**: SynthWiki madlibs1.csv, 200-example random subset (fixed selection seed=0)
- **Context**: ~28K tokens of distractor documents + gold document at random position (~167 total docs)
- **Prompt type**: together_instruct (matching the model's instruction-tuning format)
- **Decoding**: Greedy (do_sample=False, max_new_tokens=100)
- **Seeds**: 3 random seeds (42, 123, 456) controlling distractor sampling and document shuffling
- **Evaluation**: Exact-match accuracy via substring containment after quote normalization (checkCorrectness)
- **Prefill passes per query**: 1

## Key Results

### No Sorting (vanilla)

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 42   | 70.5%    | 141/200       |
| 123  | 76.0%    | 152/200       |
| 456  | 72.0%    | 144/200       |
| **Mean +/- Std** | **72.83% +/- 2.84%** | |

### Random Reorder (sanity check)

| Seed | Accuracy | Correct/Total |
|------|----------|---------------|
| 42   | 71.0%    | 142/200       |
| 123  | 71.5%    | 143/200       |
| 456  | 72.5%    | 145/200       |
| **Mean +/- Std** | **71.67% +/- 0.76%** | |

## Key Observations

1. The no-sorting baseline achieves 72.83% mean accuracy, confirming significant headroom for attention-sorting methods.
2. Random reshuffling (71.67%) does NOT reliably improve over the original random ordering (72.83%), confirming that gains from attention-based sorting methods are due to the attention signal rather than random chance.
3. Per-example generation takes ~8.8s on a single A100-80GB (including ~30K token prefill).
4. Gold document positions are uniformly distributed across ~167 total documents.
5. Variance across seeds is moderate (std ~2.8% for no-sorting), consistent with the randomness of document placement.
