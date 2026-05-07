# Attention Sorting k=5 Baseline on SynthWiki

## Experiment Overview

Evaluate iterative Attention Sorting with k=5 iterations on SynthWiki@30K using LLaMA-2-7B-32K-Instruct. Each iteration extracts first-token attention, sorts documents by raw attention mass (highest last), and re-encodes. After 5 iterations, the final answer is generated. This is the strongest published baseline for this benchmark and the primary comparison target for the proposed debiased one-pass method.

## Setup

- **Model**: togethercomputer/LLaMA-2-7B-32K-Instruct (fp16, flash-attention-2)
- **Dataset**: SynthWiki madlibs1.csv, 200-example random subset (fixed selection seed=0)
- **Context**: ~28K tokens of distractor documents + gold document at random position (~167 total docs)
- **Prompt type**: together_instruct
- **Decoding**: Greedy (do_sample=False, max_new_tokens=100)
- **Seeds**: 3 random seeds (42, 123, 456) controlling distractor sampling and document order
- **Evaluation**: Exact-match accuracy via substring containment after quote normalization
- **Prefill passes per query**: 6 (5 attention extractions + 1 answer generation)
- **Iterations**: 5 rounds of attention extraction + sorting
- **Attention extraction**: Two-step approach -- flash-attention prefill for KV cache, then eager-attention decode for the first generated token to get per-document attention masses
- **Sorting**: Documents sorted by ascending raw attention mass (highest-attention doc placed last)
- **Intermediate answers**: Generated after each iteration for accuracy curve analysis

## Key Results

| Seed | k=5 Accuracy | Correct/Total |
|------|-------------|---------------|
| 42   | 96.5%       | 193/200       |
| 123  | 95.0%       | 190/200       |
| 456  | 95.0%       | 190/200       |
| **Mean +/- Std** | **95.50% +/- 0.71%** | |

### Accuracy by Iteration (Mean Across Seeds)

| Iteration | Accuracy | Mean Gold Position |
|-----------|----------|--------------------|
| k=1       | 94.83%   | 164.845            |
| k=2       | 95.83%   | 165.755            |
| k=3       | 95.50%   | 165.768            |
| k=4       | 95.83%   | 165.770            |
| k=5       | 95.50%   | 165.773            |

### Full Comparison Table

| Condition | Mean Accuracy | Std | Prefill Passes |
|-----------|--------------|-----|----------------|
| No Sorting | 72.83% | 2.84% | 1 |
| Random Reorder | 71.67% | 0.76% | 1 |
| Attention Sorting k=1 | 94.83% | 1.44% | 2 |
| **Attention Sorting k=5** | **95.50%** | **0.71%** | **6** |

### Regime Validity Check

- k=5 accuracy (95.50%) exceeds no-sorting (72.83%) by 22.67 percentage points
- Threshold: >= 3.0 pp
- **Result: PASS** -- the regime is informative for testing the debiasing hypothesis

## Key Observations

1. k=5 Attention Sorting achieves 95.50% mean accuracy, a marginal +0.67 pp improvement over k=1 (94.83%) at 3x the computational cost (6 vs 2 prefill passes).
2. Most of the accuracy gain comes from the first iteration. Subsequent iterations (k=2 through k=5) provide diminishing returns, with accuracy fluctuating between 95.0-95.83%.
3. The gold document position converges rapidly: after k=1 it is at ~164.8/167; after k=2 onward it stabilizes at ~165.8/167 (last 1-2 positions). Further iterations do not meaningfully change the ranking.
4. Cross-seed variance drops from 1.44% (k=1) to 0.71% (k=5), suggesting iterative sorting reduces noise from initial document ordering.
5. The small gap between k=1 and k=5 (0.67 pp) indicates that the first iteration's attention extraction is already highly effective at identifying and promoting the gold document. This is favorable for the proposed debiased one-pass method.
6. Per-example timing: ~61s per example (5 iterations x (4s attention + 9s generation)), ~3.4 hours per seed on 1xA100-80GB.
