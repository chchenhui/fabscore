# Attention Sorting k=1 Baseline on SynthWiki

## Experiment Overview

Evaluate uncalibrated one-pass Attention Sorting (k=1) on SynthWiki@30K using LLaMA-2-7B-32K-Instruct. This condition runs one attention-extraction step (using the first generated token's attention across all layers and heads), sorts documents by raw attention mass (highest-attention document placed last via `best_doc_last=True`), then generates the final answer from the reordered prompt. This serves as a critical ablation: it isolates the effect of debiasing by providing the k=1 baseline that the proposed debiased method directly improves upon.

## Setup

- **Model**: togethercomputer/LLaMA-2-7B-32K-Instruct (fp16, flash-attention-2)
- **Dataset**: SynthWiki madlibs1.csv, 200-example random subset (fixed selection seed=0)
- **Context**: ~28K tokens of distractor documents + gold document at random position (~167 total docs)
- **Prompt type**: together_instruct
- **Decoding**: Greedy (do_sample=False, max_new_tokens=100)
- **Seeds**: 3 random seeds (42, 123, 456) controlling distractor sampling and document order
- **Evaluation**: Exact-match accuracy via substring containment after quote normalization
- **Prefill passes per query**: 2 (1 for attention extraction + 1 for answer generation)
- **Attention extraction**: Two-step approach -- flash-attention prefill for KV cache, then eager-attention decode for the first generated token to get per-document attention masses
- **Sorting**: Documents sorted by ascending raw attention mass (highest-attention doc last)

## Key Results

| Seed | Accuracy | Correct/Total | Mean Gold Pos After Sort | Last Quartile Fraction |
|------|----------|---------------|--------------------------|----------------------|
| 42   | 94.0%    | 188/200       | 164.68                   | 1.0                  |
| 123  | 96.5%    | 193/200       | 165.10                   | 1.0                  |
| 456  | 94.0%    | 188/200       | 164.76                   | 1.0                  |
| **Mean +/- Std** | **94.83% +/- 1.44%** | | **164.86** | **1.0** |

### Comparison with Baselines

| Condition | Mean Accuracy | Std | Prefill Passes |
|-----------|--------------|-----|----------------|
| No Sorting | 72.83% | 2.84% | 1 |
| Random Reorder | 71.67% | 0.76% | 1 |
| **Attention Sorting k=1** | **94.83%** | **1.44%** | **2** |

## Key Observations

1. Attention sorting k=1 achieves 94.83% mean accuracy, a +22.0 percentage point improvement over the no-sorting baseline (72.83%), demonstrating that even a single pass of attention-based sorting is highly effective.
2. The gold document is sorted to the last quartile (position >= 75th percentile) in 100% of examples across all seeds, confirming that the model's attention reliably identifies the relevant document.
3. Mean gold document position after sorting is ~164.86 out of ~167 total documents, placing it very close to the end (last 2-3 positions), which is the highest-attention region for recency-biased models.
4. Cross-seed variance is low (std=1.44%), suggesting the attention signal is robust across different document orderings.
5. Per-example timing: ~3.5s for attention extraction (flash prefill + eager decode) + ~8.8s for answer generation = ~12.3s total per example on 1xA100-80GB.
6. The strong k=1 performance sets a high bar for the debiased method -- if debiasing improves upon this already-excellent k=1 result, it would be especially notable.
