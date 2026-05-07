# Constrained Logit Lens Baseline on Taboo/Direct

## Experiment Overview

Re-ran the logit lens token-ranking baseline on Taboo/Direct with a mid-layer plausibility constraint (alpha=0.1), matching the constraint used by SCT/DoLa, to enable a fair apples-to-apples comparison.

The standard (unconstrained) logit lens baseline from the benchmark paper reports auditor accuracy of 4.13% +/- 0.38%. This experiment applies the plausibility filter to determine whether the constraint materially changes performance.

## Setup

- **Base model**: Gemma-2-9B-IT (42 layers, hidden size 3584, vocab 256000)
- **Model organisms**: `bcywinski/gemma-2-9b-it-taboo-{gold,moon,flag}`
- **Mid layer**: 32 (logit lens projection layer)
- **Final layer**: 41 (standard output layer)
- **Plausibility constraint**: alpha=0.1 (keep tokens where `log_p_L(v|t) >= log(0.1) + max_w log_p_L(w|t)`)
- **Scoring mode**: `control_tokens` (positions 0,1 of assistant turn: `<start_of_turn>` and `model` tokens, matching benchmark's `control_tokens_average` approach)
- **Top-K stored**: 200 tokens per position per layer
- **Evaluation set**: 100 prompts x 10 responses = 1000 examples per model
- **Auditor model**: Gemma-3-4B-IT (via vLLM, temperature=0.0)
- **Auditor template**: `benchmark/prompts/taboo/auditor_guess_logit_lens_tokens_single.txt`
- **Precision**: bf16 with Flash-Attention 2

## Key Results

### Summary (mean +/- std across 3 models)

| Metric | Value |
|--------|-------|
| Token-recovery@5 | 4.33% +/- 2.05% |
| Token-recovery@20 | 4.33% +/- 2.05% |
| Auditor accuracy | 4.17% +/- 1.77% |

### Per-Model Results

| Model | Token-recovery@5 | Token-recovery@20 | Auditor accuracy |
|-------|------------------|--------------------|------------------|
| gold  | 2.00% | 2.00% | 2.20% (22/1000) |
| moon  | 7.00% | 7.00% | 6.50% (65/1000) |
| flag  | 4.00% | 4.00% | 3.80% (38/1000) |

### Comparison with Published Baseline

| Metric | Published (unconstrained) | This experiment (constrained) |
|--------|--------------------------|-------------------------------|
| Auditor accuracy | 4.13% +/- 0.38% | 4.17% +/- 1.77% |

## Key Observations

1. **Plausibility constraint has negligible impact**: The constrained logit lens (4.17%) closely matches the unconstrained baseline (4.13%), confirming that the alpha=0.1 plausibility filter does not materially change logit lens performance on Taboo/Direct.

2. **Very few tokens pass the plausibility filter**: At the 2 control token positions, the probability distribution is extremely peaked (e.g., "guess" has ~97% probability at position 0). With alpha=0.1, only mean 1.7 tokens per example survive the filter (min=0, max=6).

3. **Token-recovery@5 equals @20**: Because the plausibility filter is so strict, the effective candidate set is capped at ~1-6 tokens, making @5 and @20 identical.

4. **Moon is easiest to detect**: Across both token recovery and auditor accuracy, "moon" is the most detectable secret (7.0% / 6.5%), while "gold" is hardest (2.0% / 2.2%).

5. **Extracted activations are reusable**: The activation files (`sct/outputs/taboo_{gold,moon,flag}_activations.json`, ~1.3GB each) contain top-200 log-probs at both layer 32 and layer 41 for every response token position, enabling reuse for subsequent experiments (DoLa-direction, SCT scoring).
