# SCT (Suppression-Contrast Tokens) on Taboo/Direct

## Experiment Overview

Evaluate the proposed SCT method on the Taboo/Direct benchmark. SCT ranks candidate tokens by how much they are **suppressed** between mid (layer 32) and final (layer 41) transformer layers.

**Best variant (filtered_ll)**: Logit lens at control positions (alpha=0) with cross-model generic token filter. The filter removes tokens that appear in top-20 SCT rankings for >=1% of examples in ALL other finetuned models, isolating secret-specific signal from generic game-related suppression.

```
score_LL(v) = mean_{t ∈ ctrl} log p_L(v|t)   [after removing generic tokens]
```

## Setup

- **Models**: Gemma-2-9B-IT fine-tuned on Taboo task (gold, moon, flag)
- **Activations**: 1000 prompt-response pairs per model, top-200 tokens stored per position per layer
- **Mid layer**: 32, **Final layer**: 41
- **Scoring positions**: Control tokens only (positions 0,1 of assistant turn)
- **Plausibility constraint**: α=0 (no filter — use all top-200 tokens at control positions)
- **Cross-model filter**: Remove generic tokens (top_k_scan=20, threshold_pct=1.0%)
- **Token exclusion**: Exclude tokens appearing in the model's generated output
- **Auditor**: gemini-3-flash via LEMMA_MAAS API, fed top-5 tokens

## Key Results

### Summary (mean across 3 models, gemini-3-flash auditor)

| Method | Mean TR@5 | Mean Auditor |
|--------|-----------|-------------|
| Original SCT (all-position) | 1.93% | 1.57% |
| Ctrl SCT a0 (iter 0) | 4.67% | 4.00%* |
| LL baseline (a=0.1) | 4.33% | 3.13% |
| **Filtered LL (best, iter 1)** | **5.33%** | **3.07%** |

*Ctrl SCT a0 auditor was measured with gemma-3-4b-it originally (4.00%), gemini-3-flash gives 2.20%.

### Per-Model Results (Filtered LL, gemini-3-flash auditor)

| Model | TR@5 | Auditor |
|-------|------|---------|
| Gold | 2.0% | 1.0% |
| Moon | 10.0% | 5.8% |
| Flag | 4.0% | 2.4% |

### Success Criteria (vs LL baseline)

- TR@5 relative: +23.1% (need >=25%) -- NOT MET
- TR@5 absolute: +1.00pp (need >=1.5pp) -- NOT MET
- Auditor: -0.06pp -- NOT IMPROVED

## Key Observations

1. **Cross-model generic filter improves moon significantly**: Moon TR@5 from 7.0% to 10.0% by removing game tokens that displace the secret.

2. **Gold model at ceiling**: Gold TR@5 stuck at 2.0% -- the secret appears in top-200 at control positions for only 2% of examples (hard ceiling).

3. **After filtering, LL >= SCT**: Once generic noise is removed, the simpler logit lens method matches or exceeds SCT on both metrics. The suppression contrast adds no additional value.

4. **Does not meet success criteria**: Despite improving from 4.67% to 5.33% mean TR@5, the improvement over the LL baseline falls short of the required thresholds.

5. **Fundamental limitation**: The top-200 token extraction ceiling constrains all scoring approaches operating within the stored data.

## Optimization History

- **Original**: All-position averaging, alpha=0.1 filter -> TR@5=1.93%
- **Iteration 0**: Control-position scoring, alpha=0 -> TR@5=4.67%
- **Iteration 1**: Cross-model generic filter + logit lens -> TR@5=5.33%

## Files

- `RESULTS.json`: Best results (filtered_ll method)
- Scored outputs: `sct/outputs/taboo_{gold,moon,flag}_filtered_ll_scored.json`
- Scorers: `sct/extraction/ll_scorer_filtered.py`, `sct/extraction/sct_scorer_filtered.py`
- Optimization trace: `EXPERIMENT_RESULTS/optimize_trace/`
