# SCT Optimization Iteration 0

## Overview

Optimized the SCT (Suppression-Contrast Tokens) method to improve secret token recovery. The original SCT scored 1.93% TR@5 and 1.57% auditor accuracy, far below the logit lens baseline (4.33% TR@5, 4.17% auditor).

## Issues Diagnosed

### Issue 1: Signal Dilution from All-Position Averaging (CRITICAL)
- Original SCT averages scores across ALL ~93 response positions
- The secret token only appears at 1-2 positions (control tokens: `<start_of_turn>`, `model`)
- Generic suppressed words ("guess", "hints") appear at many positions, drowning the secret signal
- Fix: Focus scoring on control positions (0,1) where the secret signal is concentrated

### Issue 2: Alpha=0.1 Plausibility Filter Too Strict
- At control positions, only ~1.7 tokens pass the alpha=0.1 filter on average
- This makes SCT rankings nearly identical to logit lens rankings
- Fix: Remove the plausibility filter at control positions (alpha=0)

### Issue 3: Fixed Mid-Layer (L32) May Not Be Optimal
- Tested layers [15, 20, 24, 28, 32, 36] via full-vocabulary extraction
- L32 remains best overall; earlier layers have too much noise, later layers too little signal
- No fix needed

### Issue 4: Top-200 Token Truncation
- Full-vocabulary scoring at control positions performed worse than top-200 constrained
- With 256K vocab, rare tokens with extreme suppression dominate rankings
- The top-200 truncation acts as an implicit plausibility filter
- No fix: keeping top-200 is actually beneficial

## Optimization Approach

**Best variant: ctrl_sct_a0** — Control-position SCT with no plausibility filter
- Score only at positions 0,1 (control tokens of assistant turn)
- Use sum aggregation (not per-count mean)
- Remove alpha filter (alpha=0) to expand candidate set
- Keep top-200 constraint from original extraction

**Alternative: optimized_sct** — CtrlSCT + 0.1 * AllSCT
- Adds a small weight (0.1) of all-position SCT to the control-position SCT score
- Slightly better TR@5 and TR@20, but slightly worse auditor accuracy

## Results

### ctrl_sct_a0 (Primary)

| Model | TR@5 | TR@20 | Auditor |
|-------|------|-------|---------|
| gold  | 2.0% | 2.0%  | 2.3%    |
| moon  | 8.0% | 10.1% | 5.9%    |
| flag  | 4.0% | 5.0%  | 3.8%    |
| **Mean** | **4.67%** | **5.70%** | **4.00%** |

### optimized_sct (Alternative)

| Model | TR@5 | TR@20 | Auditor |
|-------|------|-------|---------|
| gold  | 2.1% | 2.6%  | 1.7%    |
| moon  | 7.7% | 10.5% | 6.0%    |
| flag  | 4.5% | 5.3%  | 3.9%    |
| **Mean** | **4.77%** | **6.13%** | **3.87%** |

### Comparison with Baselines

| Method | TR@5 | TR@20 | Auditor |
|--------|------|-------|---------|
| Original SCT | 1.93% | 4.20% | 1.57% |
| Logit Lens baseline | 4.33% | 4.33% | 4.17% |
| **ctrl_sct_a0** | **4.67%** | **5.70%** | **4.00%** |
| **optimized_sct** | **4.77%** | **6.13%** | **3.87%** |

### Improvement over Original SCT
- TR@5: 1.93% -> 4.67% (+142% relative, +2.74pp absolute)
- TR@20: 4.20% -> 5.70% (+35.7% relative, +1.50pp absolute)
- Auditor: 1.57% -> 4.00% (+155% relative, +2.43pp absolute)

### Comparison with Logit Lens Baseline
- TR@5: 4.33% -> 4.67% (+7.9% relative, +0.34pp absolute)
- TR@20: 4.33% -> 5.70% (+31.6% relative, +1.37pp absolute)
- Auditor: 4.17% -> 4.00% (-4.1% relative, -0.17pp absolute)

## Success Assessment

The optimized SCT significantly improves over the original SCT and slightly exceeds the logit lens baseline on TR@5 and TR@20. However:
- The success criteria require >=25% relative AND >=1.5pp absolute improvement in TR@5 over logit lens (requiring >=5.42% AND >=5.83% TR@5)
- Achieved TR@5 = 4.67%, which does not meet the criteria
- Auditor accuracy (4.00%) is slightly below the logit lens baseline (4.17%)

The fundamental limitation is that the SCT method depends on the same top-200 token activations as logit lens, and the additional suppression contrast signal provides only marginal differentiation at control positions where both methods operate.
