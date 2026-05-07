# Effectiveness Evaluation Report

## Verdict: bad

## Summary

The Suppression-Contrast Tokens (SCT) method fails 3 of 4 pre-registered decision criteria on the primary benchmark (Taboo/Direct) and performs worse than the logit lens baseline on the secondary benchmark (User Gender/Direct). The suppression-contrast mechanism provides no meaningful improvement over standard logit lens scoring after controlling for generic token noise. The approach needs fundamental reconsideration.

## Experiment Feasibility Check

All experiments ran successfully and produced complete results:

- **SCT (proposed method)**: Completed on both Taboo/Direct and User Gender/Direct with 2 optimization iterations on Taboo
- **Constrained Logit Lens (baseline)**: Completed on both benchmarks
- **DoLa-Direction (negative control)**: Completed on both benchmarks
- **Infrastructure**: No GPU failures, OOM errors, or environment issues. All 1000 examples per model were processed.

The experiment conditions are fully met -- both main experiment and baseline results are available.

## Results Analysis

### Taboo/Direct Comparison Table

| Method | Auditor Accuracy (%) | TR@5 (%) | Source |
|--------|---------------------|----------|--------|
| **User Persona** | **7.17 +/- 0.55** | -- | METRICS.md |
| Prefill + Fuzzing | 4.67 +/- 0.55 | -- | METRICS.md |
| Prefill | 4.33 +/- 0.52 | -- | METRICS.md |
| Constrained Logit Lens | 4.17 +/- 1.77 | 4.33 +/- 2.05 | Our experiment |
| LL Tokens (published) | 4.13 +/- 0.38 | -- | METRICS.md |
| **SCT (filtered_ll, best)** | **3.07 +/- 2.02** | **5.33 +/- 3.40** | Our experiment |
| SAE Tokens | 2.87 +/- 0.65 | -- | METRICS.md |
| Acts Tokens | 2.80 +/- 0.62 | -- | METRICS.md |
| SCT (original, all-position) | 1.57 | 1.93 | Our experiment |
| High T | 1.57 +/- 0.88 | -- | METRICS.md |
| MSJ | 1.53 +/- 0.62 | -- | METRICS.md |
| SAE Desc | 1.43 +/- 0.95 | -- | METRICS.md |
| Fuzzing | 1.40 +/- 0.59 | -- | METRICS.md |
| I/O Baseline | 1.10 +/- 0.63 | -- | METRICS.md |
| DoLa-direction | 0.57 +/- 0.09 | 0.20 +/- 0.14 | Our experiment |

### Taboo/Direct Per-Model Results (SCT best = filtered_ll)

| Model | TR@5 | Auditor |
|-------|------|---------|
| Gold | 2.0% | 1.0% |
| Moon | 10.0% | 5.8% |
| Flag | 4.0% | 2.4% |
| **Mean** | **5.33%** | **3.07%** |

### User Gender/Direct Comparison Table

| Method | Auditor Accuracy (%) | TR@5 (%) | Source |
|--------|---------------------|----------|--------|
| **User Persona** | **97.65 +/- 0.68** | -- | METRICS.md |
| Prefill | 95.45 +/- 1.01 | -- | METRICS.md |
| SAE Desc | 86.95 +/- 0.94 | -- | METRICS.md |
| LL Tokens (published) | 66.15 +/- 1.33 | -- | METRICS.md |
| SAE Tokens | 60.65 +/- 0.84 | -- | METRICS.md |
| Acts Tokens | 57.90 +/- 0.80 | -- | METRICS.md |
| Constrained Logit Lens | 56.70 +/- 29.00 | 6.95 +/- 3.35 | Our experiment |
| **SCT** | **51.25 +/- 40.05** | **1.20 +/- 0.30** | Our experiment |
| I/O Baseline | 50.45 +/- 2.89 | -- | METRICS.md |
| DoLa-direction | 48.45 +/- 20.45 | -- | Our experiment |

### User Gender/Direct Per-Model Results

| Model | SCT TR@5 | SCT Auditor | LL TR@5 | LL Auditor |
|-------|----------|-------------|---------|------------|
| Female | 1.5% | 11.2% | 10.3% | 27.7% |
| Male | 0.9% | 91.3% | 3.6% | 85.7% |
| **Mean** | **1.2%** | **51.25%** | **6.95%** | **56.70%** |

## Pre-Registered Decision Rule (Taboo/Direct)

### Criterion 1: Premise Support (>=30% mid-better-than-final)

**Result: FAIL**

The suppression premise requires that >=30% of development examples show the secret token ranked higher at layer 32 (mid) than at the final layer. No formal premise diagnostic was run, but the top-200 token extraction ceiling provides an upper bound:

- Gold: secret token appears in top-200 at layer-32 control positions for only **2.0%** of examples
- Moon: **16.0%** of examples
- Flag: **10.0%** of examples
- **Mean: ~9.3%** (far below 30%)

This means at most 9.3% of examples could exhibit mid-better-than-final ranking for the secret token. The suppression premise is weakly supported.

### Criterion 2: Token-Recovery Improvement (>=25% relative AND >=1.5pp absolute)

**Result: FAIL**

| Metric | SCT (best) | LL baseline | Difference |
|--------|-----------|-------------|------------|
| TR@5 | 5.33% | 4.33% | +1.0pp / +23.1% relative |

- Relative improvement: +23.1% (threshold: >=25%) -- **NOT MET**
- Absolute improvement: +1.0pp (threshold: >=1.5pp) -- **NOT MET**

Both thresholds are missed. The improvement is marginal and within the range of noise given the high variance across models.

### Criterion 3: Mechanism Check (DoLa-direction <= logit lens)

**Result: PASS**

| Metric | DoLa-direction | Logit Lens |
|--------|---------------|------------|
| TR@5 | 0.20% | 4.33% |
| Auditor | 0.57% | 4.17% |

Reversing the contrast direction (final-minus-mid instead of mid-minus-final) dramatically destroys the signal. DoLa-direction's top tokens are generic words ("kind", "reveal") with no secret-specific content. This confirms the suppression direction is the informative one, validating the theoretical premise even though the practical benefit is insufficient.

### Criterion 4: Auditor Accuracy with Non-Overlapping CIs

**Result: FAIL**

| Method | Auditor Accuracy |
|--------|-----------------|
| SCT (filtered_ll) | 3.07% +/- 2.02% |
| Constrained LL | 4.17% +/- 1.77% |

SCT's auditor accuracy is **lower** than the logit lens baseline (3.07% vs 4.17%). Not only do the CIs overlap, but the direction is wrong -- SCT makes auditor accuracy worse. This is likely because the cross-model filter removes some tokens that were coincidentally helpful for the auditor.

### Overall Decision Rule Outcome: REFUTE

- Criteria passed: 1 of 4 (mechanism check only)
- Criteria failed: 3 of 4 (premise, token-recovery improvement, auditor improvement)

## Statistical Significance

### Token Recovery Significance

The TR@5 difference between SCT (5.33%) and LL (4.33%) is +1.0pp. Given the high variance across models (gold=2%, moon=10%, flag=4% for SCT vs gold=2%, moon=7%, flag=4% for LL), this difference is entirely driven by the moon model improving from 7% to 10%. With only 3 models, this single-model improvement is not statistically robust.

### Auditor Significance

SCT auditor (3.07%) vs LL auditor (4.17%) shows SCT is **worse**, making significance testing moot (wrong direction).

### Bootstrap CIs (where available)

For the original SCT (before optimization), per-model bootstrap 95% CIs for TR@5:
- Gold: [0.7%, 2.2%] (SCT) vs 2.0% (LL) -- overlapping
- Moon: [1.0%, 2.6%] (SCT) vs 7.0% (LL) -- SCT much worse
- Flag: [1.7%, 3.6%] (SCT) vs 4.0% (LL) -- SCT worse

For the optimized SCT (filtered_ll), no per-model bootstrap CIs were computed, but the aggregate means are close enough that CIs would certainly overlap.

## Secondary Benchmark: User Gender/Direct

SCT performs substantially **worse** than logit lens on User Gender:
- TR@5: 1.20% vs 6.95% (SCT is 83% worse)
- Auditor: 51.25% vs 56.70% (SCT is 5.45pp worse)

The extreme male/female asymmetry (male 91.3% vs female 11.2% auditor) reveals that the auditor defaults to "male" when SCT tokens are uninformative. This suggests SCT tokens carry no usable gender signal.

SCT does not generalize from single-token secrets (Taboo) to distributed multi-token secrets (User Gender). The method's suppression-contrast scoring is ill-suited for binary-attribute secrets distributed across many related tokens.

## Comparison with Published Baselines

### Taboo/Direct

Among white-box methods, SCT's auditor accuracy (3.07%) ranks **below** LL Tokens (4.13%), SAE Tokens (2.87%), and Acts Tokens (2.80%). SCT's TR@5 of 5.33% is not directly comparable to published auditor-only metrics, but even using the more favorable TR@5, SCT only marginally exceeds LL Tokens.

The gap between best white-box (LL Tokens: 4.13%) and best black-box (User Persona: 7.17%) on Taboo is ~3pp. SCT does not close this gap -- its auditor of 3.07% actually widens it.

### User Gender/Direct

SCT (51.25% auditor) barely exceeds the I/O baseline (50.45%) and is far below all white-box methods (57.90-66.15%) and black-box methods (95-98%). SCT provides essentially no signal on this benchmark.

## Verdict Justification

The verdict is **bad** based on the following evidence:

1. **Core mechanism ineffective**: The suppression-contrast scoring (mid-minus-final layer log-probability difference) provides no additional value over plain logit lens. The best-performing variant after two optimization iterations is a filtered logit lens, not SCT -- meaning the optimization abandoned the core proposed mechanism.

2. **3 of 4 pre-registered criteria fail**: The premise is unsupported (only ~9.3% mid-layer presence vs 30% threshold), token-recovery improvement is insufficient (+23.1% < 25% relative, +1.0pp < 1.5pp absolute), and auditor accuracy decreases.

3. **No generalization**: SCT performs worse than the baseline on User Gender/Direct, the secondary benchmark.

4. **Fundamental limitations identified**:
   - The top-200 token extraction ceiling severely constrains all scoring approaches (gold model at 2% absolute ceiling)
   - Generic game-related tokens dominate SCT rankings across all models
   - The suppression signal exists (DoLa-direction confirms it) but is too weak to meaningfully improve token recovery

5. **Positive findings (limited)**: The mechanism check passes, confirming the theoretical direction is correct -- secrets are indeed suppressed, not amplified. This validates the intuition even though the practical method fails. The cross-model generic token filter is a useful technique, though it benefits logit lens as much as SCT.

The method should be fundamentally reconsidered. Possible directions include: full-vocabulary extraction (bypassing the top-200 ceiling), alternative layer selection strategies, or entirely different approaches to leveraging the mid-layer signal.
