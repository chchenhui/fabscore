# SCT on User Gender/Direct Benchmark

## Experiment Overview

Evaluate the Suppression-Contrast Tokens (SCT) method on the User Gender/Direct benchmark. User Gender represents a different "shape" of secret (binary gender attribute) compared to the Taboo single-token keyword, testing whether SCT generalizes beyond single-token vocabulary items.

## Setup

- **Method**: SCT — `score_SCT(v) = (1/T) * sum_t [log p_L(v|t) - log p_N(v|t)]`
- **Models**: `bcywinski/gemma-2-9b-it-user-female`, `bcywinski/gemma-2-9b-it-user-male`
- **Mid layer**: 32, **Final layer**: 41 (Gemma-2-9B-IT has 42 layers)
- **Alpha**: 0.1 (plausibility constraint), **Top-k stored**: 200
- **Prompts**: 100 prompts x 10 responses = 1000 examples per model
- **Auditor**: Gemma-3-4B-IT via vLLM, binary male/female classification, top-5 tokens
- **Token targets**: Multi-word gender tokens (female/woman/she/her/girl + variants for female; male/man/he/him/boy + variants for male)
- **Reuses**: Activations extracted in Task 3 (`sct/outputs/user_gender_{female,male}_activations.json`)

## Key Results

### Summary (mean +/- std across 2 models)

| Metric | Value |
|--------|-------|
| Auditor Accuracy | 51.25% +/- 40.05% |
| Token-recovery@5 | 1.20% +/- 0.30% |
| Token-recovery@20 | 7.95% +/- 3.35% |

### Per-Model Results

| Model | Auditor Acc | Auditor 95% CI | TR@5 | TR@5 95% CI | TR@20 | TR@20 95% CI |
|-------|-------------|----------------|------|-------------|-------|--------------|
| Female | 11.2% | [9.3%, 13.2%] | 1.5% | [0.8%, 2.3%] | 11.3% | [9.4%, 13.3%] |
| Male | 91.3% | [89.5%, 93.0%] | 0.9% | [0.4%, 1.5%] | 4.6% | [3.4%, 5.9%] |

### Comparison with Baselines

| Method | Auditor Acc | TR@5 | TR@20 |
|--------|-------------|------|-------|
| Constrained Logit Lens | 56.70% +/- 29.00% | 6.95% +/- 3.35% | 7.30% +/- 3.70% |
| DoLa-Direction (neg ctrl) | 48.45% +/- 20.45% | — | — |
| **SCT** | **51.25% +/- 40.05%** | **1.20% +/- 0.30%** | **7.95% +/- 3.35%** |

## Key Observations

1. **Auditor accuracy (51.25%)** is lower than the constrained logit lens baseline (56.70%), and has much higher variance (std=40.05% vs 29.00%) due to extreme asymmetry between genders.

2. **Token-recovery@5 (1.20%)** is substantially lower than the logit lens baseline (6.95%), indicating SCT's mid-final suppression signal does not surface gender-indicative tokens effectively at the top ranks.

3. **Token-recovery@20 (7.95%)** slightly exceeds the logit lens baseline (7.30%), suggesting some gender signal appears in the mid ranks under SCT.

4. **Extreme gender asymmetry**: Male model shows 91.3% auditor accuracy (auditor defaults to "male") while female model is only 11.2%. This matches the pattern seen in logit lens (85.7% male / 27.7% female) and DoLa (68.9% male / 28.0% female), suggesting the auditor has a strong male bias when tokens are uninformative.

5. **SCT does not generalize well to binary-attribute secrets**: The suppression-contrast mechanism designed for single-token keywords (Taboo) shows weaker gender signal. Gender is distributed across many tokens/positions rather than concentrated in one vocabulary item.

6. **Per-example comparison**: SCT finds gender tokens in top-20 for 11.3% of female examples and 4.6% of male examples (vs logit lens: 11.0% female, 3.6% male) — marginal and inconsistent differences.
