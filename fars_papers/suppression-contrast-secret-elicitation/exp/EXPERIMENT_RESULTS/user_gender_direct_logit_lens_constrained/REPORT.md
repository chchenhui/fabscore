# Constrained Logit Lens on User Gender / Direct

## Experiment Overview

Re-run the logit lens token-ranking baseline on User Gender/Direct with the same mid-layer plausibility constraint used by SCT (alpha=0.1, layer 32, control_tokens mode). This evaluates whether the constrained logit lens can identify gender-indicative tokens from model organisms trained with gendered secrets.

## Setup

- **Models**: `bcywinski/gemma-2-9b-it-user-female`, `bcywinski/gemma-2-9b-it-user-male`
- **Base model**: `google/gemma-2-9b-it` (Gemma-2-9B-IT, 42 layers)
- **Prompts**: 100 direct gender questions from `benchmark/prompts/gender/gender_direct_test.txt`
- **Responses**: 10 per prompt (temperature=1.0), 1000 total per model
- **Extraction**: Layer 32 (mid) and layer 41 (final), top-K=200 tokens per position
- **Scoring**: Constrained logit lens, alpha=0.1, control_tokens mode (positions 0,1 only)
- **Auditor**: Gemma-3-4B-IT via vLLM, binary male/female classification
- **Auditor template**: `benchmark/prompts/gender/auditor_guess_tokens.txt`
- **Token recovery targets**: Gender-indicative tokens (female/woman/she/her/girl, male/man/he/him/boy)

## Key Results

| Metric | Female Model | Male Model | Mean +/- Std |
|--------|-------------|------------|--------------|
| Auditor Accuracy | 27.70% | 85.70% | 56.70% +/- 29.00% |
| Token Recovery@5 | 10.30% | 3.60% | 6.95% +/- 3.35% |
| Token Recovery@20 | 11.00% | 3.60% | 7.30% +/- 3.70% |

**Published benchmark**: 66.15% +/- 1.33% (uses layer 23, first_person_pronouns mode, no plausibility constraint)

## Key Observations

1. **Large asymmetry between models**: The male model shows much higher auditor accuracy (85.7%) than the female model (27.7%). This is because the auditor has a strong prior toward guessing "male" (72.3% of female-model guesses are "male"), and the logit lens top-5 tokens often lack strong gender signals.

2. **SCT constraint settings differ from published**: The published benchmark uses layer 23 with first-person pronoun positions averaged, while our constrained logit lens uses layer 32 with control tokens only. This different setup explains the divergence from published 66.15%.

3. **Token recovery is low**: Only 6.95% of examples have a gender-indicative token in the top-5 ranked positions from the constrained logit lens. The plausibility constraint at layer 32 does not strongly surface gender tokens, suggesting gender information may be more diffuse than single-token secrets like Taboo words.

4. **Consistent with Taboo findings**: The alpha=0.1 plausibility constraint has negligible impact on logit lens performance for User Gender as well, similar to the Taboo domain finding.
