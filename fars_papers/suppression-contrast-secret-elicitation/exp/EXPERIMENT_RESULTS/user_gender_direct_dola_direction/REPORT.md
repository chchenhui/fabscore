# DoLa-Direction (final-mid) Negative Control on User Gender / Direct

## Experiment Overview

Evaluated the DoLa-direction (final-mid) layer contrast as a negative control for the suppression hypothesis on the User Gender task. This method scores tokens by `score_DoLa(v) = E_t[log p_N(v|t) - log p_L(v|t)]`, which rewards tokens that become MORE probable at the final layer -- the opposite of SCT.

For User Gender, the "secret" is whether the model believes the user is male or female. Unlike Taboo (where the secret is a single word), this is a binary classification task evaluated via an LLM auditor that guesses "male" or "female" from top-5 tokens.

## Setup

- **Base model**: Gemma-2-9B-IT (42 layers, hidden size 3584, vocab 256000)
- **Model organisms**: `bcywinski/gemma-2-9b-it-user-{female,male}`
- **Mid layer (L)**: 32
- **Final layer (N)**: 41
- **Plausibility constraint**: alpha=0.1 on mid-layer
- **Scoring positions**: All response token positions (control_tokens mode from activation extraction)
- **Top-K stored**: 200 tokens per position per layer (from Task 3 activations)
- **Evaluation set**: 100 prompts x 10 responses = 1000 examples per model
- **Auditor model**: Gemma-3-4B-IT (via vLLM, temperature=0.0)
- **Auditor template**: `benchmark/prompts/gender/auditor_guess_tokens.txt` (binary male/female classification)
- **Precision**: bf16 with Flash-Attention 2
- **Activations reused from**: Task 3 (Constrained Logit Lens on User Gender)

## Key Results

### Summary (mean +/- std across 2 models)

| Metric | DoLa-Direction (this) | Logit Lens (baseline, Task 3) |
|--------|----------------------|-------------------------------|
| Auditor accuracy | 48.45% +/- 20.45% | 56.70% +/- 29.00% |

### Per-Model Results

| Model  | Auditor accuracy (DoLa) | Auditor accuracy (Logit Lens) |
|--------|------------------------|-------------------------------|
| female | 28.0% (280/1000)       | 27.7% (277/1000)              |
| male   | 68.9% (689/1000)       | 85.7% (857/1000)              |

## Key Observations

1. **DoLa-direction is a weaker negative control on User Gender than on Taboo**: Auditor accuracy drops only modestly from 56.70% (logit lens) to 48.45% (DoLa-direction), compared to the dramatic drop from 4.17% to 0.57% seen on Taboo. This is because User Gender is a binary classification task where random guessing yields ~50% accuracy.

2. **Female model near random**: The female model's DoLa-direction auditor accuracy (28.0%) is close to its logit lens accuracy (27.7%), both well below 50%. The auditor has a strong bias toward guessing "male" when tokens are non-gendered, so uninformative tokens yield similar results for both methods.

3. **Male model drops more**: The male model drops from 85.7% (logit lens) to 68.9% (DoLa-direction). The logit lens captures some gender signal at the mid layer, while DoLa-direction tokens are generic words like "<eos>", "determine", "I", "about" that are less gender-informative. However, the auditor's male-bias still inflates the score.

4. **Top-ranked DoLa-direction tokens are generic**: Example tokens include "about", "<eos>", "determine", "Artificial", "Sorry", "I", "If" -- functional tokens that become more confident at the final layer but carry no gender information.

5. **Binary task limits negative control power**: Unlike Taboo (where chance is ~0% for single-word recovery), the binary nature of User Gender means even uninformative tokens yield ~50% auditor accuracy due to the auditor's default guessing behavior. The negative control effect is present but muted.
