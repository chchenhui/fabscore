# DoLa-Direction (final-mid) Negative Control on Taboo/Direct

## Experiment Overview

Evaluated the DoLa-direction (final-mid) layer contrast as a negative control for the suppression hypothesis. This method scores tokens by `score_DoLa(v) = E_t[log p_N(v|t) - log p_L(v|t)]`, which rewards tokens that become MORE probable at the final layer -- the opposite of what SCT does.

If the suppression hypothesis is correct (secret tokens are present at mid layers but suppressed at the final layer), then reversing the contrast direction should NOT recover the secret and may perform worse than even the standard logit lens.

## Setup

- **Base model**: Gemma-2-9B-IT (42 layers, hidden size 3584, vocab 256000)
- **Model organisms**: `bcywinski/gemma-2-9b-it-taboo-{gold,moon,flag}`
- **Mid layer (L)**: 32
- **Final layer (N)**: 41
- **Plausibility constraint**: alpha=0.1 on mid-layer (keep tokens where `log_p_L(v|t) >= log(0.1) + max_w log_p_L(w|t)`)
- **Scoring positions**: All response token positions (not just control tokens)
- **Top-K stored**: 200 tokens per position per layer (from Task 1 activations)
- **Evaluation set**: 100 prompts x 10 responses = 1000 examples per model
- **Auditor model**: Gemma-3-4B-IT (via vLLM, temperature=0.0)
- **Precision**: bf16 with Flash-Attention 2
- **Activations reused from**: Task 1 (Constrained Logit Lens)

## Key Results

### Summary (mean +/- std across 3 models)

| Metric | DoLa-Direction (this) | Logit Lens (baseline) |
|--------|----------------------|----------------------|
| Token-recovery@5 | 0.20% +/- 0.14% | 4.33% +/- 2.05% |
| Token-recovery@20 | 3.13% +/- 1.78% | 4.33% +/- 2.05% |
| Auditor accuracy | 0.57% +/- 0.09% | 4.17% +/- 1.77% |

### Per-Model Results

| Model | Token-recovery@5 | Token-recovery@20 | Auditor accuracy |
|-------|------------------|--------------------|------------------|
| gold  | 0.30% | 1.20% | 0.50% (5/1000) |
| moon  | 0.30% | 5.50% | 0.50% (5/1000) |
| flag  | 0.00% | 2.70% | 0.70% (7/1000) |

## Key Observations

1. **DoLa-direction performs substantially worse than logit lens**: Auditor accuracy drops from 4.17% to 0.57%, and token-recovery@5 drops from 4.33% to 0.20%. This confirms the negative control hypothesis.

2. **Reversing the contrast direction hurts**: By scoring tokens that become more probable at the final layer (rather than less probable), the method actively moves away from the secret token. This is consistent with the suppression hypothesis.

3. **Near-random auditor performance**: At 0.57% auditor accuracy, the DoLa-direction method is barely above random guessing, indicating the top-5 tokens carry essentially no signal about the secret.

4. **Token-recovery@20 is slightly higher than @5**: Some models show marginal recovery at k=20 (moon: 5.5%), suggesting a few secret tokens may appear in the lower ranks by chance, but far less reliably than with logit lens.

5. **Top-ranked tokens are generic words**: The highest-scoring DoLa-direction tokens are common words like "kind", "reveal", "information", "everyone" -- tokens that the model naturally produces with high confidence at the final layer, unrelated to the secret.
