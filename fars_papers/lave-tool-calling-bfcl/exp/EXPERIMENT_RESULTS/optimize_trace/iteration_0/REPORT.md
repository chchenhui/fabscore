# Optimization Iteration 0: LAVE CFG-Constrained Decoding Hyperparameter Tuning

## Experiment Overview

Attempted to improve LAVE CFG-constrained diffusion decoding (Condition C) for LLaDA-8B-Instruct on BFCL-v3 Non-Live tool-calling benchmark. The original Condition C scored 27.52% success vs 36.19% unconstrained (A).

## Diagnosis

Three issues were identified in the original LAVE experiment:

1. **Low retry budget (max_retry_num_total=5)**: When a token is rejected 5 times, the algorithm enters AR fallback mode which produces garbage/trailing content. 85/350 examples hit AR fallback; 45/85 (53%) produced corrupt output.

2. **No output truncation**: LAVE outputs contained trailing text after the first valid `[...]` expression in 37/350 examples (vs 3/350 for unconstrained). The eval function adds `]` when missing but doesn't handle trailing content after `]`.

3. **Narrow beam search**: `top_k_per_mask=5, top_n_beam=5, random_n_beam=5` was insufficient for complex string arguments in Java/JS categories (66% AR fallback rate vs 16% for simple_python).

### Key Insight
When LAVE has 0 retries, its output is **identical** to unconstrained. The grammar constraint only affects examples where a token is rejected, and the intervention tends to make things worse.

## Fixes Applied

1. **Increased retry budget**: `max_retry_num_total` 5 -> 30
2. **Wider beam search**: `top_k_per_mask` 5 -> 10, `top_n_beam` 5 -> 10, `random_n_beam` 5 -> 10
3. **Output truncation**: Added `truncate_to_bracket()` post-processing to strip content after first balanced `[...]`
4. **Longer generation**: `max_tokens` 256 -> 512, `steps` 128 -> 256

## Results

### Optimized vs Original vs Baselines

| Config | Success Rate | AST Parse | Time/Ex |
|--------|-------------|-----------|---------|
| Unconstrained (A) | 36.19% +/- 0.27% | 91.71% +/- 1.02% | 11.4s |
| Best-of-2 (B) | 36.29% +/- 0.23% | 93.62% +/- 0.27% | 23.0s |
| Original LAVE (C) | 27.52% +/- 0.36% | 88.29% +/- 0.23% | 7.0s |
| **Optimized LAVE** | **27.71% +/- 0.23%** | **86.48% +/- 0.36%** | **14.3s** |
| LAVE v2 (medium retry) | 26.38% +/- 0.27% | 86.95% +/- 0.75% | ~10s |

### Per-Category Comparison (Optimized vs Original LAVE)

| Category | Orig Success | Opt Success | Delta |
|----------|-------------|-------------|-------|
| simple_python | 52.0% | 50.0% | -2.0pp |
| simple_java | 0.0% | 0.0% | 0.0pp |
| simple_javascript | 28.0% | 30.7% | +2.7pp |
| multiple | 36.0% | 36.0% | 0.0pp |
| parallel | 46.0% | 53.3% | **+7.3pp** |
| parallel_multiple | 15.3% | 14.0% | -1.3pp |
| irrelevance | 15.3% | 10.0% | **-5.3pp** |

## Key Observations

1. **Optimization yielded minimal improvement** (+0.19pp overall, from 27.52% to 27.71%). The improvement is within noise.

2. **The `parallel` category improved significantly** (+7.3pp) because the longer generation length (512 vs 256) gives more room for multi-call outputs, and the grammar constraint helps maintain structure.

3. **The `irrelevance` category degraded** (-5.3pp) because wider beams + more retries cause the grammar to force function-call syntax when `[]` is the correct answer.

4. **AST parse rates for Java/JS worsened** (Java: 66.7% -> 60.0%, JS: 66.0% -> 58.0%). The wider beam search with more retries introduces MORE token substitutions that corrupt string arguments.

5. **More constraint intervention = worse results**: Every variation tested showed that LAVE's constraint mechanism hurts more than it helps for BFCL tool-calling. The constraint corrupts tokens (especially string content) rather than fixing syntax errors.

6. **The fundamental limitation**: LAVE's token-by-token constraint enforcement with beam search is not suited for outputs with complex string arguments. The model already achieves ~92% parse rate unconstrained; the remaining 8% parse failures are mostly in string-heavy categories where LAVE's intervention makes things worse.

## Conclusion

The LAVE CFG-constrained decoding approach cannot be optimized to match or exceed the unconstrained baseline for BFCL tool-calling. The constraint mechanism introduces more errors than it fixes, particularly for string-heavy categories (Java/JS) and the irrelevance category. The 27.71% optimized result is marginally better than the 27.52% original but still ~8.5pp below unconstrained (36.19%).
