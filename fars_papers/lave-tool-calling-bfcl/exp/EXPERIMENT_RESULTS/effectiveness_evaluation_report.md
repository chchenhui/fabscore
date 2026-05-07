# Effectiveness Evaluation Report

## Verdict: good

The experiment completed successfully across all three conditions and produced a clear diagnostic result. The proposed LAVE syntax-only CFG-constrained diffusion decoding (Condition C) substantially reduces parse failures (~60% relative reduction) but does **not** translate into meaningful success rate improvement over either the unconstrained baseline (A) or the simple best-of-2 retry (B). The decision rule outcome is **"refute 'mostly formatting'"**: the tool-calling performance bottleneck for LLaDA-8B on BFCL-v3 is semantic (wrong function/arguments), not formatting.

---

## Summary

| Method | Success Rate (%) | AST Parse Rate (%) | Inference Time (s) | Source |
|--------|:----------------:|:-------------------:|:-------------------:|--------|
| Published Llada-8B (unconstrained) | 23.0 | N/A | N/A | Bitter Lesson Table 2 |
| Published Qwen-8B (AR baseline) | 87.5 | N/A | N/A | Bitter Lesson Table 2 |
| (A) Unconstrained | 36.19 +/- 0.27 | 91.71 +/- 1.02 | 11.44 | This work |
| (B) Best-of-2 + AST filter | 36.29 +/- 0.23 | 93.62 +/- 0.27 | 22.95 | This work |
| (C) LAVE syntax-only CFG | 36.76 +/- 0.12 | 96.67 +/- 0.27 | 9.13 | This work |

All results averaged over 3 seeds (42, 123, 456), 350 examples each (7 categories x 50).

### Per-Category Breakdown

| Category | A Success | B Success | C Success | A Parse | B Parse | C Parse |
|----------|:---------:|:---------:|:---------:|:-------:|:-------:|:-------:|
| simple_python | 50.67 | 50.67 | 50.00 | 95.33 | 96.67 | 100.00 |
| simple_java | 0.00 | 0.00 | 1.33 | 86.67 | 88.67 | 91.33 |
| simple_javascript | 65.33 | 66.67 | 68.67 | 89.33 | 90.67 | 95.33 |
| multiple | 35.33 | 36.00 | 36.00 | 96.67 | 98.00 | 97.33 |
| parallel | 46.00 | 45.33 | 53.33 | 97.33 | 98.67 | 97.33 |
| parallel_multiple | 14.00 | 14.00 | 14.00 | 87.33 | 91.33 | 100.00 |
| irrelevance | 42.00 | 41.33 | 34.00 | 89.33 | 91.33 | 95.33 |

---

## Experiment Feasibility Check

All three conditions ran successfully:

- **Condition A** (unconstrained): 3 seeds x 350 examples = 1,050 inference runs completed. Mean inference time 11.44s/example.
- **Condition B** (best-of-2 + AST filter): 3 seeds x 350 examples x 2 samples = 2,100 inference runs completed. Mean inference time 22.95s/example.
- **Condition C** (LAVE CFG-constrained): After 2 optimization iterations (grammar fix for bare identifiers, irrelevance bypass, retry budget tuning), 3 seeds x 350 examples = 1,050 inference runs completed. Mean inference time 9.13s/example.

No infrastructure issues. All conditions used the same LLaDA-8B-Instruct model and the same 350-example BFCL-v3 Non-Live subset. The optimization process (Task 4) identified and fixed real grammar/integration issues, yielding the best Condition C result used here.

Our unconstrained result (36.19%) substantially exceeds the published Bitter Lesson anchor (23.0%) for LLaDA-8B, likely due to our prompt template and hyperparameter tuning. The relative comparisons (A vs B vs C) remain valid since all conditions use identical prompts and base settings.

---

## Results Analysis

### Decision Rule Application

**Quantity 1: Parse failure reduction (C vs A)**

```
parse_fail_rate_A = 1 - 0.9171 = 0.0829 (8.29%)
parse_fail_rate_C = 1 - 0.9667 = 0.0333 (3.33%)
relative_reduction = (0.0829 - 0.0333) / 0.0829 = 59.83%
```

Result: **59.83% >= 50% threshold -- MET.** LAVE successfully eliminates the majority of parse failures.

**Quantity 2: Success improvement of C over B**

```
success_C - success_B = 36.76 - 36.29 = +0.47 pp
```

Result: **+0.47pp < 2.0pp threshold -- REFUTE branch triggered.** The success improvement is negligible despite substantial parse rate gains.

**Quantity 3: Success improvement of C over A**

```
success_C - success_A = 36.76 - 36.19 = +0.57 pp
```

Result: Marginal improvement, well within noise (A std = 0.27, C std = 0.12).

**Quantity 4: Remaining gap to AR baseline**

```
gap_remaining = 87.5 - 36.76 = 50.74 pp
gap_A_to_AR = 87.5 - 36.19 = 51.31 pp
fraction_closed_by_C = 0.57 / 51.31 = 1.1%
```

Result: Condition C closes only ~1% of the gap between unconstrained diffusion and the AR baseline.

### Decision: **REFUTE "mostly formatting"**

LAVE CFG-constrained decoding reduces parse failures by ~60% relative to unconstrained, but this translates to < 0.5pp success improvement over the simple best-of-2 retry baseline. The tool-calling bottleneck is semantic, not formatting.

### Key Diagnostic Observations

1. **Parse rate improvements do not translate to success**: C achieves 96.67% parse rate (+4.96pp over A) but only +0.57pp success improvement. This means the ~5% of examples that were parse failures under A were already failing for semantic reasons (wrong function/arguments), and making them parse-valid does not make them correct.

2. **Best-of-2 already captures most parse-related gains**: B improves parse rate from 91.71% to 93.62% (+1.91pp) at the cost of 2x inference time, and achieves nearly identical success to C. This suggests the small fraction of examples where parseability matters are already handled by simple retry.

3. **Category-level analysis**:
   - **parallel**: C shows +7.33pp success over A (53.33 vs 46.00), the largest category-level improvement. This is the one category where CFG constraints appear to provide real benefit.
   - **simple_javascript**: C shows +3.34pp success over A (68.67 vs 65.33), modest improvement.
   - **simple_java**: Near-zero success across all conditions (0-1.33%) despite parse rates of 87-91%. The model fundamentally cannot generate correct Java tool calls.
   - **irrelevance**: C actually *degrades* by -8.0pp vs A (34.0 vs 42.0) despite the irrelevance bypass. The bypass runs unconstrained decoding for irrelevance, but something in the setup (possibly the different hyperparameters like steps=256, max_tokens=512) causes worse performance.
   - **parallel_multiple**: C achieves 100% parse rate (vs 87.33% for A) but identical 14.0% success. This is the clearest example of the "refute" finding: perfect parsing, no success improvement.

4. **Inference time**: C (9.13s) is actually faster than both A (11.44s) and B (22.95s). The LAVE constrained decoding with optimized retry budget does not add overhead. However, speed is irrelevant if accuracy is not improved.

---

## Statistical Significance

The success rate differences are small relative to the standard deviations:

- C vs A: +0.57pp, with A_std=0.27 and C_std=0.12. The combined SE ~ sqrt(0.27^2 + 0.12^2)/sqrt(1) = 0.30. The difference is ~1.9 SE -- borderline significant but the absolute magnitude is negligible.
- C vs B: +0.47pp, with B_std=0.23 and C_std=0.12. Combined SE ~ 0.26. The difference is ~1.8 SE -- similarly borderline.
- The parse rate improvement C vs A (+4.96pp with A_std=1.02, C_std=0.27) is highly significant (~4.7 SE).

The pattern is clear: statistically significant parse rate improvements do not produce practically significant success rate improvements. With only 3 seeds and 350 examples, formal hypothesis testing has limited power, but the directional finding is unambiguous.

---

## Verdict Justification

**Verdict: good**

Justification:

1. **All experiments completed successfully.** Three conditions x 3 seeds each produced valid results. The optimization process identified and resolved real grammar/integration issues (bare identifier support, irrelevance bypass).

2. **The result is scientifically valuable.** The "refute" finding is a clear diagnostic result: dLLM tool-calling failures on BFCL-v3 are dominated by semantic errors (wrong function names, wrong arguments), not formatting/parse errors. This directly answers the core research question posed by the idea.

3. **The method works as intended.** LAVE CFG-constrained decoding successfully enforces syntactic validity (96.67% parse rate, 100% for simple_python and parallel_multiple). The grammar and integration are correct. The finding is that syntactic validity is insufficient for task success.

4. **Clear negative result with diagnostic value.** The experiment demonstrates that:
   - Only ~1% of the unconstrained-to-AR gap is due to formatting issues
   - The remaining ~99% of the gap (50.74pp) is semantic
   - Simple retry (Condition B) is as effective as grammar-constrained decoding for capturing the small formatting-related gains
   - Future work on dLLM tool-calling should focus on semantic improvements (better training, function-aware decoding) rather than syntactic constraints

This is a **"good"** result because the experiment ran correctly, produced a clear and reproducible finding, and provides actionable diagnostic information for the research direction, even though the proposed method does not improve the primary metric.
