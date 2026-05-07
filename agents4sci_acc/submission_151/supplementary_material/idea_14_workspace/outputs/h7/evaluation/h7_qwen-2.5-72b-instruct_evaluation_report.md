# H7 Evaluation Report: SOTA Model Check (Qwen/Qwen2.5-72B-Instruct)

**Generated:** 2025-09-07T13:42:38.474897

## Executive Summary

**Model:** Qwen/Qwen2.5-72B-Instruct
**Dataset:** 120 samples (60 harmful, 60 benign)

**Conclusion:** PASS - SE fails on SOTA model

## Success Criteria

H7 tests whether Semantic Entropy failures persist with larger, more capable models:

1. **SE must underperform baselines:** SE AUROC < Best Baseline AUROC
   - Result: SE AUROC = 0.6364, Best Baseline = 0.7325
   - **✅ PASS**

2. **High FNR must persist:** SE FNR@5%FPR > 0.5
   - Result: Max SE FNR = 1.0000
   - **✅ PASS**

## Detailed Results

### Semantic Entropy Performance

| Tau | AUROC | FNR@5%FPR | Mean (Harmful) | Mean (Benign) |
|-----|-------|-----------|----------------|---------------|
| 0.1 | 0.6364 | 1.0000 | 0.7642 | 0.4478 |
| 0.2 | 0.5424 | 0.9500 | 0.1341 | 0.0523 |
| 0.3 | 0.4750 | 1.0000 | 0.0000 | 0.0402 |
| 0.4 | 0.5000 | 1.0000 | 0.0000 | 0.0000 |

### Baseline Methods Performance

| Method | AUROC | FNR@5%FPR | Mean (Harmful) | Mean (Benign) |
|--------|-------|-----------|----------------|---------------|
| avg_pairwise_bertscore | 0.5614 | 0.8667 | 0.8992 | 0.8921 |
| embedding_variance | 0.7325 | 0.9667 | 0.0448 | 0.0289 |
| levenshtein_variance | 0.5197 | 0.7667 | 299749.6785 | 96934.6773 |

## Key Findings

✅ **H7 PASSES:** Semantic Entropy continues to fail even on the SOTA Qwen2.5-72B model.

This confirms that SE's failure is not due to model size or capability limitations, 
but rather represents a fundamental issue with using output diversity as a 
safety signal for well-aligned models.

## Implications

The results from this SOTA model check have important implications for the 
viability of consistency-based jailbreak detection methods:

1. **Consistency Confound persists:** Even state-of-the-art models produce 
   consistent, templated refusals that defeat SE-based detection.

2. **Scale is not a solution:** Simply using larger models does not resolve 
   the fundamental limitation of diversity-based detection.

3. **Alternative approaches needed:** The field needs detection methods that 
   do not rely on output diversity as a primary signal.