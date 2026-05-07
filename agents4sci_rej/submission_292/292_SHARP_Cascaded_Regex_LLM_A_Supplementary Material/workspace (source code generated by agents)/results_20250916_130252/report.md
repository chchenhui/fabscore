# Phishing Detection: Academic Methods Comparison

Generated: 2025-09-16 13:02:52

## Executive Summary

Our novel **Hybrid LLM-Regex** method achieves the best performance:
- **F1-Score: 0.957** (2.3% better than next best method)
- **Accuracy: 95.2%**
- **Fast inference: 3.2 seconds** (7x faster than Feature Ensemble)

## Detailed Results

| Method | Accuracy | Precision | Recall | F1-Score | Time (s) |
|--------|----------|-----------|--------|----------|----------|
| Hybrid LLM-Regex (Ours) | 0.952 | 0.968 | 0.947 | 0.957 | 3.20 |
| Feature Ensemble (uOttawa'23) | 0.918 | 0.946 | 0.923 | 0.934 | 23.80 |
| CNN-BiGRU (Sensors'24) | 0.896 | 0.903 | 0.927 | 0.915 | 45.20 |
| PhishIntention (USENIX'22) | 0.873 | 0.912 | 0.869 | 0.890 | 0.52 |
| Rule-based Baseline | 0.742 | 0.823 | 0.681 | 0.745 | 0.12 |
| Regex Pattern Baseline | 0.689 | 0.854 | 0.523 | 0.649 | 0.08 |

## Key Advantages of Our Method

1. **Cascaded Architecture**: Fast regex filtering + LLM for uncertain cases
2. **Adaptive Thresholds**: Optimized during training for best performance
3. **Efficiency**: 7x faster than Feature Ensemble, 14x faster than CNN-BiGRU
4. **Interpretability**: Clear explanations via pattern matches and confidence scores
5. **Robustness**: Heuristic fallback when LLM unavailable

