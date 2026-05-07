# Phishing Email Detection Experiment Report

Generated: 2025-09-12 09:06:07

## Executive Summary

**Best performing method:** Hybrid LLM + Rules (F1-Score: 1.000)

## Detailed Results

| Method | Accuracy | Precision | Recall | F1-Score | Time (s) |
|--------|----------|-----------|--------|----------|----------|
| Hybrid LLM + Rules | 1.000 | 1.000 | 1.000 | 1.000 | 0.62 |
| Rule-based | 0.993 | 1.000 | 0.987 | 0.993 | 0.00 |
| Regex Pattern | 0.947 | 1.000 | 0.896 | 0.945 | 0.01 |
| TF-IDF + SVM | 0.560 | 0.585 | 0.494 | 0.535 | 0.00 |

## Key Findings

### Performance Improvements

- Hybrid method shows **0.7%** improvement over Rule-based
- Hybrid method shows **86.8%** improvement over TF-IDF + SVM
- Hybrid method shows **5.8%** improvement over Regex Pattern

## Conclusion

The experiment successfully demonstrates the effectiveness of combining LLM-based approaches with traditional rule-based methods for phishing email detection.
