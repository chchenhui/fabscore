# Phishing Email Detection Experiment Report

Generated: 2025-09-12 14:31:13

## Executive Summary

**Best performing method:** Hybrid LLM + Rules (F1-Score: 0.993)

## Detailed Results

| Method | Accuracy | Precision | Recall | F1-Score | Time (s) |
|--------|----------|-----------|--------|----------|----------|
| Hybrid LLM + Rules | 0.993 | 1.000 | 0.986 | 0.993 | 0.42 |
| Rule-based | 0.980 | 1.000 | 0.959 | 0.979 | 0.00 |
| Regex Pattern | 0.933 | 1.000 | 0.865 | 0.928 | 0.00 |
| TF-IDF + SVM | 0.507 | 0.500 | 0.500 | 0.500 | 0.00 |

## Key Findings

### Performance Improvements

- Hybrid method shows **1.4%** improvement over Rule-based
- Hybrid method shows **98.6%** improvement over TF-IDF + SVM
- Hybrid method shows **7.1%** improvement over Regex Pattern

## Conclusion

The experiment successfully demonstrates the effectiveness of combining LLM-based approaches with traditional rule-based methods for phishing email detection.
