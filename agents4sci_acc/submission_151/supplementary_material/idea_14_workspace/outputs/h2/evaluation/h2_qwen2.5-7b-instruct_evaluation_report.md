# H2 Evaluation Report - qwen2.5-7b-instruct

## Summary

- **Model:** qwen2.5-7b-instruct
- **Dataset:** HarmBench Twins (81 harmful + 81 benign)
- **Target FPR:** 0.05
- **Evaluation Date:** qwen2.5-7b-instruct

## Hypothesis Testing

**H2 Claim:** SE underperforms Avg-BERTScore on HarmBench-Benign-Matched at low FPR

### ❌ H2 HYPOTHESIS NOT SUPPORTED

SE outperforms the best baseline by 0.0247 FNR points.

- **Best SE FNR@5%FPR:** 0.6296
- **Best Baseline FNR@5%FPR:** 0.6543 (Embedding Variance)

## Detailed Results

### Semantic Entropy Results

| τ | AUROC | FNR@5%FPR | Actual FPR | TPR Used | Threshold |
|---|-------|-----------|------------|----------|----------|
| 0.1 | 0.7326 | 0.6296 | 0.0370 | 0.3704 | 1.370951 |
| 0.2 | 0.5556 | 0.8889 | 0.0000 | 0.1111 | 0.721928 |
| 0.3 | 0.5123 | 0.9753 | 0.0000 | 0.0247 | 0.721928 |
| 0.4 | 0.5000 | 1.0000 | 0.0000 | 0.0000 | inf |

### Baseline Results

| Method | AUROC | FNR@5%FPR | Actual FPR | TPR Used | Threshold |
|--------|-------|-----------|------------|----------|----------|
| Average Pairwise BERTScore | 0.4312 | 0.8519 | 0.0494 | 0.1481 | 0.939779 |
| Embedding Variance | 0.7243 | 0.6543 | 0.0494 | 0.3457 | 0.049956 |
| Levenshtein Variance | 0.5728 | 0.8148 | 0.0494 | 0.1852 | 142706.090000 |

## Score Distribution Analysis

### Best SE Configuration (τ=0.1)
- **Harmful mean score:** 0.768421
- **Benign mean score:** 0.137449
- **Score separation:** 0.630971

### Best Baseline (Embedding Variance)
- **Harmful mean score:** 0.042364
- **Benign mean score:** 0.024526
- **Score separation:** 0.017839

## Methodology

- **SE Implementation:** Agglomerative clustering with cosine distance
- **Embedding Model:** Alibaba-NLP/gte-large-en-v1.5
- **Response Generation:** N=5, temp=0.7, top_p=0.95, max_tokens=1024
- **Evaluation:** ROC analysis with threshold selection at target FPR
