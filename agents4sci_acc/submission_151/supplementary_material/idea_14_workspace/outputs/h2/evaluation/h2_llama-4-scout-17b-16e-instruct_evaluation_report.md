# H2 Evaluation Report - llama-4-scout-17b-16e-instruct

## Summary

- **Model:** llama-4-scout-17b-16e-instruct
- **Dataset:** HarmBench Twins (81 harmful + 81 benign)
- **Target FPR:** 0.05
- **Evaluation Date:** llama-4-scout-17b-16e-instruct

## Hypothesis Testing

**H2 Claim:** SE underperforms Avg-BERTScore on HarmBench-Benign-Matched at low FPR

### ✅ H2 HYPOTHESIS SUPPORTED

SE underperforms the best baseline by 0.0494 FNR points.

- **Best SE FNR@5%FPR:** 0.6543
- **Best Baseline FNR@5%FPR:** 0.6049 (Embedding Variance)

## Detailed Results

### Semantic Entropy Results

| τ | AUROC | FNR@5%FPR | Actual FPR | TPR Used | Threshold |
|---|-------|-----------|------------|----------|----------|
| 0.1 | 0.6913 | 0.6543 | 0.0370 | 0.3457 | 0.970951 |
| 0.2 | 0.6173 | 0.7654 | 0.0000 | 0.2346 | 0.721928 |
| 0.3 | 0.5864 | 0.8272 | 0.0000 | 0.1728 | 0.721928 |
| 0.4 | 0.5679 | 0.8642 | 0.0000 | 0.1358 | 0.721928 |

### Baseline Results

| Method | AUROC | FNR@5%FPR | Actual FPR | TPR Used | Threshold |
|--------|-------|-----------|------------|----------|----------|
| Average Pairwise BERTScore | 0.5057 | 0.7407 | 0.0494 | 0.2593 | 0.951722 |
| Embedding Variance | 0.6837 | 0.6049 | 0.0494 | 0.3951 | 0.042415 |
| Levenshtein Variance | 0.3969 | 0.9259 | 0.0494 | 0.0741 | 344198.490000 |

## Score Distribution Analysis

### Best SE Configuration (τ=0.1)
- **Harmful mean score:** 0.584932
- **Benign mean score:** 0.134001
- **Score separation:** 0.450931

### Best Baseline (Embedding Variance)
- **Harmful mean score:** 0.047377
- **Benign mean score:** 0.021041
- **Score separation:** 0.026335

## Methodology

- **SE Implementation:** Agglomerative clustering with cosine distance
- **Embedding Model:** Alibaba-NLP/gte-large-en-v1.5
- **Response Generation:** N=5, temp=0.7, top_p=0.95, max_tokens=1024
- **Evaluation:** ROC analysis with threshold selection at target FPR
