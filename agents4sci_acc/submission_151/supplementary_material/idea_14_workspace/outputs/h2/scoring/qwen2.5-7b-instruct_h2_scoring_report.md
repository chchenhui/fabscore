# H2 Scoring Report - qwen2.5-7b-instruct

## Summary

- **Model:** qwen2.5-7b-instruct
- **Input file:** /research_storage/outputs/h2/qwen2.5-7b-instruct_h2_responses.jsonl
- **Total response sets:** 162
- **Successfully scored:** 162
- **Success rate:** 100.0%
- **Output file:** /research_storage/outputs/h2/scoring/qwen2.5-7b-instruct_h2_scores.jsonl

## Scoring Methods

### Semantic Entropy
- **τ grid:** [0.1, 0.2, 0.3, 0.4]
- **Embedding model:** Alibaba-NLP/gte-large-en-v1.5
- **Method:** Agglomerative clustering with cosine distance

### Baseline Metrics
- **Avg Pairwise BERTScore:** Mean pairwise BERTScore F1
- **Embedding Variance:** Variance of sentence embeddings
- **Levenshtein Variance:** Variance of edit distances

## Dataset Composition

- **Harmful samples:** 81
- **Benign samples:** 81

## Score Statistics (Sample)

- **SE (τ=0.1) average:** 0.452935
- **Avg BERTScore average:** 0.895975
- **Embedding Variance average:** 0.033445
