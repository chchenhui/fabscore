# H7 Scoring Report - llama-3.3-70b-instruct

## Summary

- **Model:** llama-3.3-70b-instruct
- **Input file:** /research_storage/outputs/h7/llama-3.3-70b-instruct_h7_responses.jsonl
- **Total response sets:** 120
- **Successfully scored:** 120
- **Failed scores:** 0
- **Success rate:** 100.0%
- **Output file:** /research_storage/outputs/h7/llama-3.3-70b-instruct_h7_scores.jsonl

## Scoring Methods

### Semantic Entropy
- **τ grid:** [0.1, 0.2, 0.3, 0.4]
- **Embedding model:** Alibaba-NLP/gte-large-en-v1.5
- **Method:** Agglomerative clustering with cosine distance
- **Diagnostics captured:** Cluster counts, embedding matrices, distance thresholds

### Baseline Metrics
- **Avg Pairwise BERTScore:** Mean pairwise BERTScore F1 across response sets
- **Embedding Variance:** Variance of sentence embeddings within response sets
- **Levenshtein Variance:** Variance of edit distances within response sets

## Dataset Composition

- **Harmful samples:** 60
- **Benign samples:** 60
- **Total samples:** 120

## Score Statistics (Averages)

- **SE (τ=0.1) average:** 0.661462\n- **SE (τ=0.2) average:** 0.362464\n- **SE (τ=0.3) average:** 0.224126\n- **SE (τ=0.4) average:** 0.161197\n- **Avg Pairwise Bertscore average:** 0.906507\n- **Embedding Variance average:** 0.056240\n- **Levenshtein Variance average:** 104069.252667\n
## Technical Details

- **Processing time:** Detailed per-prompt timing logged
- **Response validation:** Minimum 2 valid responses required per sample
- **Diagnostic data:** Complete SE clustering information preserved
- **Metadata preservation:** All original generation metadata retained

## Output Structure

The scoring output file contains:
- Individual prompt scores with diagnostics
- Semantic entropy values for each τ threshold
- Complete baseline metric calculations
- Response metadata (length statistics, quality metrics)
- Original generation metadata preservation

## Next Steps

1. Run H7 evaluation pipeline: `modal run src/experiments/h7/run_h7_evaluation.py::main --model=llama-3.3-70b-instruct`
2. Compare results with H1 baseline experiments
3. Validate H7 success criteria for SOTA model performance
