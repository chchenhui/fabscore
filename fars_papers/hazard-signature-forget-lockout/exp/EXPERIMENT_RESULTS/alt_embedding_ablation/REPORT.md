# Alternative Embedding Ablation: Qwen3-Embedding-4B Robustness Check

## Experiment Overview

Robustness check to verify that the relative ranking of conditions A, B, and C is not an artifact of the specific embedding model (BAAI/bge-m3). Replaced the embedding backend with Qwen/Qwen3-Embedding-4B (4B parameter model, 2560-dimensional output) and re-ran all three main conditions.

## Setup

- **Embedding model**: Qwen/Qwen3-Embedding-4B (4B params, dim=2560)
- **Baseline model**: BAAI/bge-m3 (568M params, dim=1024)
- **Matching strategy**: Hazard-set containment (fuzzy)
- **k=3, delta=6** (same as original experiments)
- **Data**: 100 benign + 10 poisoned + 50 paraphrases, 12 eval queries
- **GPU**: 1x GPU via TrainService (job: dlc1bovthllyokqw)

## Key Results

| Condition | Embedding | PRP@3 | Benign Recall@3 | WriteBlockRate | BenignFalseBlock |
|-----------|-----------|-------|-----------------|----------------|------------------|
| A | bge-m3 | 0.9444 | 1.0 | 0% | 0% |
| A | Qwen3-Embedding-4B | 0.9167 | 1.0 | 0% | 0% |
| B (delta=6) | bge-m3 | 0.0 | 0.97 | 0% | 0% |
| B (delta=6) | Qwen3-Embedding-4B | 0.0 | 0.97 | 0% | 0% |
| C | bge-m3 | 0.0 | 1.0 | 100% | 3% |
| C | Qwen3-Embedding-4B | 0.0 | 1.0 | 100% | 3% |

## Key Observations

1. **Ordering preserved**: The relative ranking C > B > A holds identically across both embedding backends.
   - PRP@3: A (high, ~0.92-0.94) >> B (0.0) = C (0.0)
   - Benign Recall@3: C (1.0) > B (0.97), A (1.0) = C (1.0)
   - C dominates on both metrics simultaneously.

2. **Near-identical results**: Condition B and C produce byte-identical metrics across both backends (PRP=0.0, BR=0.97/1.0). The only minor difference is Condition A PRP@3 (0.9167 vs 0.9444), reflecting slightly different retrieval dynamics but no qualitative change.

3. **Embedding-independent write metrics**: WriteBlockRate (100%) and BenignFalseBlock (3%) are identical because they depend on LLM-derived hazard signatures, not embeddings.

4. **Conclusion**: The HST method's advantage is confirmed to be retrieval-backend-agnostic. The choice of embedding model does not affect the relative effectiveness of the three conditions.
