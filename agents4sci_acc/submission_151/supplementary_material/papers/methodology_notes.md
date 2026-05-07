# Methodology Notes: Semantic Entropy Implementation Details

## Semantic Entropy Variant Used

### Core Implementation
We use **Discrete Semantic Entropy via Embedding-based Clustering**, a black-box variant that requires only API access to the target model.

### Technical Specifications

#### Response Generation
- **Sampling:** N=5 responses per prompt (canonical setting)
- **Temperature:** 0.7 (balances diversity and coherence)
- **Top-p:** 0.95 (nucleus sampling for quality)
- **Max Tokens:** 1024 (allows full response development)
- **API:** OpenRouter for both Qwen-2.5-7B-Instruct and Llama-4-Scout-17B

#### Embedding Generation
- **Model:** Alibaba-NLP/gte-large-en-v1.5
- **Dimension:** 1024-dimensional dense vectors
- **Normalization:** L2-normalized for cosine similarity
- **Device:** CUDA-accelerated on A100-40GB

#### Clustering Algorithm
- **Method:** Agglomerative Hierarchical Clustering
- **Distance Metric:** Cosine distance (1 - cosine_similarity)
- **Linkage:** Average linkage
- **Threshold τ:** Grid search over {0.1, 0.2, 0.3, 0.4}
- **Implementation:** scikit-learn AgglomerativeClustering

#### Entropy Calculation
```python
# Pseudo-code for clarity
clusters = agglomerative_clustering(embeddings, threshold=τ)
cluster_probs = [count/N for count in cluster_sizes]
semantic_entropy = -sum(p * log(p) for p in cluster_probs if p > 0)
```

### Key Differences from Original SE (Nature 2024)

| Aspect | Original SE (Farquhar et al.) | Our Implementation |
|--------|------------------------------|-------------------|
| **Primary Application** | Hallucination detection | Jailbreak detection |
| **Clustering Method** | Bidirectional entailment via NLI | Embedding cosine similarity |
| **Access Required** | Token log-probabilities | Black-box API only |
| **Semantic Granularity** | Binary (equivalent/different) | Continuous (similarity threshold) |
| **Computational Cost** | O(N²) NLI calls | O(N²) embedding comparisons |

### Rationale for Our Variant

1. **Black-box Constraint:** Most production APIs (OpenAI, Anthropic, Google) don't expose token probabilities
2. **Computational Efficiency:** Embedding clustering is ~100x faster than NLI-based clustering
3. **Continuous Semantics:** Threshold τ allows tuning semantic granularity
4. **Proven Effectiveness:** Embedding similarity correlates strongly with semantic equivalence

## Contrast with Related Methods

### vs. SelfCheckGPT (Manakul et al., 2023)
- **SelfCheckGPT:** Uses consistency for hallucination detection via BERTScore/n-grams
- **Our Approach:** Adapts consistency principle to safety domain with semantic clustering
- **Key Difference:** We measure entropy over semantic clusters, not raw similarity scores

### vs. SemanticSmooth (Wang et al., 2024)
- **SemanticSmooth:** Input-side perturbations to detect jailbreaks
- **Our Approach:** Output-side sampling to measure behavioral uncertainty
- **Key Difference:** No auxiliary transformation models needed

### vs. White-box Methods (Gradient Cuff, HiddenDetect)
- **White-box:** Analyze internal gradients/activations
- **Our Approach:** Purely behavioral signal from output distribution
- **Key Difference:** Works with closed-source APIs

## Critical Design Decisions

### Why Not Use Token Probabilities?
1. **Access Limitations:** Major API providers don't expose them
2. **Generalizability:** Method works identically across all models
3. **Semantic Focus:** Embeddings capture meaning better than token distributions

### Why Agglomerative Clustering?
1. **Interpretability:** Clear threshold parameter τ
2. **Flexibility:** No need to pre-specify number of clusters
3. **Proven:** Standard approach in semantic similarity tasks

### Why These Specific τ Values?
- **0.1:** Very fine-grained (near-identical responses only)
- **0.2:** Canonical setting from preliminary experiments  
- **0.3:** Moderate grouping (paraphrases cluster together)
- **0.4:** Coarse grouping (same topic clusters together)

## Implementation Bug Fixes Applied

### Semantic Entropy -0.0 Bug
- **Issue:** Numpy's log function returns -0.0 for log(1)
- **Impact:** Comparison issues in downstream processing
- **Fix:** Added explicit handling: `entropy = 0.0 if entropy == -0.0 else entropy`

### FPR Threshold Selection
- **Issue:** Original function selected first ROC point (always FPR=0, TPR=0)
- **Fix:** Conservative selection of rightmost valid point with FPR ≤ target
- **Validation:** Comprehensive test suite with edge cases

## Experimental Controls

### Randomness Control
- **Seed:** 42 for all experiments
- **GPU Determinism:** CUDA deterministic mode enabled
- **Sampling Seeds:** Tracked per response generation

### Data Contamination Prevention
- **ID Manifests:** Separate JSON files tracking train/val/test splits
- **Leakage Guards:** Automated checks preventing overlap
- **Frozen Parameters:** No tuning on test data

## Computational Requirements

### Per-Prompt Processing
1. Generate N=5 responses: ~10-15 seconds (API latency)
2. Compute embeddings: ~0.5 seconds (GPU)
3. Clustering & entropy: ~0.01 seconds (CPU)
4. Total: ~15 seconds per prompt

### Full Experiment (200 prompts)
- Response generation: ~50 minutes
- Scoring: ~5 minutes  
- Evaluation: ~1 minute
- Total: ~1 hour per model

## Links and Citations

### Core References
- **Original Semantic Entropy:** [Farquhar et al., Nature 2024](https://www.nature.com/articles/s41586-024-07421-0)
- **SelfCheckGPT:** [Manakul et al., ACL 2023](https://aclanthology.org/2023.emnlp-main.557/)
- **Embedding Model:** [GTE-Large](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)

### Benchmark Papers
- **JailbreakBench:** [Chao et al., NeurIPS 2024](https://arxiv.org/abs/2404.01318)
- **HarmBench:** [Mazeika et al., 2024](https://arxiv.org/abs/2402.04249)
- **SORRY-Bench:** [Röttger et al., 2024](https://arxiv.org/abs/2406.14598)