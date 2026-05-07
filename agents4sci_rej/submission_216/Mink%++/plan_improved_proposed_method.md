# Implementation Plan: Improved Proposed Method

## Overview
This implementation enhances the existing distribution shape analysis method by incorporating:
1. **Weighted Harmonic Mean Aggregation** for distribution shape features (replacing simple arithmetic mean)
2. **Sequence Coherence Penalty** based on adjacent token probability consistency

## Key Changes

### 1. Weighted Harmonic Mean Aggregation
**Current Issue**: The existing method uses simple arithmetic mean for aggregating distribution shape features (lines 196-212 in `multilayer_concentration_fusion`), which can be sensitive to extreme outlier values.

**Enhancement**: Replace arithmetic mean with weighted harmonic mean that:
- Is more robust to outliers than arithmetic mean
- Provides better statistical stability
- Weights features based on their reliability/importance

### 2. Sequence Coherence Penalty
**Current Gap**: The existing method doesn't consider the consistency of probability patterns between adjacent tokens, which could help distinguish memorized sequences.

**Enhancement**: Add sequence coherence penalty that:
- Analyzes probability consistency between adjacent tokens
- Detects unnatural probability jumps that indicate memorization
- Incorporates smoothness measures into the final score

### 3. Layer-wise Feature Fusion Enhancement
- Enhanced weighted combination of features from different layers using harmonic mean
- Early layers capture syntactic patterns, deeper layers capture semantic patterns
- Improved weighting based on layer importance and coherence consistency
- Temporal consistency across layers for memorized vs. novel content

## Implementation Details

### Core Functions:

1. **`extract_intermediate_layers(model, input_ids, layer_indices)`**
   - Extract hidden states from specified intermediate layers
   - Compute probability distributions at each layer using layer-specific projection
   - Handle different model architectures (Pythia, Mamba)
   - Return layer-wise probability distributions

2. **`compute_shannon_entropy(probs)`**
   - Calculate Shannon entropy: H(X) = -Σ p(x) log p(x)
   - Higher entropy indicates more uniform probability distribution
   - Lower entropy indicates more concentrated/confident predictions
   - Normalize by log(vocab_size) for comparable values

3. **`compute_gini_coefficient(probs)`**
   - Calculate Gini coefficient to measure probability concentration
   - G = 1 - Σ (2i - n - 1) * p_i / (n * Σ p_i)
   - Values closer to 1 indicate higher concentration
   - Values closer to 0 indicate more uniform distribution

4. **`compute_topk_concentration(probs, k_values=[5, 10, 20, 50])`**
   - Calculate fraction of probability mass in top-k tokens
   - Multiple k values to capture different concentration levels
   - Higher values indicate more concentrated distributions
   - Useful for detecting overconfident memorized predictions

5. **`compute_effective_vocab_size(probs, threshold=0.9)`**
   - Number of tokens needed to capture threshold% of probability mass
   - Smaller effective vocabulary indicates higher concentration
   - Discriminative for memorized vs. novel content patterns
   - Normalized by actual vocabulary size

6. **`multilayer_concentration_fusion(mink_plus_score, layer_features, layer_weights)`**
   - Combine Min-K%++ score with multi-layer concentration features
   - Adaptive weighting based on layer depth and feature reliability
   - Account for varying importance of syntactic vs. semantic patterns
   - Return enhanced detection score

### Key Changes from Previous Method:
- Method name: `mink++_multilayer_concentration_{ratio}`
- Extract features from layers at 25%, 50%, 75% depth
- Concentration metrics: Shannon entropy, Gini coefficient, Top-k concentration, Effective vocab size
- Layer-wise adaptive weighting and temporal consistency analysis
- Maintained compatibility with existing input/output formats

## Algorithm Flow:

1. **For each text sample:**
   - Compute forward pass through model to get intermediate layer outputs
   - Extract hidden states from selected layers (1/4, 1/2, 3/4 depth)
   - Project hidden states to vocabulary space to get layer-wise probabilities
   - Handle different model architectures appropriately

2. **Feature Processing:**
   - Compute concentration metrics at each selected layer:
     - Shannon entropy for information content
     - Gini coefficient for probability inequality
     - Top-k concentration for various k values
     - Effective vocabulary size for 90% threshold
   - Apply layer-specific normalization and weighting
   - Analyze temporal consistency across layers

3. **Score Fusion:**
   - Combine Min-K%++ scores with multi-layer concentration features
   - Use adaptive weighting based on layer importance
   - Account for syntactic (early layers) vs. semantic (deep layers) patterns
   - Apply stability-based confidence weighting
   - Return final enhanced detection score

## Expected Benefits:

1. **Enhanced Discrimination**: Multi-layer analysis captures both syntactic and semantic memorization patterns
2. **Robustness**: Less sensitive to final layer biases and model-specific quirks
3. **Interpretability**: Layer-wise features provide insights into memorization mechanisms
4. **Generalizability**: Works across different transformer architectures
5. **Temporal Patterns**: Captures how confidence evolves through model depth

## Hyperparameters:

- Selected layers: [depth//4, depth//2, 3*depth//4] where depth is total layers
- Layer weights: [0.3, 0.4, 0.3] for early, middle, late layers respectively
- Top-k values: [5, 10, 20, 50] for concentration analysis
- Effective vocab threshold: 0.9 (90% probability mass)
- Base fusion alpha: 0.6 (balance between Min-K%++ and multi-layer features)
- Feature normalization ranges based on empirical observations

## Implementation Strategy:

1. ✅ Create multi-layer extraction functions for different architectures
2. ✅ Implement concentration metric calculations
3. ✅ Add layer-wise feature processing and normalization
4. ✅ Implement adaptive fusion with layer importance weighting
5. ✅ Add fallback mechanisms for models without intermediate layer access
6. ✅ Maintain compatibility with existing evaluation framework
7. ✅ Add error handling for memory and computational constraints

## Technical Details:

- **Memory Management**: Process layers sequentially to avoid memory issues
- **Architecture Handling**: Different approaches for encoder-decoder vs. decoder-only models
- **Computational Efficiency**: Cache intermediate computations where possible
- **Numerical Stability**: Proper handling of log operations and division by zero
- **Fallback Mechanism**: Graceful degradation to final layer analysis if intermediate access fails

## Architecture-Specific Considerations:

### For Pythia models:
- Use transformer blocks with attention and MLP layers
- Extract from transformer.h[layer_idx] hidden states
- Apply language model head (lm_head) for vocabulary projection

### For Mamba models:
- Use mamba layers with selective state space models
- Extract from backbone.layers[layer_idx] outputs
- Apply lm_head for vocabulary projection
- Handle potential differences in layer organization

This approach represents a significant advancement by leveraging the full depth of transformer models to capture multi-layered patterns of memorization, leading to more robust and accurate pre-training data detection.