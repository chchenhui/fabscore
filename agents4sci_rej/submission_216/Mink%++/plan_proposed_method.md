# Implementation Plan: Distribution Shape Analysis for Pre-Training Data Detection

## Overview
This implementation enhances the baseline Min-K%++ method by incorporating distribution shape analysis features to improve pre-training data detection accuracy.

## Key Components

### 1. Distribution Shape Features
We will extract three key statistical features from the conditional categorical distribution:

- **Skewness**: Measures asymmetry of the probability distribution
- **Kurtosis**: Measures tail heaviness and peakedness of the distribution  
- **Entropy**: Measures uncertainty/randomness in the distribution

### 2. Integration Strategy
The shape analysis features will be combined with the existing Min-K%++ method using:
- Weighted combination of Min-K%++ score and shape features
- Feature concatenation followed by a simple linear combination

### 3. Implementation Details

#### Core Changes:
1. **New method**: `mink++_shape_{ratio}` - combines Min-K%++ with distribution shape analysis
2. **Feature extraction**: Add functions to compute skewness, kurtosis, and entropy from softmax probabilities
3. **Score combination**: Weighted average of Min-K%++ score and normalized shape features

#### Key Functions to Add:
- `compute_distribution_shape_features(probs)` - Extract shape statistics
- `combine_scores(mink_plus_score, shape_features)` - Combine scores with learned weights
- Modified scoring logic in main processing loop

### 4. Output Compatibility
- Maintains same input/output format as baseline.py
- Outputs `results.json` and `scores.pkl` with identical schema
- Entry script: `proposed_method.py` with same CLI arguments

### 5. Technical Approach

#### Shape Feature Computation:
```python
# From softmax probabilities at each position
probs = F.softmax(logits[0, :-1], dim=-1)

# Skewness: asymmetry measure
skewness = torch.mean(compute_skewness(probs))

# Kurtosis: tail heaviness measure  
kurtosis = torch.mean(compute_kurtosis(probs))

# Entropy: uncertainty measure
entropy = torch.mean(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
```

#### Score Combination:
```python
# Normalize shape features to [-1, 1] range
normalized_features = normalize_features([skewness, kurtosis, entropy])

# Weighted combination (weights can be learned/tuned)
shape_score = np.mean(normalized_features)
combined_score = alpha * mink_plus_score + (1 - alpha) * shape_score
```

### 6. Configuration
- Add new method `mink++_shape_{ratio}` to config
- Use same models and datasets as baseline
- Maintain backward compatibility with existing methods

## Expected Benefits
1. **Robustness**: Shape analysis provides complementary information to local maxima identification
2. **Accuracy**: Combined approach should improve AUROC and TPR scores
3. **Interpretability**: Shape features provide insights into distribution characteristics of training vs non-training data