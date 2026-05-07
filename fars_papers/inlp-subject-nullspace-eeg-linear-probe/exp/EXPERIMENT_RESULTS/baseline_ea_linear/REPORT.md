# EA + Frozen CBraMod + Linear Head Baseline on BNCI2014001

## Experiment Overview

Establish the primary frozen-encoder baseline: apply Euclidean Alignment (EA) to raw EEG, extract frozen CBraMod embeddings, and train a linear classification head (logistic regression) under the LOSO protocol on BNCI2014001 (9 subjects, 4-class motor imagery).

## Setup

- **Dataset**: BNCI2014001 (BCI Competition IV-2a), 9 subjects, 4-class motor imagery, 22 EEG channels
- **Preprocessing**: Bandpass 0.3-50Hz at 250Hz, resample to 200Hz, epoch 2-6s MI window (800 samples), Euclidean Alignment per subject
- **Model**: Frozen CBraMod (pretrained, proj_out = Identity), embedding shape (batch, 22, 4, 200)
- **Pooling strategies**: avgpool (mean over channels and patches -> 200-dim) and flatten (-> 17600-dim)
- **Classifier**: LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
- **Evaluation**: 9-fold LOSO, balanced accuracy, 3 seeds [42, 123, 456]
- **Note**: LogisticRegression with lbfgs solver is deterministic -- results are identical across seeds

## Key Results

| Pooling Strategy | Mean LOSO Acc | Std (across folds) | vs Vanilla (41.45%) |
|---|---|---|---|
| **flatten (17600-dim)** | **56.27%** | 8.09% | **+14.82 pp** |
| avgpool (200-dim) | 33.54% | 6.47% | -7.91 pp |

### Per-Subject Accuracy (Best: flatten)

| Subject | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 60.42% | 40.10% | 62.50% | 50.00% | 51.39% | 53.99% | 63.89% | 64.58% | 59.55% | 56.27% |

## Key Observations

1. **Flatten >> avgpool**: The flatten strategy (17600-dim) dramatically outperforms avgpool (200-dim) by ~23 pp. This indicates that spatial (channel) and temporal (patch) structure in the embeddings carries important discriminative information that is lost by averaging.

2. **EA + flatten exceeds vanilla linear probing**: 56.27% vs reported 41.45% (+14.82 pp). EA provides substantial benefit, and using the full embedding dimensionality is critical.

3. **EA + flatten approaches fine-tuning**: The reported full fine-tuning result is 53.03%. Our frozen EA + flatten baseline at 56.27% actually exceeds this, suggesting that the high-dimensional linear probe on aligned embeddings is very effective.

4. **Deterministic results**: lbfgs solver produces identical results across seeds (std across seeds = 0). The reported std of 8.09% is across the 9 LOSO folds, reflecting inter-subject variability.

5. **Subject variability**: Subject 2 is consistently the hardest (40.10% with flatten, 24.65% with avgpool). Subjects 7 and 8 are easiest (>64%).
