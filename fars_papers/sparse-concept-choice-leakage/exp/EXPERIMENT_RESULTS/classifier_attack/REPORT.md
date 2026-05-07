# MLP Classifier Attack on Concept-Choice Fingerprints

## Experiment Overview

This experiment evaluates concept-choice leakage under a more realistic attacker who does not have exact knowledge of the concept covariance templates. Instead of template matching (which assumes perfect knowledge of each concept's noise covariance), a lightweight MLP classifier is trained on fingerprint vectors to predict the concept label. The classifier is trained on a separate calibration set (validation-split fingerprints) to simulate an attacker with approximate but not exact template knowledge.

## Setup

**Training Data (Calibration Set)**:
- Source: Validation split embeddings (`val_dplus.npy`), separate from test split
- G_train = 30 groups per concept, M = 200 documents per group, N = 10 releases
- Replacement sampling used for all concepts (especially weekdays with only 21 docs)
- Total: 150 training fingerprints (30 per class, 5 classes), each 768-d

**MLP Architecture**:
- Input: 768-d normalized fingerprint
- Hidden layers: 256 (ReLU) -> 64 (ReLU)
- Output: 5 classes (softmax)
- Optimizer: Adam, lr=1e-3, batch_size=32
- Training: 5-fold stratified CV to select best epoch (out of 200 max)
- StandardScaler applied to fingerprints before training

**Test Data**:
- Condition A (anisotropic): G_test = 50 groups per concept, M = 200, N = 10 releases
- Condition C (smoothed, lambda=0.2): same parameters
- Total: 250 test fingerprints per condition per seed
- Evaluated across 3 seeds: [42, 123, 456]

**Baselines** (from Main Experiments):
- Template matching on Condition A: accuracy = 1.000 +/- 0.000
- Template matching on Condition C: accuracy = 1.000 +/- 0.000

## Key Results

| Condition | Attacker | Accuracy (mean +/- std) | Macro-F1 (mean +/- std) |
|-----------|----------|------------------------|------------------------|
| A (anisotropic) | Template Matching | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| A (anisotropic) | MLP Classifier | **1.000 +/- 0.000** | **1.000 +/- 0.000** |
| C (smoothed, lambda=0.2) | Template Matching | 1.000 +/- 0.000 | 1.000 +/- 0.000 |
| C (smoothed, lambda=0.2) | MLP Classifier | **0.959 +/- 0.007** | **0.959 +/- 0.007** |

**Training Details**:
- Best CV epoch: 9 (all 5 folds achieved 1.0 validation accuracy from early epochs)
- Final training accuracy: 1.000

## Key Observations

1. **Condition A**: The MLP classifier achieves perfect accuracy (1.000), matching the template-matching baseline. This confirms that concept-choice leakage is robust and can be exploited even without exact knowledge of the covariance templates.

2. **Condition C**: The MLP classifier still achieves 95.9% accuracy under smoothed covariance (lambda=0.2), far above the 20% chance level. While slightly lower than the template-matching result (1.000), this is expected since the classifier is trained on Condition A fingerprints and tested on Condition C fingerprints (domain shift). The high accuracy demonstrates that even an approximate attacker can exploit concept-choice leakage under covariance smoothing.

3. **Training efficiency**: The MLP converges extremely fast (best epoch = 9), indicating that the concept fingerprints are highly separable in the 768-d space. All 5 CV folds achieve perfect validation accuracy.

4. **Robustness of the leakage finding**: The near-perfect classifier performance validates that the concept-choice leakage finding is not an artifact of the idealized template-matching attack. A realistic attacker with only approximate calibration data can achieve comparable results.
