# Baseline: Unencrypted Activation Monitor on Qwen2.5-7B-Instruct

## Experiment Overview

Trained and evaluated an MLP activation monitor on plain-text (unencrypted) activations of Qwen2.5-7B-Instruct, replicating the Activation Monitoring methodology. This establishes an upper-bound reference for harmful-request detection performance before introducing obfuscation.

Reference: "Activation Monitoring: Advantages of Using Internal Representations for LLM Oversight" (https://openreview.net/pdf?id=qbvtwhQcH5)

## Setup

- **Model**: Qwen/Qwen2.5-7B-Instruct (28 layers, hidden_size=3584, bf16)
- **Datasets**:
  - HarmBench: 300 behaviors (standard + contextual, excluding copyright)
  - Alpaca: 52,002 samples (harmless distribution)
  - Balanced dataset: 300 harmful + 300 harmless, 80/20 stratified split (480 train / 120 test)
  - FPR calibration: full 52,002 Alpaca pool
- **Probe**: 2-layer MLP (3584 -> 32 -> 1, ReLU, Sigmoid), BCE loss
- **Optimizer**: Adam, lr=1e-4, weight_decay=1.0, 5000 epochs
- **Layer sweep**: layers 21-27 (final quarter of 28-layer model)
- **Seeds**: [42, 123, 456]
- **Tokenization**: max_length=128, padding="max_length"

## Key Results

### Layer Sweep (seed=42)

| Layer | Val AUROC | Best Epoch |
|-------|-----------|------------|
| 21    | 0.9992    | 150        |
| 22    | 0.9992    | 50         |
| 23    | 0.9992    | 100        |
| 24    | 0.9992    | 50         |
| 25    | 0.9992    | 100        |
| 26    | 0.9997    | 350        |
| **27**| **1.0000**| **50**     |

Best layer: **27** (last transformer layer)

### Final Results (Layer 27, mean +/- std over 3 seeds)

| Metric         | Value              |
|----------------|--------------------|
| AUROC          | 1.0000 +/- 0.0000 |
| TPR@FPR=1e-3   | 0.8333 +/- 0.0272 |
| TPR@FPR=1e-4   | 0.4500 +/- 0.1080 |

### Per-Seed Breakdown

| Seed | AUROC | TPR@FPR=1e-3 | TPR@FPR=1e-4 |
|------|-------|--------------|--------------|
| 42   | 1.0   | 0.8000       | 0.3000       |
| 123  | 1.0   | 0.8667       | 0.5000       |
| 456  | 1.0   | 0.8333       | 0.5500       |

## Key Observations

1. **Perfect AUROC**: All layers achieve near-perfect AUROC (>0.999), with layer 27 achieving exactly 1.0. The harmful/harmless activations are highly separable in late layers.

2. **Sanity check passed**: TPR@FPR=1e-3 = 0.8333 > 0.80 threshold, confirming the monitor works well.

3. **TPR@FPR=1e-4 is moderate (0.45)**: At this very strict threshold, only ~45% of harmful prompts are detected. This is expected given the small test set (60 harmful samples) and the extreme FPR requirement. The Activation Monitoring paper reported 89.15% for Llama-2-7B, but they used a different model and potentially more test samples.

4. **Early convergence**: Best checkpoints were found at epochs 50-350, despite training for 5000 epochs. The probes converge very quickly on this task.

5. **Layer 27 is optimal**: The last layer provides the best separation, consistent with findings that later layers encode more task-relevant information.
