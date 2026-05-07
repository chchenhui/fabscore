# Encrypted Activation Monitor and Key-Search Attack

## Experiment Overview

Trained an MLP activation monitor on encrypted traffic (each prompt encrypted with a fresh random key) and evaluated the best-of-K key-search attack. The monitor uses the same architecture as the baseline (2-layer MLP probe, hidden_dim=32) but operates on activations from encrypted inputs. The key-search attack samples K keys per prompt and selects the key that minimizes the monitor score.

## Setup

- **Model**: Qwen2.5-7B-Instruct (frozen, bf16)
- **Encryptor**: High-diversity OSNIP-like MLP (65M params, seed 123, diversity margin=2.0, lambda2=0.5)
  - KL divergence: 0.031, KNN ASR@10: 0.526, Key diversity L2: 1.037
- **Monitor layer**: 27 (same as baseline)
- **Monitor architecture**: Linear(3584, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
- **Training**: BCE loss, Adam, lr=1e-4, weight_decay=1.0, 5000 epochs
- **Seeds**: [42, 123, 456]
- **Train set**: 480 samples (240 harmful HarmBench + 240 harmless Alpaca), encrypted with random keys
- **Test set**: 120 samples (60 harmful + 60 safe), encrypted with random keys
- **FPR calibration**: 52,002 Alpaca samples (encrypted, one random key each)
- **Key-search K values**: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
- **Attack test set**: 240 harmful HarmBench test prompts (fresh random keys per K evaluation)
- **Hardware**: 1x A100-80GB GPU

## Key Results

### Encrypted Monitor K=1 (Honest User) Performance

| Metric | Mean |
|--------|------|
| AUROC | 1.0000 |
| TPR@FPR=1e-3 | 0.8333 |
| TPR@FPR=1e-4 | 0.4722 |

### Key-Search Attack: TPR@FPR vs K (mean across 3 probe seeds)

| K | TPR@FPR=1e-3 | Drop from K=1 | TPR@FPR=1e-4 | Drop from K=1 |
|---|-------------|--------------|-------------|--------------|
| 1 | 0.8486 | -- | 0.5139 | -- |
| 2 | 0.8347 | 1.4pp | 0.4792 | 3.5pp |
| 4 | 0.8139 | 3.5pp | 0.4611 | 5.3pp |
| 8 | 0.7931 | 5.6pp | 0.4306 | 8.3pp |
| 16 | 0.7500 | 9.9pp | 0.3917 | 12.2pp |
| 32 | 0.6931 | 15.6pp | 0.3611 | 15.3pp |
| 64 | 0.5986 | **25.0pp** | 0.3069 | 20.7pp |
| 128 | 0.4708 | **37.8pp** | 0.2444 | 26.9pp |
| 256 | 0.3333 | **51.5pp** | 0.1528 | 36.1pp |
| 512 | 0.1625 | **68.6pp** | 0.0597 | 45.4pp |

### Reference Baselines

| Setting | TPR@FPR=1e-3 | TPR@FPR=1e-4 |
|---------|-------------|-------------|
| Unencrypted monitor (baseline) | 0.8333 | 0.4500 |
| Encrypted monitor K=1 | 0.8486 | 0.5139 |
| Encrypted monitor K=32 | 0.6931 | 0.3611 |
| Encrypted monitor K=64 | 0.5986 | 0.3069 |
| Encrypted monitor K=512 | 0.1625 | 0.0597 |

## Key Observations

1. **Encrypted monitor is effective for honest users**: At K=1, the monitor achieves AUROC=1.0 and TPR@FPR=1e-3 of 0.83-0.85, comparable to the unencrypted baseline (0.83). Encryption does not destroy harmful-intent signal at layer 27.

2. **Key-search attack dramatically bypasses the monitor**: The TPR drop from K=1 to K=64 is 25.0pp at FPR=1e-3, exceeding the 20pp target. At K=512, TPR drops by 68.6pp (from 0.85 to 0.16), rendering the monitor nearly useless.

3. **Key diversity is the critical factor**: The 4x increase in embedding-level key diversity (L2: 0.26 -> 1.04) is the primary driver. Higher diversity at the embedding level propagates to meaningfully different activations at layer 27, creating high variance in monitor scores across keys.

4. **Attack scales log-linearly with K**: TPR drops approximately linearly with log(K), suggesting consistent diminishing returns but continued effectiveness at higher K values.

5. **The hypothesis is strongly supported**: The central hypothesis -- that key-search bypasses encrypted activation monitors -- is confirmed. A computationally cheap best-of-K search (K=64-128 requires only 64-128 forward passes) can reduce the monitor's detection rate from ~85% to ~47-60%, fundamentally undermining the defense when the encryptor has sufficient key diversity.
