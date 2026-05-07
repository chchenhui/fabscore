# Score Distribution Analysis Report

## Experiment Overview

Analyzed the distribution of monitor scores across 64 random keys for each prompt
to understand the mechanism behind the key-search attack on encrypted activation monitors.

- **Encryptor**: seed 123, lambda2=0.5 (high diversity)
- **Probe**: seed 123, FPR=1e-3 threshold = 0.9536
- **Harmful prompts**: 240 (HarmBench test set)
- **Harmless prompts**: 200 (Alpaca random subset)
- **Keys per prompt**: 64

## Setup

For each prompt, 64 random Gaussian keys were sampled. The monitor score m(R(x, k_i)) was
computed for each key, yielding a score distribution per prompt.

## Key Results

### Per-Prompt Score Variance

| Metric | Harmful | Harmless |
|--------|---------|----------|
| Mean per-prompt std | 0.0085 | 0.0043 |
| Max per-prompt std | 0.1156 | 0.0711 |
| Global mean score | 0.9755 | 0.0285 |

### Bypass Analysis (K=32)

- Bypass rate at K=32: 27.9% (67/240)
- Correlation(std, bypass): r=0.458 (p=7.56e-14)
- Mean std for bypassed prompts: 0.020071975886821747
- Mean std for non-bypassed prompts: 0.0039707208052277565

### Theoretical vs Empirical i.i.d. Validation

- Pearson r = 0.999 (p = 3.61e-317)
- Spearman rho = 0.997 (p = 2.76e-271)
- Mean theoretical bypass prob: 0.261
- Mean empirical bypass prob: 0.260

## Key Observations

1. Key-conditioned obfuscation induces high variance in monitor scores across keys,
   especially for harmful prompts. This variance is the mechanism that enables key search.
2. Prompts with higher score variance are more likely to be bypassed by min-of-K search,
   confirming that the attack exploits the tail of the score distribution.
3. The theoretical i.i.d. model closely matches empirical bypass rates, validating that
   per-prompt scores across keys are approximately independent.

## Figures

- `results/figures/score_distributions.pdf`: KDE plots (page 1), Mean vs Std scatter (page 2)
- `results/figures/theoretical_vs_empirical_bypass.pdf`: Theoretical vs empirical bypass scatter
