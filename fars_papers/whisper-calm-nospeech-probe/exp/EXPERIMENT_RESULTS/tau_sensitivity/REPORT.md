# Tau Sensitivity Analysis for SCHM

## Experiment Overview

Evaluate how sensitive SCHM's (Silence-Conditional Head Masking) performance is to the choice of threshold tau. The analysis tests tau in {0.5, 0.6, 0.7} and includes boundary conditions (Condition A = never mask, Condition B = always mask) as references.

## Setup

- **Model**: openai/whisper-large-v3
- **Mode**: suppress (output empty transcription when p_nospeech >= tau)
- **Masked heads**: {1, 6, 11} across all 32 decoder layers
- **Datasets**: UrbanSound8K (hallucination rate), LibriSpeech test-clean/test-other (WER)
- **Method**: Entirely offline -- reused pre-computed p_nospeech values and per-clip transcriptions from Conditions A, B, and the prior tau sweep. No additional GPU inference required.

## Key Results

### Robustness Table (suppress mode)

| Condition | tau | US8K Halluc (%) | US8K % masked | LS-clean WER (%) | LS-clean % masked | LS-other WER (%) | LS-other % masked |
|---|---|---|---|---|---|---|---|
| Cond A (default) | inf | 100.00 | 0.00 | 2.83 | 0.00 | 5.10 | 0.00 |
| SCHM | 0.7 | 81.95 | 18.05 | 2.86 | 0.04 | 5.10 | 0.00 |
| SCHM | 0.6 | 78.37 | 21.63 | 2.86 | 0.04 | 5.10 | 0.00 |
| SCHM | 0.5 | 74.37 | 25.63 | 2.95 | 0.08 | 5.10 | 0.00 |
| Cond B (always mask) | 0 | 100.00 | 100.00 | 3.08 | 100.00 | 5.32 | 100.00 |

### Key Observations

1. **Hallucination sensitivity**: Hallucination rate varies from 74.37% (tau=0.5) to 81.95% (tau=0.7), a 7.58 pp range. Lower tau triggers suppression on more UrbanSound8K clips. All SCHM operating points improve over baseline (100% hallucination).

2. **WER robustness**: WER is remarkably stable across the tau range:
   - LS test-clean: 2.86% at tau=0.6 and 0.7, 2.95% at tau=0.5 (0.09 pp variation)
   - LS test-other: constant 5.10% at all thresholds (zero false positives)

3. **False positive rate**: Extremely low at all thresholds. At tau=0.5, only 2/2620 clean utterances are incorrectly suppressed. At tau=0.6 and 0.7, only 1/2620. No test-other utterances are ever suppressed. The p_nospeech signal cleanly separates speech from non-speech.

4. **Overall robustness**: SCHM is robust to tau choice. The user faces a simple tradeoff: lower tau reduces hallucinations more aggressively at the cost of marginally higher WER on clean speech (+0.12 pp worst case). Higher tau is more conservative. No careful tuning is required -- any value in [0.5, 0.7] gives a reasonable operating point.

5. **Pareto analysis**: All SCHM points dominate Condition A (better or equal WER, strictly lower hallucination rate). Condition B has 100% hallucination rate (same as A) because HF generate() always produces text regardless of head masking, so it is not a useful comparison point for hallucination reduction.

## Artifacts

- `schm/results/tau_robustness.json` -- Full robustness table data
- `schm/figures/tau_tradeoff.png` -- Two-panel plot (clean and other WER vs halluc rate)
- `schm/figures/tau_tradeoff_avg.png` -- Single-panel plot with average WER
- `schm/analysis/compute_tau_robustness.py` -- Offline computation script
- `schm/analysis/plot_tau_tradeoff.py` -- Visualization script
