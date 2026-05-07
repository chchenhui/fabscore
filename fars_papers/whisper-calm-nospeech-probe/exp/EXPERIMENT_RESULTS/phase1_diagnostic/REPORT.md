# Phase-1 Go/No-Go Diagnostic: p_no_speech Trigger Quality

## Experiment Overview

Tests whether the `p_no_speech` trigger signal fires reliably on hallucination cases in UrbanSound8K. This is the Phase-1 cheap diagnostic before investing compute in the full SCHM (Silence-Conditional Head Masking) evaluation.

**Revised Go/No-Go criterion**: There exists a tau in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8} where >= 20% of hallucinating clips have `p_no_speech > tau`.

## Setup

- **Model**: `openai/whisper-large-v3` (unmodified, fp16)
- **Dataset**: UrbanSound8K (8,732 clips of urban environmental sounds, no speech)
- **Method**: Single encoder + 1-step decoder forward pass per clip, extracting softmax probability at the `<|nospeech|>` token (ID 50363)
- **Batch size**: 16 (1x A100 GPU)
- **Hallucination definition**: Non-empty transcription from Condition A (default Whisper greedy decoding)
- **Validation**: Compared standalone computation with HF native WhisperNoSpeechDetection on 100 clips (correlation 0.999995)

## Key Results

### Trigger Sweep (all 8,732 hallucinating clips)

| tau | Clips Above | Fraction |
|-----|-------------|----------|
| 0.2 | 5,214 | 59.7% |
| 0.3 | 3,483 | 39.9% |
| 0.4 | 2,674 | 30.6% |
| 0.5 | 2,238 | 25.6% |
| 0.6 | 1,889 | 21.6% |
| 0.7 | 1,576 | 18.1% |
| 0.8 | 1,190 | 13.6% |

### Per-Class p_no_speech Summary

| Class | n | Mean p | >0.3 | >0.5 |
|-------|---|--------|------|------|
| gun_shot | 374 | 0.784 | 93.8% | 88.8% |
| jackhammer | 1000 | 0.527 | 63.3% | 47.5% |
| car_horn | 429 | 0.501 | 56.6% | 48.5% |
| drilling | 1000 | 0.467 | 62.3% | 35.7% |
| dog_bark | 1000 | 0.372 | 46.0% | 30.8% |
| engine_idling | 1000 | 0.365 | 43.5% | 26.4% |
| siren | 929 | 0.280 | 26.6% | 13.4% |
| air_conditioner | 1000 | 0.272 | 33.0% | 11.6% |
| children_playing | 1000 | 0.182 | 10.3% | 3.9% |
| street_music | 1000 | 0.167 | 5.8% | 1.5% |

### p_no_speech Distribution (all clips)

| Statistic | Value |
|-----------|-------|
| Mean | 0.3574 |
| Median | 0.2383 |
| Std | 0.2799 |
| Min | 0.0010 |
| Max | 0.9980 |
| P5 | 0.0703 |
| P25 | 0.1531 |
| P50 | 0.2383 |
| P75 | 0.5181 |
| P90 | 0.8648 |
| P95 | 0.9374 |

### Go/No-Go Decision: **GO**

At tau=0.3, 39.9% of hallucinating clips trigger (well above 20%). At tau=0.4, 30.6% trigger. The SCHM trigger signal fires on a meaningful fraction of clips at lower tau values, making the method viable for full evaluation.

## Key Observations

1. **100% hallucination rate**: All 8,732 UrbanSound8K clips produce non-empty transcriptions under Condition A. This is because our pipeline uses HuggingFace's `model.generate()` without the OpenAI Whisper pipeline's built-in no-speech suppression (`no_speech_threshold`).

2. **Class-dependent trigger effectiveness**: The trigger works strongly on non-speech-like sounds (gun_shot: 88.8% at tau=0.5, jackhammer: 47.5%, car_horn: 48.5%) but poorly on speech-like environmental sounds (children_playing: 3.9%, street_music: 1.5%). This explains the overall moderate trigger rate.

3. **Lower tau values are viable**: At tau=0.3, nearly 40% of clips trigger. While a lower tau also means some speech clips might be incorrectly masked (potentially hurting WER), the tradeoff can be evaluated empirically in the full SCHM experiment (Condition C).

4. **Computation validated**: Standalone 1-step forward pass matches HF's native WhisperNoSpeechDetection (correlation 0.999995, max |diff| < 0.002). The p_no_speech values are correct.

5. **Recommended SCHM operating point**: tau=0.3 or tau=0.4 for the full evaluation, rather than the originally proposed tau=0.6 which only triggers on 21.6% of clips.
