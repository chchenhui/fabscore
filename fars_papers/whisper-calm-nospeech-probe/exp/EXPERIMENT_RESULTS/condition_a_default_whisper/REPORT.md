# Condition A: Default Whisper-large-v3 Baseline

## Experiment Overview
Evaluate unmodified Whisper-large-v3 with standard greedy decoding on UrbanSound8K (non-speech hallucination rate) and LibriSpeech test-clean/test-other (WER). This establishes the Condition A baseline for comparison with Conditions B (Always-Mask) and C (SCHM).

## Setup
- **Model**: `openai/whisper-large-v3` (HuggingFace Transformers, float16)
- **Decoding**: Greedy (num_beams=1, do_sample=False, language="en", task="transcribe")
- **max_new_tokens**: 128
- **Batch size**: 16
- **Hardware**: 1x GPU via TrainService
- **Datasets**:
  - UrbanSound8K: 8,732 non-speech clips (danavery/urbansound8K from HuggingFace, resampled to 16kHz)
  - LibriSpeech test-clean: 2,620 utterances
  - LibriSpeech test-other: 2,939 utterances

## Key Results

| Metric | Our Result | Calm-Whisper Cited | Delta |
|--------|-----------|-------------------|-------|
| UrbanSound8K hallucination rate | **100.00%** | 99.97% | +0.03% |
| LibriSpeech test-clean WER | **2.83%** | 2.12% | +0.71% |
| LibriSpeech test-other WER | **5.10%** | 4.07% | +1.03% |

## Key Observations
1. **Hallucination rate** is essentially identical to Calm-Whisper's reported value (100% vs 99.97%). Whisper hallucinates on virtually all non-speech audio clips.
2. **WER is slightly higher** than Calm-Whisper's cited values. Possible causes:
   - We explicitly set `language="en"` and `task="transcribe"`, which changes the forced decoder IDs compared to auto-detection mode.
   - Text normalization differences (we use lowercase + strip punctuation via jiwer).
   - Different `transformers` library version.
3. **For fair comparison** across Conditions A/B/C, the exact same codepath and decoding config will be used, so relative differences remain valid regardless of absolute WER.
4. **Runtime**: UrbanSound8K ~924s, LibriSpeech ~5 min total. Total wall time ~25 min on 1 GPU.

## Result Files
- `schm/results/condition_a_urbansound8k.json` - Per-clip transcriptions (8,732 clips)
- `schm/results/condition_a_librispeech_test_clean.json` - Per-utterance results (2,620 utts)
- `schm/results/condition_a_librispeech_test_other.json` - Per-utterance results (2,939 utts)
- `schm/results/condition_a_summary.json` - Aggregate metrics
