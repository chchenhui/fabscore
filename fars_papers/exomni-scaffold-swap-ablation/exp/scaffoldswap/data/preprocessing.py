"""Shared prosody feature extraction: F0 + energy at 50 Hz (20ms hop).

Used by all three conditions (A/B/C). Computes frame-level F0 via librosa.pyin
and RMS energy. Both are per-utterance normalized (zero-mean, unit-variance).
"""
import numpy as np
import librosa


def extract_prosody_features(audio, sr=16000, hop_length=320, fmin=75, fmax=500):
    """Extract F0 and energy at 50 Hz (matching WavLM/HuBERT native rate).

    Args:
        audio: (n_samples,) waveform at sr Hz
        sr: sample rate (default 16000)
        hop_length: 320 samples = 20ms at 16kHz = 50 Hz
        fmin/fmax: pitch range for pyin

    Returns:
        prosody: (T, 2) float32 array with [F0, energy] per frame, normalized
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)

    energy = librosa.feature.rms(
        y=audio, hop_length=hop_length, frame_length=hop_length * 2
    )[0].astype(np.float32)

    min_len = min(len(f0), len(energy))
    f0 = f0[:min_len]
    energy = energy[:min_len]

    f0 = _normalize(f0)
    energy = _normalize(energy)

    prosody = np.stack([f0, energy], axis=-1)  # (T, 2)
    return prosody


def _normalize(x):
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std
