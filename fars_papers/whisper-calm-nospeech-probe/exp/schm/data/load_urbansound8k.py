"""Load UrbanSound8K from HuggingFace (danavery/urbansound8K).
Provides iterator and full-load functions that return 16 kHz audio arrays
ready for Whisper feature extraction. The dataset contains 8,732 clips of
urban environmental sounds (no speech).
"""

import librosa
import numpy as np
from datasets import load_dataset

DATASET_NAME = "danavery/urbansound8K"
TARGET_SR = 16000


def _resample_if_needed(audio_array: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr != TARGET_SR:
        audio_array = librosa.resample(
            audio_array.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SR
        )
    return audio_array.astype(np.float32)


def load_urbansound8k(streaming: bool = False):
    """Load UrbanSound8K and yield (clip_id, audio_array_16k, metadata) tuples.

    Args:
        streaming: If True, use streaming mode to avoid downloading all data upfront.

    Yields:
        Tuple of (clip_id: str, audio: np.ndarray at 16 kHz, meta: dict)
        where meta contains 'fold', 'classID', 'class' when available.
    """
    ds = load_dataset(DATASET_NAME, split="train", streaming=streaming, trust_remote_code=True)

    for idx, sample in enumerate(ds):
        audio_info = sample["audio"]
        audio_array = np.array(audio_info["array"], dtype=np.float32)
        orig_sr = audio_info["sampling_rate"]

        audio_16k = _resample_if_needed(audio_array, orig_sr)

        clip_id = sample.get("slice_file_name", f"clip_{idx}")
        meta = {
            "fold": sample.get("fold", None),
            "classID": sample.get("classID", None),
            "class": sample.get("class", None),
        }
        yield clip_id, audio_16k, meta


def load_urbansound8k_all():
    """Load entire UrbanSound8K into memory as a list.

    Returns:
        List of (clip_id, audio_array_16k, meta) tuples.
    """
    return list(load_urbansound8k(streaming=False))
