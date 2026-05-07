"""Load LibriSpeech test-clean and test-other from HuggingFace (librispeech_asr).
Audio arrays are already at 16 kHz. test-clean has 2,620 utterances;
test-other has 2,939 utterances.
"""

import numpy as np
from datasets import load_dataset
from typing import Literal

DATASET_NAME = "librispeech_asr"
TARGET_SR = 16000


def load_librispeech(
    split: Literal["test.clean", "test.other"] = "test.clean",
    streaming: bool = False,
):
    """Load a LibriSpeech split and yield (utt_id, audio_16k, reference) tuples.

    Args:
        split: "test.clean" or "test.other".
        streaming: If True, use streaming mode.

    Yields:
        Tuple of (utt_id: str, audio: np.ndarray at 16 kHz, reference: str)
    """
    ds = load_dataset(DATASET_NAME, "all", split=split, streaming=streaming, trust_remote_code=True)

    for sample in ds:
        audio_info = sample["audio"]
        audio_array = np.array(audio_info["array"], dtype=np.float32)
        sr = audio_info["sampling_rate"]
        assert sr == TARGET_SR, f"Expected {TARGET_SR} Hz, got {sr} Hz"

        utt_id = str(sample["id"])
        reference = sample["text"]
        yield utt_id, audio_array, reference


def load_librispeech_all(split: Literal["test.clean", "test.other"] = "test.clean"):
    """Load entire LibriSpeech split into memory.

    Returns:
        List of (utt_id, audio_16k, reference) tuples.
    """
    return list(load_librispeech(split=split, streaming=False))
