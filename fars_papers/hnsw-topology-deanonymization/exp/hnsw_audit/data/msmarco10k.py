"""Download MS MARCO passages, sample 10K, and embed with sentence-transformers.

Uses microsoft/ms_marco v1.1 train split from HuggingFace. Extracts unique
passages, samples 10,000, embeds with msmarco-distilbert-base-v2 (d=768).
Caches vectors.npy and passage_ids.json to outputs/msmarco10k/.
"""

import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "msmarco10k")
VECTORS_PATH = os.path.join(OUTPUT_DIR, "vectors.npy")
PASSAGE_IDS_PATH = os.path.join(OUTPUT_DIR, "passage_ids.json")

N_SAMPLES = 10000
EMBED_MODEL = "sentence-transformers/msmarco-distilbert-base-v2"
EMBED_DIM = 768
BATCH_SIZE = 256


def load_msmarco10k(seed: int = 42) -> np.ndarray:
    if os.path.exists(VECTORS_PATH):
        print(f"Loading cached MSMARCO-10K embeddings from {VECTORS_PATH}")
        return np.load(VECTORS_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from datasets import load_dataset

    print("Downloading MS MARCO v1.1 train split...")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    print("Extracting unique passages...")
    passages = set()
    for row in ds:
        for text in row["passages"]["passage_text"]:
            passages.add(text)
    passages = sorted(passages)
    print(f"  Total unique passages: {len(passages)}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(passages), size=N_SAMPLES, replace=False)
    indices.sort()
    sampled_passages = [passages[i] for i in indices]
    sampled_ids = indices.tolist()

    with open(PASSAGE_IDS_PATH, "w") as f:
        json.dump(sampled_ids, f)
    print(f"Saved {len(sampled_ids)} passage IDs to {PASSAGE_IDS_PATH}")

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Encoding {N_SAMPLES} passages (batch_size={BATCH_SIZE})...")
    embeddings = model.encode(
        sampled_passages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = embeddings.astype(np.float32)

    assert embeddings.shape == (N_SAMPLES, EMBED_DIM), f"Unexpected shape: {embeddings.shape}"
    np.save(VECTORS_PATH, embeddings)
    print(f"Saved embeddings to {VECTORS_PATH}, shape={embeddings.shape}")

    return embeddings


if __name__ == "__main__":
    vecs = load_msmarco10k()
    print(f"MSMARCO-10K: shape={vecs.shape}, dtype={vecs.dtype}")
