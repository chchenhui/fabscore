"""Embed D_plus test-split sentences for each concept using gtr-t5-base (768-d).
Saves numpy arrays of shape (N_docs, 768) under outputs/embeddings/<concept>/."""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
BATCH_SIZE = 256
MODEL_NAME = "sentence-transformers/gtr-t5-base"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs" / "data"
EMB_DIR = BASE_DIR / "outputs" / "embeddings"


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.half()

    for concept in CONCEPTS:
        test_path = DATA_DIR / concept / "test.json"
        if not test_path.exists():
            print(f"WARNING: {test_path} not found, skipping {concept}")
            continue

        with open(test_path) as f:
            records = json.load(f)

        sentences = [r["d_plus"] for r in records]
        print(f"\nEmbedding {concept}: {len(sentences)} sentences")

        embeddings = model.encode(
            sentences,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = embeddings.astype(np.float32)

        out_dir = EMB_DIR / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "embeddings.npy"
        np.save(out_path, embeddings)
        print(f"  Saved {embeddings.shape} to {out_path}")

    print("\nEmbedding generation complete.")


if __name__ == "__main__":
    main()
