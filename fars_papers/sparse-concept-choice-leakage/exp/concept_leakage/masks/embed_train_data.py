"""Pre-embed D_plus and D_minus sentences for train/val splits using gtr-t5-base.
Saves numpy arrays for mask training (avoids re-embedding each epoch)."""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
MODEL_NAME = "sentence-transformers/gtr-t5-base"
BATCH_SIZE = 256

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "outputs" / "data"
EMB_DIR = BASE_DIR / "outputs" / "embeddings"


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.half()

    for concept in CONCEPTS:
        for split in ["train", "val"]:
            data_path = DATA_DIR / concept / f"{split}.json"
            if not data_path.exists():
                print(f"WARNING: {data_path} not found, skipping")
                continue

            with open(data_path) as f:
                records = json.load(f)

            out_dir = EMB_DIR / concept
            out_dir.mkdir(parents=True, exist_ok=True)

            for label_key, suffix in [("d_plus", "dplus"), ("d_minus", "dminus")]:
                out_path = out_dir / f"{split}_{suffix}.npy"
                if out_path.exists():
                    existing = np.load(out_path)
                    if existing.shape[0] == len(records):
                        print(f"  {concept}/{split}_{suffix}: already exists ({existing.shape}), skipping")
                        continue

                sentences = [r[label_key] for r in records]
                print(f"  Embedding {concept}/{split}_{suffix}: {len(sentences)} sentences")
                embs = model.encode(
                    sentences, batch_size=BATCH_SIZE,
                    show_progress_bar=True, convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                embs = embs.astype(np.float32)
                np.save(out_path, embs)
                print(f"    Saved {embs.shape} to {out_path}")

    print("\nTrain/val embedding complete.")


if __name__ == "__main__":
    main()
