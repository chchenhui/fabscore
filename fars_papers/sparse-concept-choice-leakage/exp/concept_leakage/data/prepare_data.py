"""Download and cache the PII-Masking-300K dataset and gtr-t5-base embedding model.
Verifies that the model produces 768-dimensional embeddings."""

import sys
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def download_dataset():
    print("Downloading PII-Masking-300K dataset...")
    ds = load_dataset("ai4privacy/pii-masking-300k")
    print(f"Dataset loaded. Splits: {list(ds.keys())}")
    for split in ds:
        print(f"  {split}: {len(ds[split])} examples")
    return ds


def download_model():
    print("Downloading gtr-t5-base embedding model...")
    model = SentenceTransformer("sentence-transformers/gtr-t5-base")
    print("Model loaded successfully.")
    return model


def verify_embedding_dim(model, expected_dim=768):
    test_sentence = "The quick brown fox jumps over the lazy dog."
    embedding = model.encode([test_sentence])
    actual_dim = embedding.shape[1]
    print(f"Test embedding shape: {embedding.shape}")
    assert actual_dim == expected_dim, (
        f"Expected {expected_dim}-d embeddings, got {actual_dim}-d"
    )
    print(f"Verified: model produces {expected_dim}-d embeddings.")
    return embedding


def main():
    ds = download_dataset()
    model = download_model()
    verify_embedding_dim(model, expected_dim=768)
    print("\nAll assets downloaded and verified successfully.")
    return ds, model


if __name__ == "__main__":
    main()
