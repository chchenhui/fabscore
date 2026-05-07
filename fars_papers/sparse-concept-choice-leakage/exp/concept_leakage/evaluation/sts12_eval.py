"""STS12 utility evaluation under isotropic noise.
Embeds sentence pairs with gtr-t5-base, adds isotropic noise (eps=10),
computes cosine similarity, and reports Pearson correlation with gold scores."""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concept_leakage.noise.isotropic import IsotropicNoiseSampler

MODEL_NAME = "sentence-transformers/gtr-t5-base"
EPSILON = 10.0
DIM = 768
SEEDS = [42, 123, 456]
BATCH_SIZE = 256

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def cosine_similarity_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.sum(a_norm * b_norm, axis=1)


def main():
    import mteb
    tasks = mteb.get_tasks(tasks=["STS12"])
    task = list(tasks)[0]
    task.load_data()
    ds_test = task.dataset["test"]

    sent1 = [ex["sentence1"] for ex in ds_test]
    sent2 = [ex["sentence2"] for ex in ds_test]
    gold_scores = np.array([ex["score"] for ex in ds_test], dtype=np.float64)

    print(f"STS12 test set: {len(sent1)} pairs")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.half()

    print("Embedding sentence1...")
    emb1 = model.encode(sent1, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    emb1 = emb1.astype(np.float32)
    print("Embedding sentence2...")
    emb2 = model.encode(sent2, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    emb2 = emb2.astype(np.float32)

    clean_cos = cosine_similarity_batch(emb1, emb2)
    clean_pearson, _ = pearsonr(clean_cos, gold_scores)
    print(f"\nClean (no noise) Pearson: {clean_pearson:.4f}")

    seed_results = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sampler = IsotropicNoiseSampler(dim=DIM, epsilon=EPSILON, rng=rng)

        noisy_emb1 = emb1 + sampler.sample(len(emb1))
        noisy_emb2 = emb2 + sampler.sample(len(emb2))

        noisy_cos = cosine_similarity_batch(noisy_emb1, noisy_emb2)
        noisy_pearson, _ = pearsonr(noisy_cos, gold_scores)
        print(f"Seed {seed}: noisy Pearson = {noisy_pearson:.4f}")
        seed_results.append({"seed": seed, "pearson": float(noisy_pearson)})

    pearsons = [r["pearson"] for r in seed_results]
    summary = {
        "benchmark": "STS12",
        "noise_type": "isotropic",
        "epsilon": EPSILON,
        "n_pairs": len(sent1),
        "clean_pearson": float(clean_pearson),
        "per_seed": seed_results,
        "noisy_pearson_mean": float(np.mean(pearsons)),
        "noisy_pearson_std": float(np.std(pearsons)),
    }

    print(f"\n=== STS12 Summary ===")
    print(f"  Clean Pearson:  {summary['clean_pearson']:.4f}")
    print(f"  Noisy Pearson:  {summary['noisy_pearson_mean']:.4f} +/- {summary['noisy_pearson_std']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "isotropic_results.json"

    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)

    existing["sts12"] = summary

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
