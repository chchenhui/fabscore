"""Multi-release fingerprint computation for concept-choice leakage attack.
For each document embedding, samples N independent noise vectors, computes
element-wise squared differences across all N*(N-1)/2 pairs, groups documents
into synthetic users, and averages to produce normalized fingerprints.
Adapts group parameters to available data: uses replacement sampling when needed."""

import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concept_leakage.noise.isotropic import IsotropicNoiseSampler

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
G = 50
M = 200
DIM = 768
EPSILON = 10.0

BASE_DIR = Path(__file__).resolve().parent.parent
EMB_DIR = BASE_DIR / "outputs" / "embeddings"
FP_DIR = BASE_DIR / "outputs" / "fingerprints"


def compute_fingerprints(
    embeddings: np.ndarray,
    sampler,
    n_groups: int = G,
    group_size: int = M,
    n_releases: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized group fingerprints from multi-release embedding differences.
    With N releases, averages delta^2 over all N*(N-1)/2 pairs for lower variance."""
    rng = rng or np.random.default_rng()
    n_docs = embeddings.shape[0]
    dim = embeddings.shape[1]

    needed = n_groups * group_size
    use_replacement = needed > n_docs

    if use_replacement:
        indices = rng.integers(0, n_docs, size=(n_groups, group_size))
    else:
        flat = rng.permutation(n_docs)[:needed]
        indices = flat.reshape(n_groups, group_size)

    fingerprints = np.zeros((n_groups, dim), dtype=np.float64)

    for g in range(n_groups):
        group_emb = embeddings[indices[g]]
        releases = [sampler.sample(group_size) for _ in range(n_releases)]
        pair_sum = np.zeros(dim, dtype=np.float64)
        n_pairs = 0
        for i in range(n_releases):
            for j in range(i + 1, n_releases):
                delta = (group_emb + releases[i]) - (group_emb + releases[j])
                pair_sum += np.mean(delta * delta, axis=0)
                n_pairs += 1
        v = pair_sum / n_pairs
        v_sum = np.sum(v)
        if v_sum > 0:
            fingerprints[g] = v / v_sum

    return fingerprints.astype(np.float32), indices


def run_fingerprint_computation(noise_type: str = "isotropic", seed: int = 42):
    rng = np.random.default_rng(seed)

    for concept in CONCEPTS:
        emb_path = EMB_DIR / concept / "embeddings.npy"
        if not emb_path.exists():
            print(f"WARNING: {emb_path} not found, skipping {concept}")
            continue

        embeddings = np.load(emb_path)
        n_docs = embeddings.shape[0]
        n_groups = G
        group_size = M

        if n_docs < n_groups * group_size:
            group_size = max(n_docs // n_groups, 1)
            if group_size < 5:
                group_size = min(n_docs, M)
                n_groups = max(n_docs // group_size, 1)
                n_groups = min(n_groups, G)

        print(f"\n{concept}: {n_docs} embeddings -> G={n_groups}, M={group_size} "
              f"(replacement={'yes' if n_groups * group_size > n_docs else 'no'})")

        sampler = IsotropicNoiseSampler(
            dim=DIM, epsilon=EPSILON,
            rng=np.random.default_rng(rng.integers(0, 2**31))
        )

        fingerprints, group_indices = compute_fingerprints(
            embeddings, sampler,
            n_groups=n_groups, group_size=group_size,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        out_dir = FP_DIR / noise_type / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"fingerprints_seed{seed}.npy", fingerprints)
        np.save(out_dir / f"group_indices_seed{seed}.npy", group_indices)

        meta = {"n_groups": n_groups, "group_size": group_size, "n_docs": n_docs}
        with open(out_dir / f"meta_seed{seed}.json", "w") as f:
            json.dump(meta, f)
        print(f"  Saved {fingerprints.shape} fingerprints to {out_dir}")


def main():
    for seed in [42, 123, 456]:
        print(f"\n=== Seed {seed} ===")
        run_fingerprint_computation(noise_type="isotropic", seed=seed)


if __name__ == "__main__":
    main()
