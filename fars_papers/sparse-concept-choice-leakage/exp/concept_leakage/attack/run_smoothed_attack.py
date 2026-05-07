"""Multi-release fingerprint attack under smoothed covariance noise.
Loads sigma_smoothed.npy (covariance-smoothed), computes group fingerprints
with N releases, and runs template matching. Supports configurable lambda
and n_releases for parameter sweeps."""

import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concept_leakage.noise.mahalanobis import MahalanobisNoiseSampler
from concept_leakage.attack.fingerprint import compute_fingerprints
from concept_leakage.attack.template_match import predict_concepts

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
K = len(CONCEPTS)
DIM = 768
EPSILON = 10.0
G = 50
M = 200
N_RELEASES = 10
SEEDS = [42, 123, 456]
MASK_SEED = 42
DEFAULT_LAM = 0.2

BASE_DIR = Path(__file__).resolve().parent.parent
EMB_DIR = BASE_DIR / "outputs" / "embeddings"
CKPT_DIR = BASE_DIR / "checkpoints_opt"
FP_DIR = BASE_DIR / "outputs" / "fingerprints"
RESULTS_DIR = BASE_DIR / "results"


def load_sigma_smoothed(concept: str, ckpt_dir: Path = CKPT_DIR,
                        mask_seed: int = MASK_SEED, lam: float = DEFAULT_LAM) -> np.ndarray:
    fname = f"sigma_smoothed_lam{lam:.2f}.npy"
    p = ckpt_dir / concept / f"seed{mask_seed}" / fname
    if p.exists():
        return np.load(p)
    return np.load(ckpt_dir / concept / f"seed{mask_seed}" / "sigma_smoothed.npy")


def build_smoothed_templates(ckpt_dir: Path = CKPT_DIR,
                             mask_seed: int = MASK_SEED,
                             lam: float = DEFAULT_LAM) -> np.ndarray:
    templates = np.zeros((K, DIM), dtype=np.float32)
    for k, concept in enumerate(CONCEPTS):
        sigma = load_sigma_smoothed(concept, ckpt_dir, mask_seed, lam)
        templates[k] = sigma / sigma.sum()
    return templates


def main(mask_seed: int = MASK_SEED, n_releases: int = N_RELEASES,
         ckpt_dir: Path = CKPT_DIR, results_dir: Path = RESULTS_DIR,
         fp_subdir: str = "smoothed", group_size_min: int = 30,
         lam: float = DEFAULT_LAM):
    templates = build_smoothed_templates(ckpt_dir, mask_seed, lam)

    all_seed_results = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        all_true = []
        all_pred = []

        for k, concept in enumerate(CONCEPTS):
            emb_path = EMB_DIR / concept / "embeddings.npy"
            if not emb_path.exists():
                print(f"WARNING: {emb_path} not found, skipping {concept}")
                continue

            embeddings = np.load(emb_path)
            n_docs = embeddings.shape[0]

            n_groups = G
            group_size = max(group_size_min, min(M, n_docs))
            if n_docs < 5:
                n_groups = 1
                group_size = n_docs
            elif n_docs < n_groups * group_size:
                n_groups = max(n_docs // group_size, 1)
                n_groups = min(n_groups, G)

            sigma = load_sigma_smoothed(concept, ckpt_dir, mask_seed, lam)
            sampler = MahalanobisNoiseSampler(
                sigma_diag=sigma, epsilon=EPSILON,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
            )

            fingerprints, group_indices = compute_fingerprints(
                embeddings, sampler,
                n_groups=n_groups, group_size=group_size,
                n_releases=n_releases,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
            )

            out_dir = FP_DIR / fp_subdir / concept
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"fingerprints_seed{seed}.npy", fingerprints)
            np.save(out_dir / f"group_indices_seed{seed}.npy", group_indices)
            meta = {"n_groups": n_groups, "group_size": group_size,
                    "n_docs": n_docs, "n_releases": n_releases}
            with open(out_dir / f"meta_seed{seed}.json", "w") as f:
                json.dump(meta, f)

            print(f"  {concept} (seed={seed}): {n_docs} docs, G={n_groups}, "
                  f"M={group_size}, N={n_releases}")

            preds = predict_concepts(fingerprints, templates,
                                     rng=np.random.default_rng(rng.integers(0, 2**31)))
            true_labels = np.full(n_groups, k, dtype=np.int64)

            all_true.append(true_labels)
            all_pred.append(preds)

        if not all_true:
            print(f"No data for seed {seed}")
            continue

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print(f"Seed {seed}: N={len(y_true)}, accuracy={acc:.4f}, macro-F1={f1:.4f}")
        all_seed_results.append({"seed": seed, "n_samples": int(len(y_true)),
                                  "accuracy": float(acc), "macro_f1": float(f1)})

    accs = [r["accuracy"] for r in all_seed_results]
    f1s = [r["macro_f1"] for r in all_seed_results]

    summary = {
        "noise_type": "smoothed",
        "smoothing_lambda": lam,
        "mask_seed": mask_seed,
        "n_releases": n_releases,
        "n_concepts": K,
        "seeds": SEEDS,
        "per_seed": all_seed_results,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "chance_level": 1.0 / K,
    }

    print(f"\n=== Summary (smoothed, lambda={lam}, mask_seed={mask_seed}, N={n_releases}) ===")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Macro-F1: {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"  Chance:   {summary['chance_level']:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"smoothed_attack_lam{lam:.2f}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAttack results saved to {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_seed", type=int, default=MASK_SEED)
    parser.add_argument("--n_releases", type=int, default=N_RELEASES)
    parser.add_argument("--ckpt_dir", type=str, default=str(CKPT_DIR))
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--fp_subdir", type=str, default="smoothed")
    parser.add_argument("--group_size_min", type=int, default=30)
    parser.add_argument("--lam", type=float, default=DEFAULT_LAM)
    args = parser.parse_args()
    main(
        mask_seed=args.mask_seed, n_releases=args.n_releases,
        ckpt_dir=Path(args.ckpt_dir), results_dir=Path(args.results_dir),
        fp_subdir=args.fp_subdir, group_size_min=args.group_size_min,
        lam=args.lam,
    )
