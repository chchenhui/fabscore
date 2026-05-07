"""Covariance smoothing mitigation for concept-choice leakage.
Mixes concept-specific covariance with identity: Sigma_mix = (1-lam)*Sigma_Ck + lam*I,
then re-normalizes to trace d. Reduces anisotropy while preserving concept-aware protection."""

import argparse
import json
import numpy as np
from pathlib import Path

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
DIM = 768
MASK_SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "checkpoints_opt"


def smooth_covariance(sigma_diag: np.ndarray, lam: float = 0.2) -> np.ndarray:
    d = len(sigma_diag)
    identity = np.ones(d, dtype=np.float64)
    sigma_mix = (1.0 - lam) * sigma_diag.astype(np.float64) + lam * identity
    sigma_mix *= d / sigma_mix.sum()
    return sigma_mix


def smooth_all_concepts(ckpt_dir: Path = CKPT_DIR, mask_seed: int = MASK_SEED,
                        lam: float = 0.2):
    print(f"Covariance smoothing: lambda={lam}, d={DIM}")
    for concept in CONCEPTS:
        sigma_path = ckpt_dir / concept / f"seed{mask_seed}" / "sigma.npy"
        sigma = np.load(sigma_path)

        sigma_smoothed = smooth_covariance(sigma, lam=lam)

        out_path = ckpt_dir / concept / f"seed{mask_seed}" / f"sigma_smoothed_lam{lam:.2f}.npy"
        np.save(out_path, sigma_smoothed)
        legacy_path = ckpt_dir / concept / f"seed{mask_seed}" / "sigma_smoothed.npy"
        np.save(legacy_path, sigma_smoothed)

        orig_std = np.std(sigma)
        smooth_std = np.std(sigma_smoothed)
        print(f"  {concept}: orig_std={orig_std:.4f} -> smooth_std={smooth_std:.4f}, "
              f"trace={sigma_smoothed.sum():.1f}, "
              f"min={sigma_smoothed.min():.4f}, max={sigma_smoothed.max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=str(CKPT_DIR))
    parser.add_argument("--mask_seed", type=int, default=MASK_SEED)
    parser.add_argument("--lam", type=float, default=0.2)
    args = parser.parse_args()
    smooth_all_concepts(ckpt_dir=Path(args.ckpt_dir), mask_seed=args.mask_seed,
                        lam=args.lam)
