"""Mahalanobis (anisotropic) noise sampler for SPARSE-style embedding sanitization.
SPARSE Algorithm 1: f(z) proportional to exp(-eps * ||z||_M).
For diagonal Sigma = diag(sigma_diag), the square root is element-wise sqrt.
Z = Y * Sigma^{1/2} * X, where X uniform on sphere, Y ~ Gamma(d, 1/eps)."""

import numpy as np
from scipy.stats import pearsonr


class MahalanobisNoiseSampler:
    def __init__(self, sigma_diag: np.ndarray, epsilon: float = 10.0,
                 rng: np.random.Generator | None = None):
        self.sigma_diag = sigma_diag.astype(np.float64)
        self.dim = len(sigma_diag)
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()
        self.sigma_sqrt = np.sqrt(self.sigma_diag)

    def sample(self, n: int = 1) -> np.ndarray:
        normal = self.rng.standard_normal((n, self.dim))
        norms = np.linalg.norm(normal, axis=1, keepdims=True)
        directions = normal / norms
        scales = self.rng.gamma(shape=self.dim, scale=1.0 / self.epsilon, size=(n, 1))
        return (scales * self.sigma_sqrt[None, :] * directions).astype(np.float32)

    def sample_single(self) -> np.ndarray:
        return self.sample(1)[0]


def verify_mahalanobis(sigma_diag: np.ndarray, epsilon: float = 10.0,
                       n_samples: int = 10_000):
    rng = np.random.default_rng(42)
    sampler = MahalanobisNoiseSampler(sigma_diag=sigma_diag, epsilon=epsilon, rng=rng)
    noise = sampler.sample(n_samples)
    per_dim_var = np.var(noise, axis=0)

    corr, pval = pearsonr(per_dim_var, sigma_diag)
    print(f"Mahalanobis verification (eps={epsilon}, d={len(sigma_diag)}, n={n_samples}):")
    print(f"  Per-dim variance range: [{per_dim_var.min():.4f}, {per_dim_var.max():.4f}]")
    print(f"  Sigma_diag range: [{sigma_diag.min():.4f}, {sigma_diag.max():.4f}]")
    print(f"  Pearson(empirical_var, sigma_diag) = {corr:.4f} (p={pval:.2e})")
    assert corr > 0.9, f"Correlation too low: {corr:.4f}"
    print("  PASSED: empirical variance tracks sigma_diag.")
    return per_dim_var, corr


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sigma = rng.exponential(1.0, size=768)
    sigma = sigma * (768.0 / sigma.sum())
    verify_mahalanobis(sigma + 1e-6)
