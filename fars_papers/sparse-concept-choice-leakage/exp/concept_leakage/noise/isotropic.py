"""Isotropic (generalized Laplace) noise sampler for embedding sanitization.
Samples noise eta such that f(eta) proportional to exp(-eps * ||eta||_2).
Algorithm: eta = Y * X, where X = N(0,I)/||N|| (uniform on sphere), Y ~ Gamma(d, 1/eps)."""

import numpy as np


class IsotropicNoiseSampler:
    def __init__(self, dim: int = 768, epsilon: float = 10.0, rng: np.random.Generator | None = None):
        self.dim = dim
        self.epsilon = epsilon
        self.rng = rng or np.random.default_rng()

    def sample(self, n: int = 1) -> np.ndarray:
        normal = self.rng.standard_normal((n, self.dim))
        norms = np.linalg.norm(normal, axis=1, keepdims=True)
        directions = normal / norms
        scales = self.rng.gamma(shape=self.dim, scale=1.0 / self.epsilon, size=(n, 1))
        return (scales * directions).astype(np.float32)

    def sample_single(self) -> np.ndarray:
        return self.sample(1)[0]


def verify_isotropic(epsilon: float = 10.0, dim: int = 768, n_samples: int = 50_000):
    """Verify per-dimension variance is approximately uniform across all dimensions."""
    rng = np.random.default_rng(42)
    sampler = IsotropicNoiseSampler(dim=dim, epsilon=epsilon, rng=rng)
    noise = sampler.sample(n_samples)
    per_dim_var = np.var(noise, axis=0)

    mean_var = np.mean(per_dim_var)
    std_var = np.std(per_dim_var)
    max_var = np.max(per_dim_var)
    min_var = np.min(per_dim_var)
    cv = std_var / mean_var

    print(f"Isotropic noise verification (eps={epsilon}, d={dim}, n={n_samples}):")
    print(f"  Per-dim variance: mean={mean_var:.6f}, std={std_var:.6f}")
    print(f"  Range: [{min_var:.6f}, {max_var:.6f}]")
    print(f"  Coefficient of variation: {cv:.4f}")
    print(f"  Expected per-dim variance ~ d/eps^2 = {dim / epsilon**2:.6f}")

    assert cv < 0.10, f"Variance too non-uniform: CV={cv:.4f}"
    print("  PASSED: per-dimension variances are approximately uniform.")
    return per_dim_var


if __name__ == "__main__":
    verify_isotropic()
