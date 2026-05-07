# Isotropic correlated Gaussian noise perturbations (GP-1200km baseline).
# Generates spatially correlated noise on the sphere using a Gaussian covariance
# C(d) = sigma^2 * exp(-d^2 / (2*L^2)) with L = 1200 km decorrelation length.
# The Legendre spectral representation: C_l^GP ~ exp(-l(l+1)*(L/R)^2)
# where R = 6371 km. Perturbations are sampled in SH space and variance-matched
# to the calibration residual variance.

import numpy as np
import pyshtools as sh

EARTH_RADIUS_KM = 6371.0


class IsotropicGPSampler:

    def __init__(self, length_scale_km=1200, lmax=None):
        self.length_scale_km = length_scale_km
        self.lmax = lmax if lmax is not None else 359
        L_over_R = self.length_scale_km / EARTH_RADIUS_KM
        ells = np.arange(self.lmax + 1, dtype=np.float64)
        self.C_l_gp = np.exp(-ells * (ells + 1) * L_over_R ** 2)

    def sample_perturbations(self, n_members, lmax=None, seed=0):
        if lmax is None:
            lmax = self.lmax
        rng = np.random.default_rng(seed)
        sqrt_cl = np.sqrt(self.C_l_gp[:lmax + 1])
        mask = np.zeros((2, lmax + 1, lmax + 1), dtype=bool)
        for l in range(lmax + 1):
            mask[0, l, 0] = True
            for m in range(1, l + 1):
                mask[0, l, m] = True
                mask[1, l, m] = True
        scale = np.zeros((2, lmax + 1, lmax + 1))
        for l in range(lmax + 1):
            scale[:, l, :l + 1] = sqrt_cl[l]
        scale[~mask] = 0.0
        n_active = int(mask.sum())
        perturbations = []
        for j in range(n_members):
            noise = rng.standard_normal((2, lmax + 1, lmax + 1))
            coeffs = noise * scale
            coeffs[~mask] = 0.0
            grid_data = sh.expand.MakeGridDH(coeffs, sampling=2, extend=0, norm=1)
            perturbations.append(grid_data)
        result = np.stack(perturbations, axis=0)
        result = result[:, ::-1, :]
        south_pole = result[:, 0:1, :].mean(axis=2, keepdims=True).repeat(result.shape[2], axis=2)
        result = np.concatenate([south_pole, result], axis=1)
        return result

    def apply_variance_matching(self, perturbations, residual_variance):
        V_eta = np.mean(perturbations ** 2)
        if V_eta == 0:
            return perturbations
        alpha = np.sqrt(residual_variance / V_eta)
        return perturbations * alpha, alpha
