# Degree-only Spectral Error Dressing (SED) baseline.
# Matches the empirical residual degree power spectrum C_l from calibration data
# and samples isotropic (uniform across m) Gaussian perturbations in SH space.
# C_l is estimated from SHT of calibration residuals, smoothed with 10 log-spaced bins.
# Perturbation variance Var(eta_{lm}) = C_l for all m (no within-degree anisotropy).

import numpy as np
import pyshtools as sh

from ased.spectral.sht_utils import (
    forward_sht,
    degree_power_spectrum,
    smooth_spectrum,
    DEFAULT_LMAX,
)


class SEDSampler:

    def __init__(self, lmax=None):
        self.lmax = lmax if lmax is not None else DEFAULT_LMAX
        self.C_l = None

    def estimate_degree_spectrum(self, residuals_calib, n_bins=10):
        n_times = residuals_calib.shape[0]
        lmax = self.lmax
        all_Pl = np.zeros((n_times, lmax + 1))
        for t in range(n_times):
            coeffs = forward_sht(residuals_calib[t], lmax=lmax)
            all_Pl[t] = degree_power_spectrum(coeffs, lmax=lmax)
            if (t + 1) % 20 == 0 or t + 1 == n_times:
                print(f"    SHT {t+1}/{n_times}")
        C_l = np.mean(all_Pl, axis=0)
        self.C_l = smooth_spectrum(C_l, n_bins=n_bins, log_spaced=True)
        return self.C_l

    def sample_perturbations(self, n_members, seed=0):
        if self.C_l is None:
            raise RuntimeError("Must call estimate_degree_spectrum first")
        lmax = self.lmax
        rng = np.random.default_rng(seed)
        sqrt_cl = np.sqrt(self.C_l[:lmax + 1])
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
            return perturbations, 0.0
        alpha = np.sqrt(residual_variance / V_eta)
        return perturbations * alpha, alpha
