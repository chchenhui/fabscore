# Anisotropic Spectral Error Dressing (ASED) — the proposed method.
# Extends SED by adding within-degree anisotropy profile that redistributes
# variance across SH orders m based on mu = |m|/l bins.
# Supports: N mu-bins (not just 2), degree-band-dependent weights.
# Per-degree mean variance is preserved: (1/(2l+1)) * sum_m g_{lm} = 1 for all l.

import numpy as np
import pyshtools as sh

from ased.spectral.sht_utils import (
    forward_sht,
    DEFAULT_LMAX,
)


def _make_degree_bands(l_min, lmax, n_bands):
    if n_bands <= 1:
        return [(l_min, lmax)]
    edges = np.logspace(np.log10(l_min), np.log10(lmax + 1), n_bands + 1)
    edges = np.unique(np.round(edges).astype(int))
    edges[0] = l_min
    edges[-1] = lmax + 1
    bands = []
    for i in range(len(edges) - 1):
        if edges[i] < edges[i + 1]:
            bands.append((int(edges[i]), int(edges[i + 1]) - 1))
    return bands


def _make_mu_edges(n_mu_bins):
    return np.linspace(0.0, 1.0 + 1e-9, n_mu_bins + 1)


def _mu_bin_index(mu, mu_edges):
    for i in range(len(mu_edges) - 1):
        if mu < mu_edges[i + 1]:
            return i
    return len(mu_edges) - 2


class ASEDSampler:

    def __init__(self, lmax=None, mu_split=0.5, l_min=10, n_aniso_bands=1, n_mu_bins=2):
        self.lmax = lmax if lmax is not None else DEFAULT_LMAX
        self.mu_split = mu_split
        self.l_min = l_min
        self.n_aniso_bands = n_aniso_bands
        self.n_mu_bins = n_mu_bins
        if n_mu_bins == 2:
            self.mu_edges = np.array([0.0, mu_split, 1.0 + 1e-9])
        else:
            self.mu_edges = _make_mu_edges(n_mu_bins)
        self.w_low = None
        self.w_high = None
        self.band_weights = None
        self.mu_bin_weights = None
        self.bands = None
        self.g_lm = None

    def estimate_anisotropy_profile(self, residuals_calib):
        n_times = residuals_calib.shape[0]
        lmax = self.lmax
        l_min = self.l_min
        n_mu_bins = self.n_mu_bins
        mu_edges = self.mu_edges

        P_bin_all = np.zeros((n_mu_bins, n_times, lmax + 1))
        count_bin = np.zeros((n_mu_bins, lmax + 1), dtype=int)

        for l in range(l_min, lmax + 1):
            for m in range(l + 1):
                mu = m / l if l > 0 else 0.0
                b = _mu_bin_index(mu, mu_edges)
                n = 1 if m == 0 else 2
                count_bin[b, l] += n

        for t in range(n_times):
            coeffs = forward_sht(residuals_calib[t], lmax=lmax)
            for l in range(l_min, lmax + 1):
                power_per_bin = np.zeros(n_mu_bins)
                for m in range(l + 1):
                    mu = m / l if l > 0 else 0.0
                    b = _mu_bin_index(mu, mu_edges)
                    p = coeffs[0, l, m] ** 2
                    if m > 0:
                        p += coeffs[1, l, m] ** 2
                    power_per_bin[b] += p
                for b in range(n_mu_bins):
                    if count_bin[b, l] > 0:
                        P_bin_all[b, t, l] = power_per_bin[b] / count_bin[b, l]
            if (t + 1) % 20 == 0 or t + 1 == n_times:
                print(f"    Anisotropy SHT {t+1}/{n_times}")

        P_bin_mean = np.mean(P_bin_all[:, :, l_min:], axis=1)

        self.bands = _make_degree_bands(l_min, lmax, self.n_aniso_bands)
        self.band_weights = []

        for band_lo, band_hi in self.bands:
            idx_lo = band_lo - l_min
            idx_hi = band_hi - l_min + 1
            sl = slice(idx_lo, idx_hi)
            bin_w = []
            for b in range(n_mu_bins):
                valid = count_bin[b, band_lo:band_hi + 1] > 0
                w = float(np.mean(P_bin_mean[b, sl][valid])) if np.any(valid) else 1.0
                bin_w.append(w)
            self.band_weights.append(bin_w)
            ratios = [f"{w:.4e}" for w in bin_w]
            print(f"  Band l=[{band_lo},{band_hi}]: weights={ratios}")

        global_weights = []
        for b in range(n_mu_bins):
            valid = count_bin[b, l_min:] > 0
            w = float(np.mean(P_bin_mean[b][valid])) if np.any(valid) else 1.0
            global_weights.append(w)
        self.mu_bin_weights = global_weights

        if n_mu_bins == 2:
            self.w_low = global_weights[0]
            self.w_high = global_weights[1]
            print(f"  Global average: w_low = {self.w_low:.6e}, w_high = {self.w_high:.6e}")
            print(f"  Global w_low/w_high ratio = {self.w_low / self.w_high:.4f}")
        else:
            print(f"  Global mu-bin weights: {[f'{w:.6e}' for w in global_weights]}")
            if global_weights[-1] > 0:
                print(f"  w[0]/w[-1] ratio = {global_weights[0] / global_weights[-1]:.4f}")
            self.w_low = global_weights[0]
            self.w_high = global_weights[-1]

        return self.w_low, self.w_high

    def _get_band_weights_for_l(self, l):
        if self.band_weights is None or len(self.band_weights) == 0:
            return self.mu_bin_weights if self.mu_bin_weights else [self.w_low, self.w_high]
        if self.n_aniso_bands <= 1:
            return self.mu_bin_weights if self.mu_bin_weights else [self.w_low, self.w_high]
        for (band_lo, band_hi), bw in zip(self.bands, self.band_weights):
            if band_lo <= l <= band_hi:
                return bw
        return self.mu_bin_weights if self.mu_bin_weights else [self.w_low, self.w_high]

    def compute_per_lm_multiplier(self, w_low=None, w_high=None):
        override_2bin = (w_low is not None or w_high is not None)

        if override_2bin:
            if w_low is None:
                w_low = self.w_low
            if w_high is None:
                w_high = self.w_high
            if w_low is None or w_high is None:
                raise RuntimeError("Must call estimate_anisotropy_profile first or provide w_low/w_high")

        lmax = self.lmax
        mu_edges = self.mu_edges
        l_min = self.l_min

        g = np.ones((2, lmax + 1, lmax + 1))

        for l in range(l_min, lmax + 1):
            if override_2bin:
                bin_w = [w_low, w_high]
            else:
                bin_w = self._get_band_weights_for_l(l)

            w_sum = 0.0
            for m in range(l + 1):
                mu = m / l if l > 0 else 0.0
                if override_2bin:
                    b = 0 if mu < self.mu_split else 1
                else:
                    b = _mu_bin_index(mu, mu_edges)
                w = bin_w[b]
                n = 1 if m == 0 else 2
                w_sum += w * n
            w_bar_l = w_sum / (2 * l + 1)

            for m in range(l + 1):
                mu = m / l if l > 0 else 0.0
                if override_2bin:
                    b = 0 if mu < self.mu_split else 1
                else:
                    b = _mu_bin_index(mu, mu_edges)
                w = bin_w[b]
                g_val = w / w_bar_l
                g[0, l, m] = g_val
                if m > 0:
                    g[1, l, m] = g_val

        self.g_lm = g
        return g

    def sample_perturbations(self, C_l, n_members, seed=0):
        if self.g_lm is None:
            raise RuntimeError("Must call compute_per_lm_multiplier first")
        lmax = self.lmax
        g = self.g_lm
        rng = np.random.default_rng(seed)

        sqrt_cl = np.sqrt(C_l[:lmax + 1])
        scale = np.zeros((2, lmax + 1, lmax + 1))
        mask = np.zeros((2, lmax + 1, lmax + 1), dtype=bool)
        for l in range(lmax + 1):
            mask[0, l, 0] = True
            for m in range(1, l + 1):
                mask[0, l, m] = True
                mask[1, l, m] = True
            scale[:, l, :l + 1] = sqrt_cl[l]

        scale = scale * np.sqrt(g)
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
