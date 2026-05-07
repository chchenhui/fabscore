# Spherical Harmonic Transform utilities using pyshtools.
# Provides wrappers for forward SHT, inverse iSHT, degree power spectrum computation,
# and spectrum smoothing. Handles the WB2 0.25-deg grid (721x1440, lats ascending -90..90)
# by trimming to 720x1440 DH2 format and flipping lats to N-to-S for pyshtools.
# Uses MakeGridDH with sampling=2 for inverse to preserve 720x1440 shape.

import numpy as np
import pyshtools as sh

EARTH_NLAT_WB2 = 721
EARTH_NLON_WB2 = 1440
DEFAULT_LMAX = 359


def _wb2_to_pyshtools(field):
    nlat, nlon = field.shape
    if nlat == EARTH_NLAT_WB2:
        field = field[:-1, :]
    return field[::-1, :]


def _pyshtools_to_wb2(grid_720x1440):
    flipped = grid_720x1440[::-1, :]
    south_pole = np.full((1, flipped.shape[1]), flipped[0, :].mean())
    return np.concatenate([south_pole, flipped], axis=0)


def forward_sht(field, lmax=None, normalization='4pi'):
    if lmax is None:
        lmax = DEFAULT_LMAX
    prepared = _wb2_to_pyshtools(field)
    grid = sh.SHGrid.from_array(prepared, grid='DH')
    coeffs = grid.expand(normalization=normalization, lmax_calc=lmax)
    return coeffs.to_array()


def inverse_sht(coeffs, lmax=None, target_nlat=EARTH_NLAT_WB2):
    if isinstance(coeffs, np.ndarray):
        arr = coeffs
    else:
        arr = coeffs.to_array()
    grid_data = sh.expand.MakeGridDH(arr, sampling=2, extend=0, norm=1)
    if target_nlat == EARTH_NLAT_WB2:
        return _pyshtools_to_wb2(grid_data)
    return grid_data[::-1, :]


def degree_power_spectrum(coeffs, lmax=None):
    if isinstance(coeffs, np.ndarray):
        arr = coeffs
    else:
        arr = coeffs.to_array()
    if lmax is None:
        lmax = arr.shape[1] - 1
    C_l = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        power = arr[0, l, 0] ** 2
        for m in range(1, l + 1):
            power += arr[0, l, m] ** 2 + arr[1, l, m] ** 2
        C_l[l] = power / (2 * l + 1)
    return C_l


def smooth_spectrum(C_l, n_bins=10, log_spaced=True):
    lmax = len(C_l) - 1
    smoothed = C_l.copy()
    if log_spaced:
        bin_edges = np.unique(np.logspace(0, np.log10(lmax + 1), n_bins + 1).astype(int))
        bin_edges[0] = 0
        bin_edges[-1] = lmax + 1
    else:
        bin_edges = np.linspace(0, lmax + 1, n_bins + 1).astype(int)
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if lo >= hi:
            continue
        bin_mean = np.mean(C_l[lo:hi])
        smoothed[lo:hi] = bin_mean
    return smoothed
