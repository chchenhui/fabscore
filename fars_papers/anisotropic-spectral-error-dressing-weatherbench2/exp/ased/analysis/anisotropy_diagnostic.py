# Calibration anisotropy index (A_cal) computation and spectral diagnostics.
# Computes P_low(l), P_high(l), A(l), A_cal from bias-corrected calibration residuals.
# Generates two figures: degree_spectrum.pdf and anisotropy_profile.pdf.
# Saves results to ased/results/anisotropy_diagnostic.json.
# No GPU required — runs entirely on CPU using cached local data.

import gc
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ased.data.wb2_loader import (
    load_graphcast_z500,
    load_era5_z500,
    split_calibration_evaluation,
    align_forecast_truth,
    compute_bias_correction,
    apply_bias_correction,
)
from ased.spectral.sht_utils import (
    forward_sht,
    degree_power_spectrum,
    smooth_spectrum,
    DEFAULT_LMAX,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
LEAD_TIME_DAYS = 5
LMAX = DEFAULT_LMAX
L_MIN = 10
MU_SPLIT = 0.5
EARTH_RADIUS_KM = 6371.0
GP_LENGTH_SCALE_KM = 1200.0


def load_calibration_residuals():
    print("Loading GraphCast Z500 forecasts (5-day lead, 2020)...")
    t0 = time.time()
    fc = load_graphcast_z500(year=2020, lead_time_days=LEAD_TIME_DAYS)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("Loading ERA5 Z500 truth (2020)...")
    t0 = time.time()
    truth = load_era5_z500(year=2020)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("Splitting calibration / evaluation...")
    fc_calib, _ = split_calibration_evaluation(fc)
    del fc
    gc.collect()

    fc_calib_a, truth_calib_a = align_forecast_truth(fc_calib, truth, lead_time_days=LEAD_TIME_DAYS)
    del fc_calib, truth
    gc.collect()

    print("Computing bias correction on calibration split...")
    bias = compute_bias_correction(fc_calib_a, truth_calib_a)
    fc_calib_bc = apply_bias_correction(fc_calib_a, bias)

    residuals = fc_calib_bc["geopotential"].values - truth_calib_a["geopotential"].values
    print(f"  Calibration residuals shape: {residuals.shape}")
    print(f"  V_res = {np.mean(residuals**2):.2f}")

    del fc_calib_a, truth_calib_a, fc_calib_bc, bias
    gc.collect()
    return residuals


def compute_anisotropy_index(residuals, lmax=LMAX, l_min=L_MIN, mu_split=MU_SPLIT):
    n_times = residuals.shape[0]

    count_low = np.zeros(lmax + 1, dtype=int)
    count_high = np.zeros(lmax + 1, dtype=int)
    for l in range(l_min, lmax + 1):
        for m in range(l + 1):
            mu = m / l if l > 0 else 0.0
            n = 1 if m == 0 else 2
            if mu < mu_split:
                count_low[l] += n
            else:
                count_high[l] += n

    power_low_all = np.zeros((n_times, lmax + 1))
    power_high_all = np.zeros((n_times, lmax + 1))
    C_l_all = np.zeros((n_times, lmax + 1))

    for t in range(n_times):
        coeffs = forward_sht(residuals[t], lmax=lmax)
        C_l_all[t] = degree_power_spectrum(coeffs, lmax=lmax)

        for l in range(l_min, lmax + 1):
            p_low = 0.0
            p_high = 0.0
            for m in range(l + 1):
                mu = m / l if l > 0 else 0.0
                p = coeffs[0, l, m] ** 2
                if m > 0:
                    p += coeffs[1, l, m] ** 2
                if mu < mu_split:
                    p_low += p
                else:
                    p_high += p
            if count_low[l] > 0:
                power_low_all[t, l] = p_low / count_low[l]
            if count_high[l] > 0:
                power_high_all[t, l] = p_high / count_high[l]

        if (t + 1) % 20 == 0 or t + 1 == n_times:
            print(f"    SHT {t+1}/{n_times}")

    P_low = np.mean(power_low_all[:, l_min:], axis=0)
    P_high = np.mean(power_high_all[:, l_min:], axis=0)

    denom = P_high + P_low
    A_l = np.where(denom > 0, (P_high - P_low) / denom, 0.0)

    A_cal = float(np.mean(A_l))
    n_l = len(A_l)
    A_cal_se = float(np.std(A_l) / np.sqrt(n_l))

    C_l_mean = np.mean(C_l_all, axis=0)
    C_l_smoothed = smooth_spectrum(C_l_mean, n_bins=10, log_spaced=True)

    degrees = np.arange(l_min, lmax + 1)

    print(f"\n=== Anisotropy Index ===")
    print(f"  A_cal = {A_cal:.6f} +/- {A_cal_se:.6f}")
    print(f"  |A_cal| < 0.1 => {'YES (approximately isotropic)' if abs(A_cal) < 0.1 else 'NO (anisotropic)'}")
    if A_cal < 0:
        print(f"  A_cal < 0 => P_low > P_high (quasi-zonal error structure)")

    return {
        "degrees": degrees,
        "P_low": P_low,
        "P_high": P_high,
        "A_l": A_l,
        "A_cal": A_cal,
        "A_cal_se": A_cal_se,
        "C_l_mean": C_l_mean,
        "C_l_smoothed": C_l_smoothed,
    }


def compute_gp_spectrum(C_l_empirical, lmax=LMAX):
    L_over_R = GP_LENGTH_SCALE_KM / EARTH_RADIUS_KM
    ells = np.arange(lmax + 1, dtype=np.float64)
    C_l_gp_raw = np.exp(-ells * (ells + 1) * L_over_R ** 2)

    total_emp = np.sum((2 * ells + 1) * C_l_empirical[:lmax + 1])
    total_gp = np.sum((2 * ells + 1) * C_l_gp_raw)
    if total_gp > 0:
        C_l_gp = C_l_gp_raw * (total_emp / total_gp)
    else:
        C_l_gp = C_l_gp_raw
    return C_l_gp


def save_results(diag, C_l_gp, lmax=LMAX, l_min=L_MIN):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    degrees = list(range(l_min, lmax + 1))
    output = {
        "method": "calibration_anisotropy_diagnostic",
        "variable": "geopotential_500hPa",
        "lead_time_days": LEAD_TIME_DAYS,
        "lmax": lmax,
        "l_min": l_min,
        "mu_split": MU_SPLIT,
        "A_cal": diag["A_cal"],
        "A_cal_se": diag["A_cal_se"],
        "interpretation": (
            "approximately isotropic (|A_cal| < 0.1, ASED ~ SED expected)"
            if abs(diag["A_cal"]) < 0.1
            else "anisotropic (|A_cal| >= 0.1, ASED should improve over SED)"
        ),
        "quasi_zonal": diag["A_cal"] < 0,
        "per_degree": {
            "l": degrees,
            "A_l": [float(v) for v in diag["A_l"]],
            "P_low": [float(v) for v in diag["P_low"]],
            "P_high": [float(v) for v in diag["P_high"]],
        },
        "degree_spectrum": {
            "C_l_smoothed": [float(v) for v in diag["C_l_smoothed"]],
            "C_l_gp_1200km": [float(v) for v in C_l_gp],
        },
    }
    out_path = os.path.join(RESULTS_DIR, "anisotropy_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


def plot_degree_spectrum(diag, C_l_gp, lmax=LMAX, l_min=L_MIN):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    ells_full = np.arange(lmax + 1)
    degrees = np.arange(l_min, lmax + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    valid = diag["C_l_smoothed"][1:] > 0
    ells_plot = ells_full[1:][valid]
    cl_plot = diag["C_l_smoothed"][1:][valid]
    ax.loglog(ells_plot, cl_plot, 'b-', linewidth=1.5, label='Empirical $C_\\ell$ (smoothed)')
    valid_gp = C_l_gp[1:] > 0
    ells_gp = ells_full[1:][valid_gp]
    cl_gp = C_l_gp[1:][valid_gp]
    ax.loglog(ells_gp, cl_gp, 'r--', linewidth=1.5, label='GP-1200km $C_\\ell^{GP}$')
    ax.set_xlabel('Degree $\\ell$', fontsize=12)
    ax.set_ylabel('Degree power $C_\\ell$', fontsize=12)
    ax.set_title('(a) Degree Power Spectrum', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1, lmax)

    ax = axes[1]
    ax.plot(degrees, diag["A_l"], 'k-', linewidth=0.8, alpha=0.7)
    window = min(15, len(diag["A_l"]) // 5)
    if window > 1:
        A_smooth = np.convolve(diag["A_l"], np.ones(window) / window, mode='valid')
        d_smooth = degrees[:len(A_smooth)] + window // 2
        ax.plot(d_smooth, A_smooth, 'b-', linewidth=2.0, label=f'Smoothed (window={window})')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.0)
    ax.axhspan(-0.1, 0.1, color='green', alpha=0.15, label='$|A| < 0.1$ (isotropic)')
    ax.set_xlabel('Degree $\\ell$', fontsize=12)
    ax.set_ylabel('Anisotropy contrast $A(\\ell)$', fontsize=12)
    ax.set_title('(b) Within-Degree Anisotropy', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(l_min, lmax)

    A_cal = diag["A_cal"]
    A_se = diag["A_cal_se"]
    ax.text(0.97, 0.03, f'$A_{{cal}} = {A_cal:.4f} \\pm {A_se:.4f}$',
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "degree_spectrum.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def plot_anisotropy_profile(diag, lmax=LMAX, l_min=L_MIN):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    degrees = np.arange(l_min, lmax + 1)

    ased_main_path = os.path.join(RESULTS_DIR, "ased_main.json")
    w_low_label, w_high_label = None, None
    if os.path.exists(ased_main_path):
        with open(ased_main_path) as f:
            ased_data = json.load(f)
        w_low_label = ased_data.get("anisotropy", {}).get("w_low")
        w_high_label = ased_data.get("anisotropy", {}).get("w_high")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    valid_low = diag["P_low"] > 0
    valid_high = diag["P_high"] > 0
    ax.loglog(degrees[valid_low], diag["P_low"][valid_low], 'b-', linewidth=1.5,
              label='$P_{low}(\\ell)$ ($|m|/\\ell < 0.5$)')
    ax.loglog(degrees[valid_high], diag["P_high"][valid_high], 'r-', linewidth=1.5,
              label='$P_{high}(\\ell)$ ($|m|/\\ell \\geq 0.5$)')
    ax.axvspan(10, 40, color='orange', alpha=0.1, label='Synoptic ($\\ell \\approx 10$-$40$)')
    ax.set_xlabel('Degree $\\ell$', fontsize=12)
    ax.set_ylabel('Mean power per order', fontsize=12)
    ax.set_title('(a) Low-$\\mu$ vs High-$\\mu$ Power', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(l_min, lmax)

    ax = axes[1]
    valid = valid_low & valid_high & (diag["P_high"] > 0)
    ratio = np.where(valid, diag["P_low"] / diag["P_high"], np.nan)
    ax.plot(degrees[valid], ratio[valid], 'k-', linewidth=0.8, alpha=0.7)
    window = min(15, np.sum(valid) // 5)
    if window > 1:
        r_valid = ratio[valid]
        r_smooth = np.convolve(r_valid, np.ones(window) / window, mode='valid')
        d_valid = degrees[valid]
        d_smooth = d_valid[:len(r_smooth)] + window // 2
        ax.plot(d_smooth, r_smooth, 'b-', linewidth=2.0, label=f'Smoothed (window={window})')
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.0, label='Isotropic (ratio=1)')
    ax.axvspan(10, 40, color='orange', alpha=0.1, label='Synoptic ($\\ell \\approx 10$-$40$)')

    if w_low_label is not None and w_high_label is not None:
        w_ratio = w_low_label / w_high_label
        ax.axhline(y=w_ratio, color='purple', linestyle=':', linewidth=1.5,
                    label=f'ASED $w_{{low}}/w_{{high}} = {w_ratio:.2f}$')
        ax.text(0.97, 0.97,
                f'ASED 2-bin weights:\n$w_{{low}} = {w_low_label:.3f}$\n$w_{{high}} = {w_high_label:.3f}$',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8))

    ax.set_xlabel('Degree $\\ell$', fontsize=12)
    ax.set_ylabel('$P_{low}(\\ell) / P_{high}(\\ell)$', fontsize=12)
    ax.set_title('(b) Power Ratio (Low-$\\mu$ / High-$\\mu$)', fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(l_min, lmax)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "anisotropy_profile.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {out_path}")


def main():
    print("=" * 60)
    print("Calibration Anisotropy Diagnostic")
    print("=" * 60)

    residuals = load_calibration_residuals()

    print(f"\nComputing anisotropy index from {residuals.shape[0]} calibration fields...")
    t0 = time.time()
    diag = compute_anisotropy_index(residuals, lmax=LMAX, l_min=L_MIN)
    elapsed = time.time() - t0
    print(f"  Computed in {elapsed:.1f}s")

    del residuals
    gc.collect()

    print("\nComputing GP-1200km theoretical spectrum...")
    C_l_gp = compute_gp_spectrum(diag["C_l_smoothed"], lmax=LMAX)

    print("\nSaving results...")
    save_results(diag, C_l_gp, lmax=LMAX, l_min=L_MIN)

    print("\nGenerating degree_spectrum.pdf...")
    plot_degree_spectrum(diag, C_l_gp, lmax=LMAX, l_min=L_MIN)

    print("\nGenerating anisotropy_profile.pdf...")
    plot_anisotropy_profile(diag, lmax=LMAX, l_min=L_MIN)

    print("\n=== Done ===")
    print(f"  A_cal = {diag['A_cal']:.6f} +/- {diag['A_cal_se']:.6f}")
    if abs(diag['A_cal']) < 0.1:
        print("  Interpretation: Residuals are approximately isotropic (|A_cal| < 0.1)")
        print("  => ASED ~ SED is expected")
    else:
        print("  Interpretation: Residuals are anisotropic (|A_cal| >= 0.1)")
        print("  => ASED should improve over SED")
    if diag['A_cal'] < 0:
        print("  Quasi-zonal error structure confirmed (A_cal < 0, P_low > P_high)")


if __name__ == "__main__":
    main()
