# Plot spatial CRPS difference (SED - ASED) to test whether ASED gains
# concentrate in extra-tropics (storm-track regions) consistent with the
# anisotropy hypothesis.
# Panel (a): Global map on Robinson projection with diverging colormap.
# Panel (b): Zonal-mean ΔCRPS as function of latitude with 30-60 bands marked.

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    crps_sed = np.load(os.path.join(RESULTS_DIR, "crps_spatial_sed.npy"))
    crps_ased = np.load(os.path.join(RESULTS_DIR, "crps_spatial_ased.npy"))
    lat = np.load(os.path.join(RESULTS_DIR, "crps_spatial_coords_lat.npy"))
    lon = np.load(os.path.join(RESULTS_DIR, "crps_spatial_coords_lon.npy"))

    delta = crps_sed - crps_ased

    zonal_mean = np.mean(delta, axis=1)

    lon2d, lat2d = np.meshgrid(lon, lat)

    vmax = np.percentile(np.abs(delta), 99)
    vmax = np.ceil(vmax)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.25)

    # --- Panel (a): Spatial map ---
    ax_map = fig.add_subplot(gs[0], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, color="0.3")

    pcm = ax_map.pcolormesh(
        lon2d, lat2d, delta,
        transform=ccrs.PlateCarree(),
        cmap="RdBu",
        vmin=-vmax, vmax=vmax,
        shading="auto",
    )

    cb = fig.colorbar(pcm, ax=ax_map, orientation="horizontal",
                      fraction=0.06, pad=0.06, shrink=0.7)
    cb.set_label(r"$\Delta$CRPS  [CRPS$_\mathrm{SED}$ $-$ CRPS$_\mathrm{ASED}$]  (m$^2$ s$^{-2}$)",
                 fontsize=10)
    ax_map.set_title("(a) Spatial CRPS difference: positive (blue) = ASED better", fontsize=11)

    # --- Panel (b): Zonal-mean ---
    ax_zonal = fig.add_subplot(gs[1])
    ax_zonal.plot(lat, zonal_mean, color="k", linewidth=1.5)
    ax_zonal.axhline(0, color="0.5", linewidth=0.5, linestyle="--")

    ax_zonal.axvspan(30, 60, alpha=0.15, color="steelblue", label="Extra-tropics 30\u201360\u00b0N")
    ax_zonal.axvspan(-60, -30, alpha=0.15, color="steelblue", label="Extra-tropics 30\u201360\u00b0S")

    ax_zonal.set_xlabel("Latitude (\u00b0)", fontsize=10)
    ax_zonal.set_ylabel(r"Zonal-mean $\Delta$CRPS", fontsize=10)
    ax_zonal.set_title("(b) Zonal-mean CRPS improvement (ASED over SED)", fontsize=11)
    ax_zonal.set_xlim(-90, 90)
    ax_zonal.legend(fontsize=8, loc="upper right")

    mean_extra_n = np.mean(zonal_mean[(lat >= 30) & (lat <= 60)])
    mean_extra_s = np.mean(zonal_mean[(lat >= -60) & (lat <= -30)])
    mean_tropics = np.mean(zonal_mean[(lat > -30) & (lat < 30)])
    mean_global = np.mean(zonal_mean)

    text_lines = (
        f"Global mean: {mean_global:.2f}\n"
        f"Tropics (<30\u00b0): {mean_tropics:.2f}\n"
        f"Extra-tropics NH (30\u201360\u00b0N): {mean_extra_n:.2f}\n"
        f"Extra-tropics SH (30\u201360\u00b0S): {mean_extra_s:.2f}"
    )
    ax_zonal.text(0.02, 0.95, text_lines, transform=ax_zonal.transAxes,
                  fontsize=7.5, verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    out_path = os.path.join(FIGURES_DIR, "crps_spatial_diff.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")

    png_path = os.path.join(FIGURES_DIR, "crps_spatial_diff.png")
    fig2 = plt.figure(figsize=(12, 8))
    gs2 = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.25)
    ax_map2 = fig2.add_subplot(gs2[0], projection=ccrs.Robinson())
    ax_map2.set_global()
    ax_map2.add_feature(cfeature.COASTLINE, linewidth=0.5, color="0.3")
    pcm2 = ax_map2.pcolormesh(
        lon2d, lat2d, delta,
        transform=ccrs.PlateCarree(),
        cmap="RdBu", vmin=-vmax, vmax=vmax, shading="auto",
    )
    cb2 = fig2.colorbar(pcm2, ax=ax_map2, orientation="horizontal",
                        fraction=0.06, pad=0.06, shrink=0.7)
    cb2.set_label(r"$\Delta$CRPS  [CRPS$_\mathrm{SED}$ $-$ CRPS$_\mathrm{ASED}$]  (m$^2$ s$^{-2}$)",
                  fontsize=10)
    ax_map2.set_title("(a) Spatial CRPS difference: positive (blue) = ASED better", fontsize=11)
    ax_zonal2 = fig2.add_subplot(gs2[1])
    ax_zonal2.plot(lat, zonal_mean, color="k", linewidth=1.5)
    ax_zonal2.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax_zonal2.axvspan(30, 60, alpha=0.15, color="steelblue", label="Extra-tropics 30\u201360\u00b0N")
    ax_zonal2.axvspan(-60, -30, alpha=0.15, color="steelblue", label="Extra-tropics 30\u201360\u00b0S")
    ax_zonal2.set_xlabel("Latitude (\u00b0)", fontsize=10)
    ax_zonal2.set_ylabel(r"Zonal-mean $\Delta$CRPS", fontsize=10)
    ax_zonal2.set_title("(b) Zonal-mean CRPS improvement (ASED over SED)", fontsize=11)
    ax_zonal2.set_xlim(-90, 90)
    ax_zonal2.legend(fontsize=8, loc="upper right")
    ax_zonal2.text(0.02, 0.95, text_lines, transform=ax_zonal2.transAxes,
                   fontsize=7.5, verticalalignment="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"PNG copy saved to {png_path}")


if __name__ == "__main__":
    main()
