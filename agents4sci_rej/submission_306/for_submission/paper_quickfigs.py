#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def unify_id(df):
    for c in ["id","uniprot_id","uid","entry","accession"]:
        if c in df.columns:
            if c != "id": df = df.rename(columns={c:"id"})
            return df
    for c in df.columns:
        if "id" in c.lower():
            return df.rename(columns={c:"id"})
    raise KeyError("id column not found")

def pick_score_col(df):
    for c in ["bcr_q_effect","z_bcr_q","bcr_diff","z_bcr_diff"]:
        if c in df.columns:
            return c
    raise KeyError("no score column found (need one of bcr_q_effect/z_bcr_q/bcr_diff/z_bcr_diff)")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def safe_spearman(x, y):
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    xr = (xr - xr.mean())/ (xr.std() + 1e-12)
    yr = (yr - yr.mean())/ (yr.std() + 1e-12)
    return float((xr*yr).mean())

def annotate_stats(ax, x, y, loc="upper left"):
    if len(x) < 3:
        ax.text(0.02, 0.98, "n<3", transform=ax.transAxes, va="top")
        return
    x = np.asarray(x, float); y = np.asarray(y, float)
    # Pearson
    xm = x.mean(); ym = y.mean()
    xs = x.std() + 1e-12; ys = y.std() + 1e-12
    r = float(np.mean((x-xm)*(y-ym))/(xs*ys))
    rho = safe_spearman(x, y)
    text = f"n={len(x)}\nPearson r={r:.2f}\nSpearman ρ={rho:.2f}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top")

def plot_score_hist(mech, outdir):
    score = pick_score_col(mech)
    fdr = mech["FDR_pass"] if "FDR_pass" in mech.columns else (mech.get("q_bh", pd.Series([1]*len(mech)))<=0.05)
    fig, ax = plt.subplots(figsize=(6,4))
    vals_pass = mech.loc[fdr, score].dropna().values
    vals_fail = mech.loc[~fdr, score].dropna().values
    bins = 30
    ax.hist(vals_fail, bins=bins, alpha=0.6, label="FDR fail")
    ax.hist(vals_pass, bins=bins, alpha=0.6, label="FDR pass")
    ax.set_xlabel(score); ax.set_ylabel("count"); ax.set_title("Score distribution (pass/fail)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_score_hist.png"), dpi=300); plt.close(fig)

def plot_fdr_bar(mech, outdir):
    fdr = mech["FDR_pass"] if "FDR_pass" in mech.columns else (mech.get("q_bh", pd.Series([1]*len(mech)))<=0.05)
    cnt_pass = int(fdr.sum()); cnt_fail = int((~fdr).sum())
    fig, ax = plt.subplots(figsize=(4.5,4))
    ax.bar(["FDR pass","FDR fail"], [cnt_pass, cnt_fail])
    ax.set_ylabel("count"); ax.set_title("BH-FDR results")
    for i,v in enumerate([cnt_pass, cnt_fail]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_fdr_bar.png"), dpi=300); plt.close(fig)

def plot_hinge_hist(mech, outdir):
    if "hinge_len" not in mech.columns: return
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(mech["hinge_len"].dropna().values, bins=30)
    ax.set_xlabel("hinge_len"); ax.set_ylabel("count"); ax.set_title("Hinge length distribution")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_hinge_len_hist.png"), dpi=300); plt.close(fig)

def plot_asymmetry_scatter(mech, outdir):
    asym = "asymmetry_index" if "asymmetry_index" in mech.columns else ("pae_asym" if "pae_asym" in mech.columns else None)
    if asym is None: return
    score = pick_score_col(mech)
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    x = mech[score].values; y = mech[asym].values
    ax.scatter(x, y, s=10, alpha=0.6)
    ax.set_xlabel(score); ax.set_ylabel("asymmetry_index"); ax.set_title("Score vs PAE asymmetry")
    annotate_stats(ax, x[~np.isnan(x)], y[~np.isnan(y)])
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_score_vs_asymmetry.png"), dpi=300); plt.close(fig)

def plot_rmsd_scatter(top, evidence, outdir, which):
    if evidence is None: return
    cols_map = {
        "pdbflex": ["pdbflex_maxRMSD_max","pdbflex_avgRMSD_max","pdbflex_maxRMSD","pdbflex_avgRMSD"],
        "codnas":  ["codnas_maxRMSD","codnas_avgRMSD"],
    }
    ycol = next((c for c in cols_map[which] if c in evidence.columns), None)
    if ycol is None: return
    evi = evidence.copy()
    evi = unify_id(evi)
    t = unify_id(top.copy())
    score = pick_score_col(t)
    df = t.merge(evi[["id", ycol]], on="id", how="left").dropna(subset=[ycol, score])
    if len(df) < 3: return
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    ax.scatter(df[score].values, df[ycol].values, s=18, alpha=0.7)
    ax.set_xlabel(score); ax.set_ylabel(f"{which.upper()} RMSD")
    ax.set_title(f"Top-N: {score} vs {which.upper()} RMSD")
    annotate_stats(ax, df[score].values, df[ycol].values)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig_{which}_rmsd_scatter.png"), dpi=300); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default=None, help="mechspec CSV")
    ap.add_argument("--top", default=None, help="topN CSV")
    ap.add_argument("--evidence", default=None, help="evidence_top100 CSV")
    ap.add_argument("--outdir", default="paper/figs", help="output dir")
    args = ap.parse_args()

    mech = args.mech or first_existing([
        "results/bcrparts_merged/mechspec_rescued.csv",
        "results/bcrparts_merged/mechspec_concat.csv",
        "results/bcrparts/mechspec.csv",
        "results/mechspec.csv",
    ])
    top = args.top or first_existing([
        "results/bcrparts_merged/topN_perm_twoside.csv",
        "results/bcrparts_merged/topN.csv",
        "results/bcrparts/topN.csv",
        "results/topN.csv",
    ])
    evidence = args.evidence or first_existing([
        "results/bcrparts/evidence_top100.csv",
        "results/bcrparts_merged/evidence_top100.csv",
        "results/evidence_top100.csv",
    ])

    if mech is None or top is None:
        sys.exit("ERROR: mech or top CSV not found. Use --mech/--top or place files under results/...")

    print("[info] mech =", mech)
    print("[info] top  =", top)
    print("[info] evidence =", evidence)

    ensure_dir(args.outdir)

    m = pd.read_csv(mech); m = unify_id(m)
    t = pd.read_csv(top);  t = unify_id(t)
    e = pd.read_csv(evidence) if evidence else None

    plot_method_schematic(os.path.join(args.outdir, "fig_method.png"))
    plot_score_hist(m, args.outdir)
    plot_fdr_bar(m, args.outdir)
    plot_hinge_hist(m, args.outdir)
    plot_asymmetry_scatter(m, args.outdir)
    plot_rmsd_scatter(t, e, args.outdir, "pdbflex")
    plot_rmsd_scatter(t, e, args.outdir, "codnas")

    print("[done] figures ->", args.outdir)

if __name__ == "__main__":
    main()
