#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np, pandas as pd

def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p): return p
    return None

def unify_id(df):
    for c in ["id","uniprot_id","uid","entry","accession"]:
        if c in df.columns:
            return df.rename(columns={c:"id"}) if c!="id" else df
    for c in df.columns:
        if "id" in c.lower(): return df.rename(columns={c:"id"})
    raise KeyError("id column not found")

def pick_score(df):
    for c in ["bcr_q_effect","z_bcr_q","bcr_diff","z_bcr_diff"]:
        if c in df.columns: return c
    return None

def to_tex_table(df, caption, label):
    cols = df.columns
    header = " & ".join(cols) + r" \\"
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{%s}" % ("l" + "r"*(len(cols)-1)),
             r"\hline", header, r"\hline"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if abs(v) >= 1000:
                    vals.append(f"{v:.2e}")
                else:
                    vals.append(f"{v:.3g}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{%s}" % caption,
              r"\label{%s}" % label, r"\end{table}"]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default=None)
    ap.add_argument("--top", default=None)
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--outdir", default="paper/tables")
    args = ap.parse_args()

    mech = args.mech or first_existing([
        "results/bcrparts_merged/mechspec_rescued.csv",
        "results/bcrparts_merged/mechspec_concat.csv",
        "results/bcrparts/mechspec.csv","results/mechspec.csv",
    ])
    top = args.top or first_existing([
        "results/bcrparts_merged/topN_perm_twoside.csv","results/bcrparts/topN.csv","results/topN.csv",
    ])
    evidence = args.evidence or first_existing([
        "results/bcrparts/evidence_top100.csv","results/bcrparts_merged/evidence_top100.csv","results/evidence_top100.csv",
    ])
    if mech is None or top is None:
        sys.exit("ERROR: mech or top not found")

    os.makedirs(args.outdir, exist_ok=True)
    m = unify_id(pd.read_csv(mech))
    t = unify_id(pd.read_csv(top))
    e = unify_id(pd.read_csv(evidence)) if evidence else None

    score = pick_score(t) or pick_score(m) or "bcr_q_effect"
    use_cols = ["id","k",score,"p_perm","q_bh","FDR_pass","hinge_len"]
    have = [c for c in use_cols if c in t.columns]
    tbl = t[have].copy().head(50)  
    if "FDR_pass" in tbl.columns:
        tbl["FDR_pass"] = tbl["FDR_pass"].map({True:"T", False:"F"})
    if "q_bh" in tbl.columns:
        tbl["q_bh"] = tbl["q_bh"].astype(float)
    if "p_perm" in tbl.columns:
        tbl["p_perm"] = tbl["p_perm"].astype(float)
    tex = to_tex_table(tbl, "Top candidates with statistics.", "tab:topN")
    with open(os.path.join(args.outdir, "table_topN.tex"), "w") as f:
        f.write(tex)

    if e is not None:
        cov_cols = [c for c in ["pdbflex_maxRMSD_max","pdbflex_avgRMSD_max","codnas_maxRMSD","codnas_pair_count",
                                "num_pdb_structures","min_resolution","distinct_methods"] if c in e.columns]
        cov = []
        for c in cov_cols:
            s = e[c]; nonnull = int(s.notna().sum()); med = float(s.dropna().median()) if s.notna().any() else np.nan
            q1 = float(s.dropna().quantile(0.25)) if s.notna().any() else np.nan
            q3 = float(s.dropna().quantile(0.75)) if s.notna().any() else np.nan
            cov.append((c, nonnull, med, q1, q3))
        cov_df = pd.DataFrame(cov, columns=["metric","non_null","median","Q1","Q3"])
        tex2 = to_tex_table(cov_df, "Coverage of external evidence metrics.", "tab:evidence_coverage")
        with open(os.path.join(args.outdir, "table_coverage.tex"), "w") as f:
            f.write(tex2)

    print("[done] tables ->", args.outdir)

if __name__ == "__main__":
    main()
