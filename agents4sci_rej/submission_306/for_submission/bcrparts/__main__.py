import argparse
import os
import time
import json
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.afdb import AFEntry, list_local_uniprot_ids, load_entry, pae_asymmetry
from common.uniprot import fetch_proteome_accessions, fetch_feature_flags, fetch_fasta, fetch_pdb_ids
from common.rcsb import split_by_date
from common.pdbflex import load_pdbflex_evidence, scan_local_stats, fetch_pdbflex_stats
from common.identity import mmseqs_cluster, parse_clusters
from common.codnas import fetch_codnas_summary, parse_codnas_raw
from common.cath import fetch_cath_domains_from_fasta, iou_ranges
from common.segmentation import spectral_bipartition_from_pae, partition_k_spectral, Blocks
from common.metrics import bcr_score, bcr_q_ratio, quartiles_used_normalized
from common.stats import p_from_z, benjamini_hochberg
from common.geometry import (
    parse_ca_coords_from_pdb,
    parse_backbone_from_pdb,
    ensure_cb,
    principal_axis,
    angle_between_axes,
    compute_nc_distance,
    compute_terminal_orientation,
    find_latch_pairs,
)


def _download_for_ids(ids: List[str], concurrency: int = 4) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    def task(uid: str):
        # Trigger load which will cache remote if missing
        return load_entry(uid)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(task, uid): uid for uid in ids}
        for fut in as_completed(futs):
            uid = futs[fut]
            try:
                _ = fut.result()
            except Exception:
                pass
            time.sleep(0.05)


def cmd_fetch(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if getattr(args, 'ids', ''):
        if os.path.isfile(args.ids):
            with open(args.ids) as f:
                ids = [line.strip().split(',')[0] for line in f if line.strip()]
        else:
            ids = [x.strip() for x in args.ids.split(',') if x.strip()]
    elif getattr(args, 'proteome', ''):
        print(f"Fetching UniProt accessions for proteome {args.proteome} ...")
        ids = fetch_proteome_accessions(args.proteome)
        if getattr(args, 'limit', None):
            ids = ids[: args.limit]
    else:
        ids = list_local_uniprot_ids(limit=getattr(args, 'limit', None))
    os.makedirs('data/ids', exist_ok=True)
    out = os.path.join('data', 'ids', 'bcrparts_ids.txt')
    with open(out, 'w') as f:
        for i in ids:
            f.write(i + "\n")
    print(f"Saved {len(ids)} IDs to {out}")
    if getattr(args, 'download', False):
        print(f"Downloading/caching AFDB assets for {len(ids)} IDs ...")
        _download_for_ids(ids, concurrency=getattr(args, 'concurrency', 4))
        print("Download step finished.")
    return ids


def _figures(entry: AFEntry, blocks: Blocks, outdir: str):
    pae = entry.pae
    L = pae.shape[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im = axs[0].imshow(pae, cmap='magma', vmin=0, vmax=31.75, origin='lower')
    axs[0].set_title(f"PAE {entry.uniprot_id}")
    for s, e in blocks.blocks:
        axs[0].add_patch(plt.Rectangle((s, s), e - s, e - s, fill=False, ec='cyan', lw=1.2))
    plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)
    axs[1].plot(np.arange(L), entry.plddt, lw=1.2)
    axs[1].axhline(70, ls='--', c='gray', lw=0.8)
    axs[1].set_ylim(0, 100)
    axs[1].set_title("pLDDT profile")
    axs[1].set_xlabel("Residue")
    axs[1].set_ylabel("pLDDT")
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{entry.uniprot_id}_overview.png"), dpi=200)
    fig.savefig(os.path.join(outdir, f"{entry.uniprot_id}_overview.pdf"))
    plt.close(fig)


def cmd_run(args: argparse.Namespace):
    ids: List[str] = []
    if args.ids_file and os.path.isfile(args.ids_file):
        with open(args.ids_file) as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        ids = list_local_uniprot_ids(limit=args.limit)
    results = []
    outdir = os.path.join('results', 'bcrparts')
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    # Persist a simple config
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', ts)
    try:
        os.makedirs(run_dir, exist_ok=True)
        cfg = {
            'n_perm': int(getattr(args, 'n_perm', 30)),
            'null_mode': getattr(args, 'null_mode', 'perm'),
            'sym_mode': getattr(args, 'sym_mode', 'mean'),
            'alpha': float(getattr(args, 'alpha', 0.05)),
            'k': getattr(args, 'k', '2,3'),
            'min_block_len': int(getattr(args, 'min_block_len', 30)),
            'topN': int(getattr(args, 'topN', 20)),
        }
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            for k, v in cfg.items():
                f.write(f"{k}: {v}\n")
    except Exception:
        pass
    log_path = os.path.join('logs', 'bcrparts.jsonl')
    # optional chunking for large runs
    start = getattr(args, 'start', None)
    count = getattr(args, 'count', None)
    if start is not None or count is not None:
        s = int(start or 0)
        e = s + int(count) if count is not None else None
        ids = ids[s:e]
        print(f"Processing ID slice: {s}:{e} (total {len(ids)})")

    min_len = getattr(args, 'min_block_len', 30)
    # Accumulate all candidates for BH across p_perm
    all_rows = []
    for uid in ids:
        entry = load_entry(uid)
        if entry is None:
            continue
        # Skip very short proteins unlikely to yield meaningful 2-block partitions
        if entry.length < max(2 * min_len, 60):
            continue
        best_row = None
        best_key = None
        # auto mode tries k=2,3
        if getattr(args, 'k', '2,3') == 'auto':
            ks = [2,3]
        else:
            ks = [int(x) for x in (args.k.split(',') if hasattr(args, 'k') else ['2','3'])]
        for kk in ks:
            blks = partition_k_spectral(entry.pae, k=kk, min_block_len=min_len)
            if len(blks.blocks) < 2:
                continue
            null_mode = getattr(args, 'null_mode', 'perm')
            sym_mode = getattr(args, 'sym_mode', 'mean')
            bdiff, zdiff = bcr_score(entry.pae, blks, n_perm=args.n_perm, null_mode=null_mode, sym_mode=sym_mode)
            br = bcr_q_ratio(entry.pae, blks, n_perm=args.n_perm, null_mode=null_mode, sym_mode=sym_mode, return_p=True)
            if isinstance(br, tuple) and len(br) == 3:
                bratio, zratio, p_perm = br
            else:
                bratio, zratio = br  # type: ignore
                p_perm = float('nan')
            q25_intra, q75_inter = quartiles_used_normalized(entry.pae, blks)
            # MechSpec geometry
            (s1,e1),(s2,e2) = blks.blocks[:2]
            # Orientation/NC using model coordinates if available (fallback to cache/remote)
            pdb_path = os.path.join('/data/afdb/alphafold_v4', f'AF-{uid}-F1-model_v4.pdb')
            if not os.path.exists(pdb_path):
                try:
                    from common.afdb import _cache_paths, _fetch_remote
                    _pae_c, model_c = _cache_paths(uid)
                    if not os.path.exists(model_c):
                        _fetch_remote(uid)
                    if os.path.exists(model_c):
                        pdb_path = model_c
                except Exception:
                    pass
            ang = float('nan')
            nc_dist = float('nan')
            latch_pairs_str = ''
            coords = parse_ca_coords_from_pdb(pdb_path) if os.path.exists(pdb_path) else {}
            if coords:
                ncval = compute_nc_distance(coords, entry.length)
                if ncval is not None:
                    nc_dist = float(ncval)
                oa = compute_terminal_orientation(coords, entry.length)
                if oa is not None:
                    ang = float(oa)
                bb = parse_backbone_from_pdb(pdb_path)
                cb = ensure_cb(bb)
                pairs = find_latch_pairs(coords, cb, (s1,e1), (s2,e2), top_k=8)
                if pairs:
                    latch_pairs_str = ';'.join([f"{i}-{j}" for i,j in pairs])
            # pLDDT per block
            med_plddt = []
            for (bs,be) in blks.blocks:
                p = entry.plddt[bs:be]
                med_plddt.append(float(np.nanmedian(p)) if p.size else float('nan'))
            # hinge length: for k=3, central block length; else 0
            hinge_len = 0
            bl_sorted = sorted(blks.blocks, key=lambda x: x[0])
            if kk == 3 and len(bl_sorted) >= 3:
                _, (hs,he), _ = bl_sorted[:3]
                hinge_len = max(0, he - hs)
            flags = fetch_feature_flags(uid) if not getattr(args, 'no_flags', False) else {
                'flag_TM': False,
                'flag_signal': False,
                'flag_coiled': False,
                'flag_repeat': False,
            }
            # CATH IoU if requested
            cath_iou = float('nan')
            if getattr(args, 'cath', False):
                fasta = fetch_fasta(uid)
                if fasta:
                    cath_ranges = fetch_cath_domains_from_fasta(fasta)
                    pred_ranges = [ (s1+1,e1), (s2+1,e2) ]  # convert to 1-based inclusive end
                    cath_iou = iou_ranges(pred_ranges, cath_ranges)
            row = {
                'uniprot_id': uid,
                'length': entry.length,
                'k': kk,
                'bcr_diff': bdiff,
                'z_bcr_diff': zdiff,
                'bcr_q_ratio': bratio,
                'z_bcr_q': zratio,
                'p_perm': p_perm,
                'Q25_intra': q25_intra,
                'Q75_inter': q75_inter,
                'block_1': f"{blks.blocks[0][0]}-{blks.blocks[0][1]}",
                'block_2': f"{blks.blocks[1][0]}-{blks.blocks[1][1]}",
                'hinge_len': hinge_len,
                'pae_asym': pae_asymmetry(entry.pae),
                'NC_dist': nc_dist,
                'orient_angle_deg': ang,
                'median_pLDDT_per_block': ';'.join([f"{x:.1f}" for x in med_plddt if np.isfinite(x)]) if med_plddt else '',
                'latch_pairs': latch_pairs_str,
                **flags,
                'cath_iou': cath_iou,
            }
            all_rows.append(row)
            try:
                with open(log_path, 'a') as lf:
                    lf.write(json.dumps({'uniprot_id': uid, 'k': kk, 'p_perm': p_perm, 'z_bcr_q': zratio}) + "\n")
            except Exception:
                pass
            # prefer smaller p_perm; fallback to higher z
            if p_perm == p_perm:  # finite p
                key = (0, float(p_perm), float(-zratio if np.isfinite(zratio) else 0.0))
            else:
                key = (1, float('inf'), float(-zratio if np.isfinite(zratio) else 0.0))
            if best_key is None or key < best_key:
                best_key = key
                best_row = row
        if best_row:
            try:
                if best_key is not None and best_key[0] == 0:
                    best_row['reason'] = f"auto_k={best_row['k']} by p_perm"
                else:
                    best_row['reason'] = f"auto_k={best_row['k']} by z"
            except Exception:
                pass
            results.append(best_row)
            if args.figures:
                # reconstruct blocks object for plotting
                b1s,b1e = [int(x) for x in best_row['block_1'].split('-')]
                b2s,b2e = [int(x) for x in best_row['block_2'].split('-')]
                _figures(entry, Blocks([(b1s,b1e),(b2s,b2e)]), figdir)
    if not results:
        print("No results computed.")
        return
    # FDR across all candidates using p_perm
    alpha = float(getattr(args, 'alpha', 0.05))
    adf = pd.DataFrame(all_rows)
    if 'p_perm' in adf.columns and len(adf) > 0:
        q_all = benjamini_hochberg(adf['p_perm'].tolist())
        adf['q_bh'] = q_all
        # map back to selected results
        q_map = {(r['uniprot_id'], r['k']): q for r, q in zip(adf.to_dict('records'), adf['q_bh'].tolist())}
    else:
        q_map = {}
    df = pd.DataFrame(results)
    df['q_bh'] = [q_map.get((r['uniprot_id'], r['k']), float('nan')) for _, r in df.iterrows()]
    df['FDR_pass'] = df['q_bh'].apply(lambda q: (q <= alpha) if (q == q) else False)
    # Effect size alias for clarity in downstream tables
    if 'bcr_q_ratio' in df.columns and 'bcr_q_effect' not in df.columns:
        df['bcr_q_effect'] = df['bcr_q_ratio']
    # Sort by p_perm if present
    if 'p_perm' in df.columns:
        df = df.sort_values('p_perm', ascending=True)
    else:
        sort_key = 'z_bcr_q' if 'z_bcr_q' in df.columns else ('z_bcr' if 'z_bcr' in df.columns else 'bcr_diff')
        df = df.sort_values(sort_key, ascending=False)
    df.to_csv(os.path.join(outdir, 'mechspec.csv'), index=False)
    topN = df.head(args.topN)
    topN.to_csv(os.path.join(outdir, 'topN.csv'), index=False)
    # Also export a SwitchParts-100-style TSV for convenience
    catalog_dir = os.path.join('results','catalog')
    os.makedirs(catalog_dir, exist_ok=True)
    df.head(100).to_csv(os.path.join(catalog_dir, 'SwitchParts-100.tsv'), sep='\t', index=False)
    print(f"Saved {len(df)} entries and top {args.topN} to {outdir}")

    # Presets: Hinge (k=3 with central low-pLDDT block)
    try:
        preset_dir = os.path.join(outdir, 'presets')
        os.makedirs(preset_dir, exist_ok=True)
        hinge_rows = []
        ids_for_hinge = df['uniprot_id'].tolist()
        for uid in ids_for_hinge:
            entry = load_entry(uid)
            if entry is None:
                continue
            blks3 = partition_k_spectral(entry.pae, k=3, min_block_len=args.min_block_len if hasattr(args, 'min_block_len') else 30)
            if len(blks3.blocks) < 3:
                continue
            # central block assumed to be the middle in sorted order
            bl_sorted = sorted(blks3.blocks, key=lambda x: x[0])
            (s1,e1), (s2,e2), (s3,e3) = bl_sorted[:3]
            # Hinge criteria
            p2 = entry.plddt[s2:e2]
            p1 = entry.plddt[s1:e1]
            p3 = entry.plddt[s3:e3]
            med2 = float(np.nanmedian(p2)) if p2.size else float('nan')
            med1 = float(np.nanmedian(p1)) if p1.size else float('nan')
            med3 = float(np.nanmedian(p3)) if p3.size else float('nan')
            hinge_len = max(0, s2 - e1) + max(0, s3 - e2)
            if (np.isfinite(med2) and med2 <= 70.0 \
                and (e2 - s2) >= 3 and (e2 - s2) <= 25 \
                and np.isfinite(med1) and med1 >= 80.0 \
                and np.isfinite(med3) and med3 >= 80.0 \
                and hinge_len >= 3):
                # Compute scores for this k=3 partition using the first two blocks for BCR
                bdiff, zdiff = bcr_score(entry.pae, blks3, n_perm=15)
                bratio, zratio = bcr_q_ratio(entry.pae, blks3, n_perm=15)
                q25_intra, q75_inter = quartiles_used_normalized(entry.pae, blks3)
                hinge_rows.append({
                    'uniprot_id': uid,
                    'length': entry.length,
                    'k': 3,
                    'bcr_diff': bdiff,
                    'z_bcr_diff': zdiff,
                    'bcr_q_ratio': bratio,
                    'z_bcr_q': zratio,
                    'Q25_intra': q25_intra,
                    'Q75_inter': q75_inter,
                    'block_1': f"{bl_sorted[0][0]}-{bl_sorted[0][1]}",
                    'block_2': f"{bl_sorted[1][0]}-{bl_sorted[1][1]}",
                    'hinge_len': hinge_len,
                    'median_pLDDT_per_block': f"{med1:.1f};{med2:.1f};{med3:.1f}",
                })
        if hinge_rows:
            pd.DataFrame(hinge_rows).to_csv(os.path.join(preset_dir, 'hinge.csv'), index=False)
    except Exception:
        # preset generation is best-effort; proceed silently on error
        pass


def cmd_paper(args: argparse.Namespace):
    # Create a minimal LaTeX draft using provided template
    src_tex = os.path.join('agents4science_2025.tex')
    paper_dir = os.path.join('paper')
    os.makedirs(paper_dir, exist_ok=True)
    dst_tex = os.path.join(paper_dir, 'main.tex')
    # Insert placeholders for figures and tables
    with open(src_tex, 'r') as f:
        content = f.read()
    content = content.replace('TITLE_GOES_HERE', 'BCR-Parts: Mining AlphaFold PAE for Hinge-like Blocks')
    # Append simple results section at the end
    # Create a simple LaTeX table from topN.csv if available
    topN_csv = os.path.join('results','bcrparts','topN.csv')
    table_tex = os.path.join('results','bcrparts','topN.tex')
    if os.path.exists(topN_csv):
        df = pd.read_csv(topN_csv)
        # prefer BCR_Q columns if present
        cols_v2 = ['uniprot_id','length','bcr_q_ratio','z_bcr_q']
        cols_v1 = ['uniprot_id','length','bcr','z_bcr']
        cols = cols_v2 if set(cols_v2).issubset(df.columns) else [c for c in cols_v1 if c in df.columns]
        df[cols].to_latex(table_tex, index=False, float_format=lambda x: f"{x:.2f}")
    content += "\n% Auto-inserted Results\n\n" \
        + "\\section{Results}\nWe ran BCR-Parts on a subset of AlphaFold DB (local cache) and ranked proteins by BCR Z-score. Table~\\ref{tab:topN} summarizes the top hits. Figures are auto-generated per protein.\n" \
        + ("\\begin{table}[h]\\centering\\caption{Top-ranked candidates}\\label{tab:topN}\\input{../results/bcrparts/topN.tex}\\end{table}\n" if os.path.exists(table_tex) else "")

    # Insert PR curves and validation summary if present
    pr_dev = os.path.join('results','bcrparts','pr_curve_dev.png')
    pr_eval = os.path.join('results','bcrparts','pr_curve_eval.png')
    pr_id = os.path.join('results','bcrparts','pr_curve_identity.png')
    pr_id40 = os.path.join('results','bcrparts','pr_curve_identity_40.png')
    pr_id50 = os.path.join('results','bcrparts','pr_curve_identity_50.png')
    val_csv = os.path.join('results','bcrparts','validation_summary.csv')
    val_tex = os.path.join('results','bcrparts','validation_summary.tex')
    if os.path.exists(val_csv):
        vdf = pd.read_csv(val_csv)
        # keep compact columns
        keep = [c for c in ['uniprot_id','score','has_multi_state','n_pdb','pdbflex_max_rmsd','split'] if c in vdf.columns]
        vdf = vdf[keep].head(15)
        vdf.to_latex(val_tex, index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x,float) else str(x))
        content += "\\subsection{External Validation (Summary)}\n"
        content += ("\\begin{table}[h]\\centering\\caption{Validation summary (top 15 by score)}\\label{tab:val}\\input{../results/bcrparts/validation_summary.tex}\\end{table}\n")
    # Include external evidence histograms if available
    figdir = os.path.join('results','bcrparts','figures')
    cath_hist = os.path.join(figdir, 'cath_iou_hist.png')
    pdbflex_hist = os.path.join(figdir, 'pdbflex_rmsd_hist.png')
    codnas_hist = os.path.join(figdir, 'codnas_rmsd_hist.png')
    ext_figs = []
    if os.path.exists(cath_hist):
        ext_figs.append(cath_hist)
    if os.path.exists(pdbflex_hist):
        ext_figs.append(pdbflex_hist)
    if os.path.exists(codnas_hist):
        ext_figs.append(codnas_hist)
    if ext_figs:
        content += "\\subsection{External Evidence Distributions}\n"
        for pth in ext_figs:
            rel = os.path.relpath(pth, start=paper_dir)
            content += f"\\begin{figure}[h]\\centering\\includegraphics[width=.48\\textwidth]{{{rel}}}\\end{figure}\n"
    # Build Top-12 montage from overview figures if available
    try:
        import matplotlib.image as mpimg
        topN_csv = os.path.join('results','bcrparts','topN.csv')
        figdir = os.path.join('results','bcrparts','figures')
        out_m = os.path.join('results','bcrparts','figs_top12','montage_top12.png')
        os.makedirs(os.path.dirname(out_m), exist_ok=True)
        ids = []
        if os.path.exists(topN_csv):
            tdf = pd.read_csv(topN_csv)
            ids = [str(u) for u in tdf['uniprot_id'].head(12).tolist()]
        if ids:
            n = len(ids)
            rows = 3
            cols = 4
            plt.figure(figsize=(cols*3, rows*3))
            for i, uid in enumerate(ids):
                img_path = os.path.join(figdir, f"{uid}_overview.png")
                if not os.path.exists(img_path):
                    continue
                img = mpimg.imread(img_path)
                ax = plt.subplot(rows, cols, i+1)
                ax.imshow(img)
                ax.set_axis_off()
                ax.set_title(uid, fontsize=8)
            plt.tight_layout()
            plt.savefig(out_m, dpi=200)
            plt.close()
    except Exception:
        pass

    figs = []
    if os.path.exists(pr_dev):
        figs.append(pr_dev)
    if os.path.exists(pr_eval):
        figs.append(pr_eval)
    if os.path.exists(pr_id):
        figs.append(pr_id)
    if os.path.exists(pr_id40):
        figs.append(pr_id40)
    if os.path.exists(pr_id50):
        figs.append(pr_id50)
    if figs:
        content += "\\begin{figure}[h]\\centering\n"
        for fp in figs:
            content += f"\\includegraphics[width=.45\\textwidth]{{../{fp}}}"
        content += "\\caption{Precision--recall curves: temporal splits (dev/eval) and identity-reduced (30/40/50\%).}\\end{figure}\n"

    # Include validation summary table if present
    val_tex = os.path.join('results','bcrparts','validation_summary.tex')
    val_csv = os.path.join('results','bcrparts','validation_summary.csv')
    if os.path.exists(val_csv):
        try:
            vdf = pd.read_csv(val_csv)
            keep = [c for c in ['uniprot_id','score','has_multi_state','n_pdb','pdbflex_max_rmsd','split','codnas_reach'] if c in vdf.columns]
            vdf[keep].head(20).to_latex(val_tex, index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x,float) else str(x))
            content += "\\begin{table}[h]\\centering\\caption{Validation evidence (top 20 by score)}\\label{tab:evidence}\\input{../results/bcrparts/validation_summary.tex}\\end{table}\n"
        except Exception:
            pass
    with open(dst_tex, 'w') as f:
        f.write(content)
    print(f"Wrote LaTeX draft to {dst_tex}")
    # Write a simple rebuttal appendix
    rebut = os.path.join(paper_dir, 'rebuttal.tex')
    rtxt = r"""
\section*{Rebuttal Appendix}
\begin{itemize}
\item Fixed BCR score to include quantile ratio (BCR\_Q) with per-protein Q95 normalization and permutation Z.
\item Added k-selection via spectral bipartition for k∈{2,3}, selecting the best by Z(BCR\_Q).
\item Implemented MechSpec fields (nearest contact, orientation angle, latch pairs) and UniProt-based bias flags (TM, SIGNAL, COILED, REPEAT).
\item Added negatives evaluation (AUPRC), toy BCR vs angle simulation, and a sensitivity sweep of parameters.
\end{itemize}
"""
    with open(rebut, 'w') as f:
        f.write(rtxt)
    print(f"Wrote rebuttal appendix to {rebut}")


def cmd_validate(args: argparse.Namespace):
    # Toy simulation: two blocks with hinge angle controlling inter-block PAE
    import numpy as np
    import matplotlib.pyplot as plt
    L = 200
    blocks = Blocks([(0, L//2), (L//2, L)])
    angles = np.linspace(0, 90, 19)
    bcrs = []
    for ang in angles:
        # synthetic PAE: intra low noise, inter increases with angle
        intra = 2.0
        inter = 5.0 + 0.2 * ang
        pae = np.full((L, L), inter, dtype=np.float32)
        s1, e1 = blocks.blocks[0]
        s2, e2 = blocks.blocks[1]
        pae[s1:e1, s1:e1] = intra
        pae[s2:e2, s2:e2] = intra
        # add small noise
        pae += np.random.default_rng(0).normal(0, 0.2, size=pae.shape).astype(np.float32)
        bcr, z = bcr_score(pae, blocks, n_perm=20)
        bcrs.append((float(ang), float(bcr)))
    arr = np.array(bcrs)
    outdir = os.path.join('results', 'bcrparts')
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(os.path.join(outdir, 'toy_bcr_vs_angle.tsv'), arr, delimiter='\t', header='angle\tbcr', comments='')
    plt.figure(figsize=(4,3))
    plt.plot(arr[:,0], arr[:,1], marker='o')
    plt.xlabel('Hinge angle (deg)')
    plt.ylabel('BCR (synthetic)')
    plt.title('Toy BCR vs hinge angle')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'toy_bcr_vs_angle.png'), dpi=200)
    plt.savefig(os.path.join(outdir, 'toy_bcr_vs_angle.pdf'))
    plt.close()
    print('Saved toy simulation to results/bcrparts')

    if getattr(args, 'sweep', False):
        # Sensitivity sweep over delta and quantiles for first N IDs
        sweep = []
        ids = []
        ids_path = os.path.join('data','ids','bcrparts_ids.txt')
        if os.path.exists(ids_path):
            with open(ids_path) as f:
                ids = [l.strip() for l in f if l.strip()][:10]
        deltas = [5,7,9]
        qints = [0.7,0.75,0.8]
        qins = [0.2,0.25,0.3]
        for uid in ids:
            e = load_entry(uid)
            if e is None:
                continue
            blks = partition_k_spectral(e.pae, k=2)
            for d in deltas:
                for qi in qints:
                    for qj in qins:
                        _, z = bcr_q_ratio(e.pae, blks, delta=d, q_inter=qi, q_intra=qj, n_perm=10)
                        sweep.append({'uniprot_id':uid,'delta':d,'q_inter':qi,'q_intra':qj,'z_bcr_q':z})
        import pandas as pd
        df = pd.DataFrame(sweep)
        df.to_csv(os.path.join('results','bcrparts','sweep.csv'), index=False)
        print('Saved sensitivity sweep to results/bcrparts/sweep.csv')


def _auprc(scores, labels):
    import numpy as np
    scores = np.array(scores, dtype=float)
    order = np.argsort(-scores)
    scores = scores[order]
    labels = np.array(labels)[order]
    tp = 0
    fp = 0
    P = labels.sum()
    precisions = []
    recalls = []
    for y in labels:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(P, 1))
    pr = np.array(list(zip(recalls, precisions)))
    pr = pr[np.argsort(pr[:,0])]
    auprc = 0.0
    for i in range(1, len(pr)):
        x0, y0 = pr[i-1]
        x1, y1 = pr[i]
        auprc += (x1 - x0) * (y0 + y1) / 2.0
    return float(auprc), recalls, precisions


def cmd_buildnegs(args: argparse.Namespace):
    # Auto-select likely single-domain negatives by high pLDDT coverage and no PDB links
    import csv
    ids = []
    if args.ids_file and os.path.isfile(args.ids_file):
        with open(args.ids_file) as f:
            ids = [line.strip() for line in f if line.strip()]
    else:
        ids = list_local_uniprot_ids(limit=args.limit)
    out_csv = os.path.join('data','seed','negatives_auto.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for uid in ids:
        e = load_entry(uid)
        if e is None or e.plddt.size == 0:
            continue
        p = e.plddt
        cov = float(np.mean(np.isfinite(p) & (p >= args.plddt)))
        if cov < args.coverage:
            continue
        pdbs = fetch_pdb_ids(uid)
        if pdbs:
            continue
        rows.append({'uniprot_id': uid})
        if len(rows) >= args.max:
            break
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['uniprot_id'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} auto-negatives to {out_csv}")


def cmd_eval(args: argparse.Namespace):
    # Evaluate separation vs negatives using z_bcr_q if present, else z_bcr_diff
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    outdir = os.path.join('results','bcrparts')
    mech = os.path.join(outdir, 'mechspec.csv')
    if not os.path.exists(mech):
        print('Run bcrparts run first to create mechspec.csv')
        return
    df = pd.read_csv(mech)
    score_col = 'z_bcr_q' if 'z_bcr_q' in df.columns else ('z_bcr_diff' if 'z_bcr_diff' in df.columns else 'bcr_diff')
    # PDB-based multi-state labels via UniProt xrefs: positive if >=2 PDB entries
    pdb_lists = {uid: fetch_pdb_ids(uid) for uid in df['uniprot_id']}
    # Prefer PDBFlex when available: load user-provided CSV/TSV from data/external/pdbflex/raw
    pdbflex_map = load_pdbflex_evidence()
    try:
        pdbflex_map.update(scan_local_stats())
    except Exception:
        pass
    multi_state = {}
    flex_rmsd: Dict[str, float] = {}
    for uid, pdbs in pdb_lists.items():
        # multi-state true if >=2 PDBs or any PDB has cluster_size>=2 in evidence
        is_multi = len(pdbs) >= 2
        mrmsd = None
        for pid in pdbs:
            ev = pdbflex_map.get(pid.upper())
            if not ev:
                continue
            if ev.get('cluster_size') and ev['cluster_size'] >= 2:
                is_multi = True
            if ev.get('max_rmsd') is not None:
                mr = float(ev['max_rmsd'])
                mrmsd = mr if mrmsd is None else max(mrmsd, mr)
        multi_state[uid] = is_multi
        if mrmsd is not None:
            flex_rmsd[uid] = mrmsd
    splits = {uid: split_by_date(pdb_lists[uid], cutoff=args.cutoff_date) if pdb_lists.get(uid) else 'unknown' for uid in df['uniprot_id']}
    cand_scores = df[score_col].tolist()
    # Optional curated positives to ensure mixed labels
    pos_seed_path = os.path.join('data','seed','positives_seed.csv')
    pos_ids = set()
    if os.path.exists(pos_seed_path):
        try:
            import pandas as _pd
            pos_ids = set(_pd.read_csv(pos_seed_path)['uniprot_id'].astype(str).tolist())
        except Exception:
            pos_ids = set()
    # negatives
    negs_path = os.path.join('data','seed','negatives_seed.csv')
    neg_scores = []
    if os.path.exists(negs_path):
        import pandas as pd
        neg_ids = pd.read_csv(negs_path)['uniprot_id'].tolist()
        for uid in neg_ids:
            e = load_entry(uid)
            if e is None:
                continue
            blks = partition_k_spectral(e.pae, k=2)
            bdiff, zdiff = bcr_score(e.pae, blks, n_perm=10)
            bratio, zratio = bcr_q_ratio(e.pae, blks, n_perm=10)
            neg_scores.append(zratio if np.isfinite(zratio) else (zdiff if np.isfinite(zdiff) else bdiff))
    # also include auto-selected negatives if present
    negs_auto = os.path.join('data','seed','negatives_auto.csv')
    if os.path.exists(negs_auto):
        import pandas as pd
        neg_ids = pd.read_csv(negs_auto)['uniprot_id'].tolist()
        for uid in neg_ids:
            e = load_entry(uid)
            if e is None:
                continue
            blks = partition_k_spectral(e.pae, k=2)
            bdiff, zdiff = bcr_score(e.pae, blks, n_perm=10)
            bratio, zratio = bcr_q_ratio(e.pae, blks, n_perm=10)
            neg_scores.append(zratio if np.isfinite(zratio) else (zdiff if np.isfinite(zdiff) else bdiff))
    scores = cand_scores + neg_scores
    labels = [1 if (multi_state.get(uid, False) or uid in pos_ids) else 0 for uid in df['uniprot_id']] + [0]*len(neg_scores)
    if len(set(labels)) < 2:
        print('Not enough negatives evaluated; skipping AUPRC.')
        return
    auprc, recalls, precisions = _auprc(scores, labels)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'eval_negatives.json'), 'w') as f:
        json.dump({'n_candidates': len(cand_scores), 'n_negatives': len(neg_scores), 'auprc': auprc}, f)
    # Precision@K on candidates only
    df_scores = df[[ 'uniprot_id', score_col ]].copy()
    df_scores = df_scores.sort_values(score_col, ascending=False)
    y_true = [1 if (multi_state.get(uid, False) or uid in pos_ids) else 0 for uid in df_scores['uniprot_id']]
    def p_at_k(k: int) -> float:
        k = min(k, len(y_true))
        return float(sum(y_true[:k])) / max(k,1)
    p20 = p_at_k(20)
    p50 = p_at_k(50)
    with open(os.path.join(outdir, 'p_at_k.json'), 'w') as f:
        json.dump({'P@20': p20, 'P@50': p50}, f)
    # Temporal split PR curves (dev/eval)
    for split in ('dev','eval'):
        idx = [i for i, uid in enumerate(df['uniprot_id']) if splits.get(uid) == split]
        if not idx:
            continue
        split_scores = [cand_scores[i] for i in idx] + neg_scores
        split_labels = [1 if multi_state.get(df['uniprot_id'].iloc[i], False) else 0 for i in idx] + [0]*len(neg_scores)
        if len(set(split_labels)) < 2:
            continue
        au, rec, prec = _auprc(split_scores, split_labels)
        plt.figure(figsize=(4,3))
        plt.plot(rec, prec)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title(f'{split.upper()} AUPRC={au:.2f}')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'pr_curve_{split}.png'), dpi=200)
        plt.close()
    # Identity-reduced PR using MMseqs2 (if available)
    try:
        # Write candidate FASTA
        fasta_path = os.path.join(outdir, 'candidates.fasta')
        with open(fasta_path, 'w') as f:
            for uid in df['uniprot_id']:
                fa = fetch_fasta(uid)
                if fa and fa.startswith('>'):
                    # force header to be UID
                    seq = ''.join([line.strip() for line in fa.splitlines() if not line.startswith('>')])
                    f.write(f'>{uid}\n{seq}\n')
        # Sweep identity thresholds: 30%, 40%, 50%
        for thr in (0.3, 0.4, 0.5):
            cludir = os.path.join(outdir, f'mmseqs_clusters_{int(thr*100)}')
            tsv = mmseqs_cluster(fasta_path, cludir, min_seq_id=thr)
            mapping = parse_clusters(tsv)  # member->rep
            reps = set(mapping.values()) if mapping else set(df['uniprot_id'])
            # Build identity-reduced candidate list by representatives only
            rep_mask = df['uniprot_id'].isin(reps)
            df_rep = df[rep_mask].copy().sort_values(score_col, ascending=False)
            cand_rep_scores = df_rep[score_col].tolist()
            labels_rep = [1 if (multi_state.get(uid, False) or uid in pos_ids) else 0 for uid in df_rep['uniprot_id']]
            # combine with negatives for PR
            scores_rep = cand_rep_scores + neg_scores
            labels_mix = labels_rep + [0]*len(neg_scores)
            if len(set(labels_mix)) > 1:
                au_id, rec_id, prec_id = _auprc(scores_rep, labels_mix)
                plt.figure(figsize=(4,3))
                plt.plot(rec_id, prec_id)
                plt.xlabel('Recall'); plt.ylabel('Precision')
                plt.title(f'Identity-reduced AUPRC={au_id:.2f} @ {int(thr*100)}%')
                plt.tight_layout()
                out_png = os.path.join(outdir, f'pr_curve_identity_{int(thr*100)}.png')
                plt.savefig(out_png, dpi=200)
                # Keep legacy filename for 30%
                if abs(thr - 0.3) < 1e-6:
                    plt.savefig(os.path.join(outdir, 'pr_curve_identity.png'), dpi=200)
                plt.close()
                with open(os.path.join(outdir, f'eval_identity_{int(thr*100)}.json'), 'w') as f:
                    json.dump({'auprc_identity': au_id, 'n_rep_candidates': int(rep_mask.sum()), 'min_seq_id': thr}, f)
    except Exception as e:
        # MMseqs not available or failed; skip silently
        pass
    # Save a validation table (CSV) with PDB/Flex/CoDNaS summaries
    out_rows = []
    # Build CoDNaS mapping (chain-level)
    codnas_map = parse_codnas_raw()
    for _, row in df_scores.iterrows():
        uid = row['uniprot_id']
        # CoDNaS chain-level aggregation
        cod_max = None; cod_pairs = 0
        # derive chains for this UniProt
        try:
            from common.uniprot import fetch_pdb_chain_ids
            chains = fetch_pdb_chain_ids(uid)
        except Exception:
            chains = []
        for cid in chains:
            ev = codnas_map.get(cid.upper())
            if not ev:
                continue
            cod_pairs += int(ev.get('pair_count', 0))
            if ev.get('max_rmsd') is not None:
                mr = float(ev['max_rmsd'])
                cod_max = mr if cod_max is None else max(cod_max, mr)
        # CoDNaS reachability (legacy best-effort)
        cod = fetch_codnas_summary(pdb_lists.get(uid, [])) if pdb_lists.get(uid) else {}
        out_rows.append({
            'uniprot_id': uid,
            'score': float(row[score_col]),
            'has_multi_state': bool(multi_state.get(uid, False)),
            'n_pdb': len(pdb_lists.get(uid, [])),
            'pdbflex_max_rmsd': float(flex_rmsd[uid]) if uid in flex_rmsd else None,
            'split': splits.get(uid, 'unknown'),
            'codnas_reach': any((v is not None) for v in cod.values()) if cod else False,
            'codnas_max_rmsd': cod_max,
            'codnas_pair_count': cod_pairs,
        })
    vdf = pd.DataFrame(out_rows)
    vdf.to_csv(os.path.join(outdir, 'validation_summary.csv'), index=False)
    # Reliability diagram (bin by score; precision per bin)
    try:
        import numpy as _np
        import matplotlib.pyplot as _plt
        kbins = 10
        svals = df_scores[score_col].to_numpy()
        ybin = _np.array([1 if (multi_state.get(uid, False)) else 0 for uid in df_scores['uniprot_id']])
        ranks = _np.linspace(0, 1, len(svals), endpoint=False)[::-1]  # high score -> high percentile
        order = _np.argsort(-svals)
        svals = svals[order]
        ybin = ybin[order]
        ranks = ranks[order]
        bins = _np.array_split(_np.arange(len(svals)), kbins)
        px = []
        py = []
        for b in bins:
            if b.size == 0:
                continue
            px.append(float(_np.mean(ranks[b])))
            py.append(float(_np.mean(ybin[b])))
        _plt.figure(figsize=(4,3))
        _plt.plot([0,1],[0,1],'--',c='gray',lw=0.8)
        _plt.plot(px, py, marker='o')
        _plt.xlabel('Score quantile (proxy)')
        _plt.ylabel('Empirical precision')
        _plt.title('Reliability diagram (candidates)')
        _plt.tight_layout()
        _plt.savefig(os.path.join(outdir, 'reliability.png'), dpi=200)
        _plt.close()
    except Exception:
        pass
    print(f'Eval saved to {outdir} (AUPRC, P@K, PR curves, validation_summary.csv, reliability.png)')
    print(f'Eval saved to {outdir}/eval_negatives.json (AUPRC={auprc:.3f})')


def cmd_ablate(args: argparse.Namespace):
    # Lightweight ablations using existing scores/labels without full recomputation
    import json
    outdir = os.path.join('results','bcrparts')
    mech = os.path.join(outdir, 'mechspec.csv')
    if not os.path.exists(mech):
        print('Run bcrparts run first.')
        return
    df = pd.read_csv(mech)
    score_cols = []
    if {'bcr_q_ratio','z_bcr_q'}.issubset(df.columns):
        score_cols.append(('bcr_q_ratio','z_bcr_q'))
    if {'bcr_diff','z_bcr_diff'}.issubset(df.columns):
        score_cols.append(('bcr_diff','z_bcr_diff'))
    # labels via PDBs + curated positives
    from common.uniprot import fetch_pdb_ids
    pdb_lists = {uid: fetch_pdb_ids(uid) for uid in df['uniprot_id']}
    multi_state = {uid: (len(pdbs)>=2) for uid,pdbs in pdb_lists.items()}
    pos_seed_path = os.path.join('data','seed','positives_seed.csv')
    pos_ids = set()
    if os.path.exists(pos_seed_path):
        import pandas as _pd
        pos_ids = set(_pd.read_csv(pos_seed_path)['uniprot_id'].astype(str).tolist())
    def p_at_k(scores, labels, k=20):
        import numpy as _np
        order=_np.argsort(-scores)
        y=_np.array(labels)[order]
        k=min(k,len(y))
        return float(_np.sum(y[:k]))/max(k,1)
    rows=[]
    for sc_pair in score_cols:
        name = sc_pair[1]
        sc = df[sc_pair[1]].to_numpy()
        y = [1 if (multi_state.get(uid, False) or uid in pos_ids) else 0 for uid in df['uniprot_id']]
        pa20 = p_at_k(sc,y,20)
        rows.append({'setting': name, 'P@20': pa20})
    abcsv = os.path.join(outdir, 'ablations.csv')
    pd.DataFrame(rows).to_csv(abcsv, index=False)
    print(f'Saved ablations to {abcsv}')


def cmd_evidence(args: argparse.Namespace):
    # Fetch and persist external evidence (PDBFlex, CoDNaS, optional CATH) for top-N
    import json
    import pandas as pd
    os.makedirs('data/external/pdbflex', exist_ok=True)
    os.makedirs('data/external/codnas', exist_ok=True)
    os.makedirs('data/external/cath', exist_ok=True)
    if not os.path.exists(args.topN_file):
        print(f'Top-N file not found: {args.topN_file}')
        return
    print(f"[evidence] Loading Top-N from {args.topN_file} ...", flush=True)
    df = pd.read_csv(args.topN_file)
    uids = [str(u) for u in df['uniprot_id'].tolist()]
    print(f"[evidence] {len(uids)} UniProt IDs to process", flush=True)
    # Prepare mapping and lists
    import csv as _csv
    map_path = os.path.join('data','external','pdb_ids_per_uniprot.tsv')
    lst_path = os.path.join('data','external','pdb_ids_topN.txt')
    os.makedirs(os.path.dirname(map_path), exist_ok=True)
    mapping = {}
    flat = []
    # also collect chain IDs (e.g., 1A22_B)
    from common.uniprot import fetch_pdb_chain_ids
    from common.pdbe import best_structures
    chain_map_uid = {}
    flat_chains = []
    print("[evidence] Building PDB and chain maps ...", flush=True)
    for idx, uid in enumerate(uids):
        pdbs = fetch_pdb_ids(uid)
        # If UniProt xrefs are sparse, try PDBe best_structures for better coverage
        if not pdbs:
            bs = best_structures(uid)
            pdbs = [p for p,_ in bs][:args.per_uniprot_pdbs] if bs else []
            # seed chain_map with PDBe chains
            if bs:
                chain_map_uid[uid] = [f"{p}_{c}" for p,c in bs]
        mapping[uid] = pdbs
        flat.extend(pdbs)
        # Merge with UniProt chain mapping
        uni_chains = fetch_pdb_chain_ids(uid)
        if chain_map_uid.get(uid):
            merged = sorted(set(chain_map_uid[uid] + uni_chains))
        else:
            merged = uni_chains
        chain_map_uid[uid] = merged
        flat_chains.extend(merged)
        if (idx+1) % 10 == 0 or (idx+1) == len(uids):
            print(f"  - mapped {idx+1}/{len(uids)} IDs", flush=True)
    with open(map_path, 'w', newline='') as f:
        w = _csv.writer(f, delimiter='\t')
        w.writerow(['uniprot_id','pdb_ids'])
        for uid in uids:
            w.writerow([uid, ';'.join(mapping.get(uid, []))])
    flat_u = sorted(set(p.upper() for p in flat))
    with open(lst_path, 'w') as f:
        for pid in flat_u:
            f.write(pid+'\n')
    # Write chain mapping and flat list for CoDNaS submissions
    chain_map_path = os.path.join('data','external','pdb_chain_ids_per_uniprot.tsv')
    chain_list_path = os.path.join('data','external','pdb_chain_ids_topN.txt')
    with open(chain_map_path, 'w', newline='') as f:
        w = _csv.writer(f, delimiter='\t')
        w.writerow(['uniprot_id','pdb_chain_ids'])
        for uid in uids:
            w.writerow([uid, ';'.join(sorted(set(chain_map_uid.get(uid, []))))])
    flat_chains_u = sorted(set(c for c in flat_chains if c))
    with open(chain_list_path, 'w') as f:
        for cid in flat_chains_u:
            f.write(cid+'\n')

    # Prepare FASTA for CATH by_sequence
    fa_path = os.path.join('data','external','cath','topN_sequences.fasta')
    with open(fa_path, 'w') as f:
        for uid in uids:
            fa = fetch_fasta(uid)
            if fa and fa.startswith('>'):
                seq = ''.join([line.strip() for line in fa.splitlines() if not line.startswith('>')])
                f.write(f'>{uid}\n{seq}\n')
    # If chains-only requested, skip remote evidence fetches
    if getattr(args, 'chains_only', False):
        print('Wrote chain lists for CoDNaS submissions:')
        print(' -', chain_map_path)
        print(' -', chain_list_path)
        print('Wrote PDB lists and CATH FASTA:')
        print(' -', map_path)
        print(' -', lst_path)
        print(' -', fa_path)
        return

    print("[evidence] Fetching per-ID evidence (PDBFlex/CoDNaS/CATH) ...", flush=True)
    for i, uid in enumerate(uids):
        # UniProt -> PDB IDs and chain IDs
        pdbs = mapping.get(uid, [])
        from common.uniprot import fetch_pdb_chain_ids
        chains = fetch_pdb_chain_ids(uid)
        chain_map = {}
        for cid in chains:
            if '_' in cid:
                p, c = cid.split('_', 1)
                chain_map.setdefault(p.upper(), []).append(c.upper())
        # PDBFlex per-PDB (limit per_uniprot)
        for pid in pdbs[: args.per_uniprot_pdbs]:
            saved = False
            for c in chain_map.get(pid.upper(), [])[:2]:
                js = fetch_pdbflex_stats(pid, c)
                if js:
                    outp = os.path.join('data','external','pdbflex', f'{pid.upper()}_{c}.json')
                    try:
                        with open(outp, 'w') as f:
                            json.dump(js, f)
                        saved = True
                    except Exception:
                        pass
            if not saved:
                js = fetch_pdbflex_stats(pid)
                if js:
                    outp = os.path.join('data','external','pdbflex', f'{pid.upper()}.json')
                    try:
                        with open(outp, 'w') as f:
                            json.dump(js, f)
                    except Exception:
                        pass
        # CoDNaS summary stub: save reachable list
        cod = fetch_codnas_summary(pdbs) if pdbs else {}
        if cod:
            outc = os.path.join('data','external','codnas', f'{uid}.json')
            with open(outc, 'w') as f:
                json.dump(cod, f)
        # Optional CATH domain ranges from FASTA
        if args.cath:
            fa = fetch_fasta(uid)
            if fa:
                try:
                    ranges = fetch_cath_domains_from_fasta(fa)
                except Exception:
                    ranges = []
                outf = os.path.join('data','external','cath', f'{uid}.json')
                with open(outf, 'w') as f:
                    json.dump({'ranges': ranges}, f)
        if (i+1) % 5 == 0 or (i+1) == len(uids):
            print(f"  - fetched evidence for {i+1}/{len(uids)} IDs", flush=True)
    # Summarize into evidence_top100.csv
    try:
        topN_path = args.topN_file
        if os.path.exists(topN_path):
            print(f"[evidence] Summarizing to evidence_top100.csv ...", flush=True)
            tdf = pd.read_csv(topN_path)
            # Ensure required columns
            for col in ('uniprot_id', 'block_1', 'block_2'):
                if col not in tdf.columns:
                    tdf[col] = ''
            # Enrich: SIFTS-projected PDB residue spans for PDBe best chain
            try:
                from common.pdbe import best_structures as _best, sifts_map as _sifts, project_uniprot_range_to_pdb as _proj
                bchains = []
                p1 = []
                p2 = []
                for _, rr in tdf.iterrows():
                    uid = str(rr.get('uniprot_id',''))
                    b1 = str(rr.get('block_1') or '0-0')
                    b2 = str(rr.get('block_2') or '0-0')
                    try:
                        s1,e1 = [int(x) for x in b1.split('-')]
                        s2,e2 = [int(x) for x in b2.split('-')]
                    except Exception:
                        s1=e1=s2=e2=0
                    bc=''; m1=''; m2=''
                    bs = _best(uid)
                    if bs:
                        pid,ch = bs[0]
                        bc = f"{pid}_{ch}"
                        maps = _sifts(uid, pid, ch)
                        if maps:
                            pr1 = _proj(s1,e1,maps)
                            pr2 = _proj(s2,e2,maps)
                            if pr1:
                                m1 = f"{pid}_{ch}:{pr1[0]}-{pr1[1]}"
                            if pr2:
                                m2 = f"{pid}_{ch}:{pr2[0]}-{pr2[1]}"
                    bchains.append(bc); p1.append(m1); p2.append(m2)
                tdf['pdb_best_chain'] = bchains
                tdf['pdb_block_1'] = p1
                tdf['pdb_block_2'] = p2
            except Exception:
                pass
            # Basic PDBFlex cache scan
            local_flex = {}
            try:
                local_flex = scan_local_stats()
            except Exception:
                local_flex = {}
            # CoDNaS raw integration (optional if raw TSVs exist)
            from common.codnas import parse_codnas_raw
            codnas_map = {}
            try:
                codnas_map = parse_codnas_raw()
            except Exception:
                codnas_map = {}
            rows = []
            for _, rr in tdf.iterrows():
                uid = str(rr['uniprot_id'])
                # Copy existing stats
                zq = rr['z_bcr_q'] if 'z_bcr_q' in rr else float('nan')
                b_eff = rr['bcr_q_ratio'] if 'bcr_q_ratio' in rr else (rr['bcr_q_effect'] if 'bcr_q_effect' in rr else float('nan'))
                # FDR per-table fallback if needed
                if 'q_bh' in tdf.columns:
                    q_bh_v = rr['q_bh']
                elif 'p_perm' in tdf.columns:
                    q_bh_v = rr['p_perm']
                else:
                    p = p_from_z(zq, two_sided=True) if (zq == zq) else float('nan')
                    q_bh_v = p  # will be replaced after vector BH
                # CATH IoU (sequence-based)
                from common.cath import fetch_cath_domains_from_fasta, iou_ranges
                cath_max = float('nan'); cath_mean = float('nan')
                try:
                    fa = fetch_fasta(uid)
                    if fa:
                        truth = fetch_cath_domains_from_fasta(fa)
                        # pred ranges from blocks (0-based half-open)
                        preds = []
                        try:
                            b1s, b1e = [int(x) for x in str(rr.get('block_1','')).split('-')]
                            preds.append((b1s, b1e))
                        except Exception:
                            pass
                        try:
                            b2s, b2e = [int(x) for x in str(rr.get('block_2','')).split('-')]
                            preds.append((b2s, b2e))
                        except Exception:
                            pass
                        if preds and truth:
                            # overall IoU across union
                            cath_union = iou_ranges(preds, truth)
                            # per-block IoU vs truth union
                            from common.cath import iou_ranges as _iou
                            per_block = []
                            for pr in preds:
                                per_block.append(_iou([pr], truth))
                            cath_max = float(max(per_block)) if per_block else float('nan')
                            cath_mean = float(sum([x for x in per_block if x == x]) / max(len([x for x in per_block if x == x]), 1)) if per_block else float('nan')
                            # If union IoU exists and max is NaN, backfill
                            if cath_max != cath_max and (cath_union == cath_union):
                                cath_max = float(cath_union)
                            if cath_mean != cath_mean and (cath_union == cath_union):
                                cath_mean = float(cath_union)
                except Exception:
                    pass
                # PDBFlex aggregation: try up to per-uniprot-pdbs chains
                try:
                    from common.uniprot import fetch_pdb_chain_ids
                    chains = fetch_pdb_chain_ids(uid)
                except Exception:
                    chains = []
                # limit number of chains
                chains = chains[: max(1, int(getattr(args, 'per_uniprot_pdbs', 3)))] if chains else []
                flex_label = None
                avg_rmsd_max = None
                max_rmsd_max = None
                for cid in chains:
                    try:
                        if '_' in cid:
                            pdb, ch = cid.split('_', 1)
                        else:
                            pdb, ch = cid, None
                        key = f"{pdb}_{ch}" if ch else pdb
                        # Prefer local cache if available for maxRMSD
                        ev = local_flex.get(key) or local_flex.get(pdb.upper())
                        if ev and (ev.get('max_rmsd') is not None):
                            mr = float(ev.get('max_rmsd'))
                            max_rmsd_max = mr if max_rmsd_max is None else max(max_rmsd_max, mr)
                        # Fetch live stats for label / avgRMSD if needed
                        st = fetch_pdbflex_stats(pdb, ch)
                        if st:
                            if st.get('avgRMSD') is not None:
                                ar = float(st['avgRMSD'])
                                avg_rmsd_max = ar if avg_rmsd_max is None else max(avg_rmsd_max, ar)
                            if st.get('maxRMSD') is not None:
                                mr2 = float(st['maxRMSD'])
                                max_rmsd_max = mr2 if max_rmsd_max is None else max(max_rmsd_max, mr2)
                            if st.get('flexibilityLabel'):
                                flex_label = st['flexibilityLabel']
                    except Exception:
                        continue
                # CoDNaS per-UniProt summary from chain evidence
                codnas_max = None
                codnas_pairs = 0.0
                for cid in chains:
                    evc = codnas_map.get(cid)
                    if evc:
                        if evc.get('max_rmsd') is not None:
                            codnas_max = evc['max_rmsd'] if codnas_max is None else max(codnas_max, evc['max_rmsd'])
                        if evc.get('pair_count') is not None:
                            codnas_pairs += float(evc['pair_count'])
                rows.append({
                    'uniprot_id': uid,
                    'z_bcr_q': zq,
                    'bcr_q_effect': b_eff,
                    'q_bh': q_bh_v,
                    'FDR_pass': (q_bh_v <= 0.01) if (q_bh_v == q_bh_v) else False,
                    'cath_iou_max': cath_max,
                    'cath_iou_mean': cath_mean,
                    'pdbflex_label': flex_label if flex_label is not None else '',
                    'pdbflex_avgRMSD_max': avg_rmsd_max,
                    'pdbflex_maxRMSD_max': max_rmsd_max,
                    'codnas_maxRMSD': codnas_max,
                    'codnas_pair_count': codnas_pairs if codnas_pairs else None,
                })
            edf = pd.DataFrame(rows)
            # If q_bh not present in topN, compute BH across topN rows (prefer p_perm over Z)
            if 'q_bh' not in tdf.columns:
                if 'p_perm' in edf.columns:
                    pvec = [float(p) if (p == p) else float('nan') for p in edf['p_perm'].tolist()]
                elif 'z_bcr_q' in edf.columns:
                    pvec = [p_from_z(z, two_sided=True) if (z == z) else float('nan') for z in edf['z_bcr_q'].tolist()]
                else:
                    pvec = [float('nan')] * len(edf)
                qvec = benjamini_hochberg(pvec)
                edf['q_bh'] = qvec
                edf['FDR_pass'] = edf['q_bh'].apply(lambda q: (q <= 0.05) if (q == q) else False)
            outcsv = os.path.join('results','bcrparts','evidence_top100.csv')
            edf.to_csv(outcsv, index=False)
            print(f"[evidence] Wrote {len(edf)} rows to {outcsv}", flush=True)
    except Exception:
        pass

    # Distribution plots (best-effort): CATH IoU and PDBFlex/CoDNaS RMSD
    try:
        import numpy as _np
        import matplotlib.pyplot as _plt
        evp = os.path.join('results','bcrparts','evidence_top100.csv')
        if os.path.exists(evp):
            edf = pd.read_csv(evp)
            figdir = os.path.join('results','bcrparts','figures')
            os.makedirs(figdir, exist_ok=True)
            print(f"[evidence] Generating figures in {figdir} ...", flush=True)
            # Scatter plots: BCR effect vs RMSD (Top-N)
            try:
                import numpy as _np
                import matplotlib.pyplot as _plt
                if 'bcr_q_effect' in edf.columns and 'pdbflex_maxRMSD_max' in edf.columns:
                    xs = _np.array(edf['bcr_q_effect'], dtype=float)
                    ys = _np.array(edf['pdbflex_maxRMSD_max'], dtype=float)
                    m = _np.isfinite(xs) & _np.isfinite(ys)
                    if _np.any(m):
                        _plt.figure(figsize=(4,3))
                        _plt.scatter(xs[m], ys[m], s=14, alpha=0.7)
                        _plt.xlabel('BCR effect (log-ratio)')
                        _plt.ylabel('PDBFlex maxRMSD (Å)')
                        _plt.tight_layout()
                        _plt.savefig(os.path.join(figdir, 'scatter_bcr_vs_pdbflex.png'), dpi=200)
                        _plt.close()
                if 'bcr_q_effect' in edf.columns and 'codnas_maxRMSD' in edf.columns:
                    xs = _np.array(edf['bcr_q_effect'], dtype=float)
                    ys = _np.array(edf['codnas_maxRMSD'], dtype=float)
                    m = _np.isfinite(xs) & _np.isfinite(ys)
                    if _np.any(m):
                        _plt.figure(figsize=(4,3))
                        _plt.scatter(xs[m], ys[m], s=14, alpha=0.7)
                        _plt.xlabel('BCR effect (log-ratio)')
                        _plt.ylabel('CoDNaS maxRMSD (Å)')
                        _plt.tight_layout()
                        _plt.savefig(os.path.join(figdir, 'scatter_bcr_vs_codnas.png'), dpi=200)
                        _plt.close()
            except Exception:
                pass
            # CATH IoU
            vals = [v for v in edf.get('cath_iou_mean', _np.array([])).tolist() if isinstance(v,(int,float)) and v==v]
            if vals:
                _plt.figure(figsize=(4,3))
                _plt.hist(vals, bins=10, range=(0,1), color='steelblue')
                _plt.xlabel('CATH IoU (mean)')
                _plt.ylabel('Count')
                _plt.tight_layout()
                _plt.savefig(os.path.join(figdir, 'cath_iou_hist.png'), dpi=200)
                _plt.close()
            # PDBFlex RMSD
            pf = [v for v in edf.get('pdbflex_maxRMSD_max', _np.array([])).tolist() if isinstance(v,(int,float)) and v==v]
            if pf:
                _plt.figure(figsize=(4,3))
                _plt.hist(pf, bins=12, color='darkorange')
                _plt.xlabel('PDBFlex maxRMSD (Å)')
                _plt.ylabel('Count')
                _plt.tight_layout()
                _plt.savefig(os.path.join(figdir, 'pdbflex_rmsd_hist.png'), dpi=200)
                _plt.close()
            # CoDNaS RMSD
            cf = [v for v in edf.get('codnas_maxRMSD', _np.array([])).tolist() if isinstance(v,(int,float)) and v==v]
            if cf:
                _plt.figure(figsize=(4,3))
                _plt.hist(cf, bins=12, color='seagreen')
                _plt.xlabel('CoDNaS maxRMSD (Å)')
                _plt.ylabel('Count')
                _plt.tight_layout()
                _plt.savefig(os.path.join(figdir, 'codnas_rmsd_hist.png'), dpi=200)
                _plt.close()
            # BCR effect histogram
            be = [v for v in edf.get('bcr_q_effect', _np.array([])).tolist() if isinstance(v,(int,float)) and v==v]
            if be:
                _plt.figure(figsize=(4,3))
                _plt.hist(be, bins=12, color='mediumpurple')
                _plt.xlabel('BCR effect (log-ratio)')
                _plt.ylabel('Count')
                _plt.tight_layout()
                _plt.savefig(os.path.join(figdir, 'bcr_effect_hist.png'), dpi=200)
                _plt.close()
            # Coverage summary (non-null counts)
            try:
                cov = {
                    'nonnull_cath_iou_mean': int(_np.isfinite(_np.array(edf.get('cath_iou_mean', _np.array([])), dtype=float)).sum()),
                    'nonnull_pdbflex': int(_np.isfinite(_np.array(edf.get('pdbflex_maxRMSD_max', _np.array([])), dtype=float)).sum()),
                    'nonnull_codnas': int(_np.isfinite(_np.array(edf.get('codnas_maxRMSD', _np.array([])), dtype=float)).sum()),
                }
                covdf = pd.DataFrame([cov])
                covdf.to_csv(os.path.join('results','bcrparts','coverage_summary.csv'), index=False)
                covdf.to_latex(os.path.join('results','bcrparts','coverage_summary.tex'), index=False)
            except Exception:
                pass
    except Exception:
        pass

    print('External evidence saved under data/external/{pdbflex,codnas,cath} and summarized to results/bcrparts/evidence_top100.csv')


def cmd_all(args: argparse.Namespace):
    ids = []
    if getattr(args, 'ids_file', ''):
        path = args.ids_file
        if os.path.exists(path):
            try:
                import pandas as _pd
                df = _pd.read_csv(path)
                if 'uniprot_id' in df.columns:
                    ids = [str(x) for x in df['uniprot_id'].tolist()]
                else:
                    with open(path) as f:
                        ids = [l.strip().split(',')[0] for l in f if l.strip()]
            except Exception:
                with open(path) as f:
                    ids = [l.strip().split(',')[0] for l in f if l.strip()]
            os.makedirs('data/ids', exist_ok=True)
            out = os.path.join('data','ids','bcrparts_ids.txt')
            with open(out, 'w') as f:
                for u in ids:
                    f.write(u + "\n")
        else:
            print(f"ids-file not found: {path}")
            ids = cmd_fetch(argparse.Namespace(ids=args.ids, limit=args.limit))
    else:
        ids = cmd_fetch(argparse.Namespace(ids=args.ids, limit=args.limit))
    run_ns = argparse.Namespace(
        ids_file=os.path.join('data', 'ids', 'bcrparts_ids.txt'),
        limit=None,
        n_perm=args.n_perm,
        figures=True,
        topN=args.topN,
        k=args.k,
        null_mode=args.null_mode,
        sym_mode=args.sym_mode,
        alpha=args.alpha,
        min_block_len=args.min_block_len,
        no_flags=True,
    )
    cmd_run(run_ns)
    cmd_paper(argparse.Namespace())


def main():
    p = argparse.ArgumentParser(prog='bcrparts')
    sub = p.add_subparsers(dest='cmd', required=True)
    # Register cohort CLI
    try:
        from .cohort_cli import register as _register_cohort
        _register_cohort(sub)
    except Exception:
        pass
    p_fetch = sub.add_parser('fetch')
    p_fetch.add_argument('--ids', type=str, help='Comma-separated IDs or path to file', default='')
    p_fetch.add_argument('--limit', type=int, default=50)
    p_fetch.add_argument('--proteome', type=str, default='', help='UniProt proteome ID, e.g., UP000000625')
    p_fetch.add_argument('--reviewed-only', action='store_true', help='Swiss-Prot reviewed only (hint for proteome queries)')
    p_fetch.add_argument('--download', action='store_true', help='Download/cache AFDB assets for IDs')
    p_fetch.add_argument('--concurrency', type=int, default=4)
    p_fetch.set_defaults(func=cmd_fetch)

    p_run = sub.add_parser('run')
    p_run.add_argument('--ids-file', type=str, default=os.path.join('data', 'ids', 'bcrparts_ids.txt'))
    p_run.add_argument('--limit', type=int, default=None)
    p_run.add_argument('--n-perm', type=int, default=1024)
    p_run.add_argument('--figures', action='store_true')
    p_run.add_argument('--topN', type=int, default=20)
    p_run.add_argument('--start', type=int, default=None, help='Start index into IDs (chunking)')
    p_run.add_argument('--count', type=int, default=None, help='Number of IDs to process (chunking)')
    p_run.add_argument('--k', type=str, default='auto', help='Comma-separated k values to try (1,2,3) or "auto"')
    p_run.add_argument('--null-mode', type=str, default='rotation', choices=['perm','rotation'])
    p_run.add_argument('--sym-mode', type=str, default='mean', choices=['mean','min','max','asym'])
    p_run.add_argument('--alpha', type=float, default=0.05, help='FDR alpha for BH correction')
    p_run.add_argument('--min-block-len', type=int, default=30, help='Minimum allowed block length (residues)')
    p_run.add_argument('--no-flags', action='store_true', help='Skip UniProt flag fetching to avoid network overhead')
    p_run.set_defaults(func=cmd_run)

    p_paper = sub.add_parser('paper')
    p_paper.set_defaults(func=cmd_paper)

    p_val = sub.add_parser('validate')
    p_val.add_argument('--sweep', action='store_true')
    p_val.set_defaults(func=cmd_validate)

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--cutoff-date', type=str, default='2019-01-01', help='Temporal split cutoff date (YYYY-MM-DD)')
    p_eval.set_defaults(func=cmd_eval)

    p_ab = sub.add_parser('ablate')
    p_ab.set_defaults(func=cmd_ablate)

    p_evi = sub.add_parser('evidence')
    p_evi.add_argument('--topN-file', type=str, default=os.path.join('results','bcrparts','topN.csv'))
    p_evi.add_argument('--per-uniprot-pdbs', type=int, default=3)
    p_evi.add_argument('--cath', action='store_true')
    p_evi.add_argument('--chains-only', action='store_true', help='Only write PDB chain ID lists; skip remote fetches')
    p_evi.set_defaults(func=cmd_evidence)

    p_bnegs = sub.add_parser('buildnegs')
    p_bnegs.add_argument('--ids-file', type=str, default=os.path.join('data','ids','bcrparts_ids.txt'))
    p_bnegs.add_argument('--limit', type=int, default=None)
    p_bnegs.add_argument('--plddt', type=float, default=80.0)
    p_bnegs.add_argument('--coverage', type=float, default=0.9, help='Min fraction with pLDDT>=threshold')
    p_bnegs.add_argument('--max', type=int, default=100)
    p_bnegs.set_defaults(func=cmd_buildnegs)

    p_all = sub.add_parser('all')
    p_all.add_argument('--ids', type=str, default='')
    p_all.add_argument('--ids-file', type=str, default='', help='CSV with column uniprot_id or plain list file')
    p_all.add_argument('--limit', type=int, default=50)
    p_all.add_argument('--n-perm', type=int, default=4096)
    p_all.add_argument('--topN', type=int, default=20)
    # shared run options
    # Attach shared options only to p_all to avoid duplicate definition on p_run
    p_all.add_argument('--k', type=str, default='auto', help='Comma-separated k values to try (1,2,3) or "auto"')
    p_all.add_argument('--null-mode', type=str, default='rotation', choices=['perm','rotation'])
    p_all.add_argument('--sym-mode', type=str, default='mean', choices=['mean','min','max','asym'])
    p_all.add_argument('--alpha', type=float, default=0.05, help='FDR alpha for BH correction')
    p_all.add_argument('--min-block-len', type=int, default=30, help='Minimum allowed block length (residues)')
    p_all.add_argument('--cath', action='store_true', help='Fetch CATH/Gene3D domains and compute IoU')
    p_all.set_defaults(func=cmd_all)

    args = p.parse_args()
    res = args.func(args)
    if isinstance(res, list):
        # printing already handled
        pass


if __name__ == '__main__':
    main()
