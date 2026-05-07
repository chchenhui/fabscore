import os
import argparse
from typing import List, Dict

import pandas as pd

from common.uniprot import fetch_proteome_accessions
from common.pdbe import best_structures, coverage_stats


def _load_ids_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if 'uniprot_id' in df.columns:
            return [str(x) for x in df['uniprot_id'].tolist()]
    except Exception:
        pass
    with open(path) as f:
        return [l.strip().split(',')[0] for l in f if l.strip()]


def cmd_cohort(args: argparse.Namespace) -> str:
    ids: List[str] = []
    if args.ids_file:
        ids = _load_ids_from_file(args.ids_file)
    elif args.proteome:
        # Note: fetch_proteome_accessions returns all; reviewed-only filter handled by UniProt query ideally
        ids = fetch_proteome_accessions(args.proteome)
    else:
        return ''
    print(f"[cohort] Loaded {len(ids)} candidate UniProt IDs", flush=True)
    rows: List[Dict] = []
    for i, uid in enumerate(ids):
        cov = coverage_stats(uid)
        n = cov.get('num_pdb_structures') or 0
        min_res = cov.get('min_resolution')
        methods = cov.get('distinct_methods') or 0
        keep = (n >= args.min_pdb)
        if args.max_resolution is not None and min_res is not None:
            keep = keep and (min_res <= args.max_resolution)
        if keep:
            rows.append({
                'uniprot_id': uid,
                'num_pdb_structures': n,
                'min_resolution': min_res,
                'distinct_methods': methods,
            })
        if (i+1) % 50 == 0 or (i+1) == len(ids):
            print(f"  - processed {i+1}/{len(ids)} (kept {len(rows)})", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[cohort] Wrote {len(df)} rows to {args.out}", flush=True)
    return args.out


def register(sub):
    p = sub.add_parser('cohort')
    p.add_argument('--proteome', type=str, default='')
    p.add_argument('--ids-file', type=str, default='')
    p.add_argument('--reviewed-only', action='store_true', help='Swiss-Prot reviewed only (not enforced if unresolved)')
    p.add_argument('--min-pdb', type=int, default=2)
    p.add_argument('--max-resolution', type=float, default=3.5)
    p.add_argument('--out', type=str, default=os.path.join('cohorts','evidence_ready.csv'))
    p.set_defaults(func=cmd_cohort)

