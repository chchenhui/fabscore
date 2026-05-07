from typing import Dict, Iterable, Optional, Tuple

import os
import csv
import requests


def fetch_codnas_summary(pdb_ids: Iterable[str], timeout: int = 20) -> Dict[str, Optional[str]]:
    """Best-effort CoDNaS/CoDNaS-Q fetch stub.

    CoDNaS does not expose a stable REST JSON API; this function is a
    placeholder that attempts a lightweight HEAD/GET against known pages and
    returns a minimal dict with textual evidence when available. Falls back to
    empty results if endpoints change or are unavailable.
    """
    out: Dict[str, Optional[str]] = {}
    ids = list({pid.upper() for pid in pdb_ids})
    for pid in ids:
        note = None
        try:
            # Try CoDNaS-Q custom download page (best-effort)
            # We avoid scraping; just confirm reachability.
            url = f"https://www.codnasq.dcc.uchile.cl/?pdbs={pid}"
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                note = "reachable"
        except Exception:
            note = None
        out[pid] = note
    return out


def parse_codnas_raw(raw_dir: str = 'data/external/codnas/raw') -> Dict[str, Dict[str, float]]:
    """Parse CoDNaS TSV/CSV downloads into a per-chain evidence mapping.

    Returns a dict: chain_id (e.g., '1A62_A') -> {'max_rmsd': float, 'pair_count': int}
    It looks for columns named like 'PDB_1','PDB_2' and an RMSD field ('Mammoth_RMS' or similar).
    """
    evidence: Dict[str, Dict[str, float]] = {}
    if not os.path.isdir(raw_dir):
        return evidence
    files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.tsv') or f.endswith('.csv')]
    for path in files:
        try:
            with open(path, 'r', newline='') as f:
                sample = f.read(2048)
                f.seek(0)
                # Heuristic dialect choice
                dialect = csv.excel_tab if '\t' in sample and (',' not in sample or sample.find('\t') < sample.find(',')) else csv.excel
                reader = csv.DictReader(f, dialect=dialect)
                # Normalize header keys to lower
                reader.fieldnames = [h.strip() if h else h for h in reader.fieldnames] if reader.fieldnames else reader.fieldnames
                for row in reader:
                    # chain fields
                    c1 = (row.get('PDB_1') or row.get('pdb_1') or row.get('pdb1') or '').strip().upper()
                    c2 = (row.get('PDB_2') or row.get('pdb_2') or row.get('pdb2') or '').strip().upper()
                    if not c1 or not c2:
                        continue
                    # parse RMSD
                    rms = None
                    for key in ('Mammoth_RMS','mammoth_rms','RMSD','rmsd'):
                        if key in row and row[key] not in (None, ''):
                            try:
                                rms = float(row[key])
                            except Exception:
                                rms = None
                            break
                    if rms is None:
                        continue
                    for cid in (c1, c2):
                        ev = evidence.setdefault(cid, {'max_rmsd': 0.0, 'pair_count': 0.0})
                        if rms > ev['max_rmsd']:
                            ev['max_rmsd'] = rms
                        ev['pair_count'] += 1.0
        except Exception:
            continue
    return evidence
