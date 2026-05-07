from typing import Dict, Optional, List, Tuple

import os
import csv
import requests


BASE = "http://pdbflex.org/php/api"


def fetch_pdbflex_stats(pdb_id: str, chain_id: Optional[str] = None, timeout: int = 20) -> Optional[dict]:
    """Fetch PDBFlex PDBStats for a PDB (optionally a specific chain).

    Endpoint: PDBStats.php?pdbID=1a50&chainID=A
    Returns JSON list with fields including parentClusterID, maxRMSD, otherClusterMembers.
    """
    params = {'pdbID': str(pdb_id).lower()}
    if chain_id:
        params['chainID'] = chain_id
    try:
        r = requests.get(f"{BASE}/PDBStats.php", params=params, headers={"Accept":"application/json"}, timeout=timeout)
        if r.status_code != 200:
            return None
        js = r.json()
        # expected list; use first element
        if isinstance(js, list) and js:
            return js[0]
        if isinstance(js, dict):
            return js
    except Exception:
        return None
    return None


def fetch_pdbflex_representatives(pdb_id: str, chain_id: Optional[str] = None, timeout: int = 20) -> List[str]:
    params = {'pdbID': str(pdb_id).lower()}
    if chain_id:
        params['chainID'] = chain_id
    try:
        r = requests.get(f"{BASE}/representatives.php", params=params, headers={"Accept":"application/json"}, timeout=timeout)
        if r.status_code != 200:
            return []
        js = r.json()
        if isinstance(js, list):
            return [str(x) for x in js]
    except Exception:
        return []
    return []


def load_pdbflex_evidence(raw_dir: str = 'data/external/pdbflex/raw') -> Dict[str, Dict[str, Optional[float]]]:
    """Load PDBFlex evidence from user-provided TSV/CSV files.

    Expected columns (case-insensitive best-effort):
      - pdb_id
      - max_rmsd (optional)
      - cluster_size (optional)

    Returns dict mapping PDB_ID -> {'max_rmsd': float|None, 'cluster_size': int|None}.
    """
    evidence: Dict[str, Dict[str, Optional[float]]] = {}
    if not os.path.isdir(raw_dir):
        return evidence
    for name in os.listdir(raw_dir):
        if not (name.endswith('.tsv') or name.endswith('.csv')):
            continue
        path = os.path.join(raw_dir, name)
        with open(path, 'r', newline='') as f:
            sn = csv.Sniffer()
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.excel_tab if '\t' in sample and ',' not in sample else csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                pdb = (row.get('pdb_id') or row.get('PDB') or row.get('pdb') or '').strip().upper()
                if not pdb:
                    continue
                try:
                    mr = row.get('max_rmsd') or row.get('maxRmsd') or row.get('MaxRMSD')
                    max_rmsd = float(mr) if mr not in (None, '') else None
                except Exception:
                    max_rmsd = None
                try:
                    cs = row.get('cluster_size') or row.get('size')
                    cluster_size = int(cs) if cs not in (None, '') else None
                except Exception:
                    cluster_size = None
                evidence[pdb] = {'max_rmsd': max_rmsd, 'cluster_size': cluster_size}
    return evidence


def scan_local_stats(dir_path: str = 'data/external/pdbflex') -> Dict[str, Dict[str, Optional[float]]]:
    """Scan saved PDBFlex JSON stats in data/external/pdbflex and summarize per PDB or PDB_CHAIN.
    Returns mapping id -> {'max_rmsd': float|None, 'cluster_size': int|None}
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if not os.path.isdir(dir_path):
        return out
    for name in os.listdir(dir_path):
        if not name.lower().endswith('.json'):
            continue
        path = os.path.join(dir_path, name)
        try:
            import json
            js = json.load(open(path))
        except Exception:
            continue
        # id from filename without .json
        key = os.path.splitext(name)[0]
        mr = None; cs = None
        if isinstance(js, dict):
            try:
                mr = float(js.get('maxRMSD') or js.get('max_rmsd')) if (js.get('maxRMSD') or js.get('max_rmsd')) is not None else None
            except Exception:
                mr = None
            # otherClusterMembers list size approximates cluster size
            mem = js.get('otherClusterMembers')
            if isinstance(mem, list):
                cs = len(mem)
        out[key] = {'max_rmsd': mr, 'cluster_size': cs}
    return out
