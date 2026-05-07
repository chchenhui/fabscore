from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import time
import requests
import os
from .cache import cache_path, load_json, save_json


BEST_STRUCT_URL = "https://www.ebi.ac.uk/pdbe/graph-api/uniprot/best_structures/{}"
SIFTS_MAP_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{}"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "User-Agent": "agents4sci/0.1"})
    return s


def _get_json(url: str, timeout: int = 20, retries: int = 3, backoff: float = 0.5, cache_key: Optional[str] = None) -> Optional[dict]:
    # Simple file cache
    if cache_key:
        cdir = os.path.join('data','cache','pdbe')
        cpath = cache_path(cdir, f"{cache_key}.json")
        js = load_json(cpath)
        if js is not None:
            return js
    s = _session()
    for i in range(max(1, retries)):
        try:
            r = s.get(url, timeout=timeout)
            if r.status_code == 200:
                js = r.json()
                if cache_key:
                    save_json(cache_path(os.path.join('data','cache','pdbe'), f"{cache_key}.json"), js)
                return js
            if r.status_code in (429, 500, 502, 503):
                time.sleep(backoff * (2 ** i))
                continue
            return None
        except Exception:
            time.sleep(backoff * (2 ** i))
    return None


def best_structures(uniprot_id: str, timeout: int = 20) -> List[Tuple[str, str]]:
    """Return a list of (pdb_id, chain_id) tuples for representative structures.

    Uses PDBe Graph API 'best_structures'. Falls back to empty list on error.
    """
    try:
        js = _get_json(BEST_STRUCT_URL.format(uniprot_id), timeout=timeout, cache_key=f"best_structures_{uniprot_id}")
        if not js:
            return []
        # Response shape: { UNIPROT: [ {"pdb_id":"...","chain_id":"...", ...}, ... ] }
        arr = None
        if isinstance(js, dict):
            arr = js.get(uniprot_id) or next((v for v in js.values() if isinstance(v, list)), None)
        elif isinstance(js, list):
            arr = js
        if not isinstance(arr, list):
            return []
        out: List[Tuple[str, str]] = []
        for it in arr:
            if not isinstance(it, dict):
                continue
            pdb = str(it.get("pdb_id") or "").upper()
            ch = str(it.get("chain_id") or "").upper()
            if pdb and ch:
                out.append((pdb, ch))
        # de-duplicate while preserving order
        seen = set(); uniq: List[Tuple[str,str]] = []
        for t in out:
            if t not in seen:
                seen.add(t); uniq.append(t)
        return uniq
    except Exception:
        return []


def sifts_map(uniprot_id: str, pdb_id: str, chain_id: str, timeout: int = 20) -> List[Dict[str, int]]:
    """Return residue-range mappings between UniProt and a specific PDB chain.

    Uses PDBe SIFTS endpoint /api/mappings/uniprot/{pdb_id} and filters for the
    provided UniProt accession and chain. Returns a list of dicts each having:
      {'unp_start','unp_end','pdb_start','pdb_end'} (UniProt are 1-based inclusive,
       PDB author residue numbers, inclusive). Falls back to empty list.
    """
    try:
        pdb_id = str(pdb_id).lower()
        chain_id = str(chain_id).upper()
        js = _get_json(SIFTS_MAP_URL.format(pdb_id), timeout=timeout, cache_key=f"sifts_{pdb_id}")
        if not js or pdb_id not in js:
            return []
        entry = js[pdb_id]
        # entry structure: {uniprotAcc: {'mappings':[{'start':{'author_residue_number', 'chain_id', ...}, 'end':{...}, 'uniprot_start','uniprot_end'}]}}
        out: List[Dict[str,int]] = []
        for acc, obj in entry.items():
            if str(acc).upper() != str(uniprot_id).upper():
                continue
            maps = obj.get('mappings') or []
            for m in maps:
                st = m.get('start') or {}
                en = m.get('end') or {}
                ch = str(st.get('chain_id') or en.get('chain_id') or '').upper()
                if chain_id and ch != chain_id:
                    continue
                try:
                    unp_s = int(m.get('uniprot_start'))
                    unp_e = int(m.get('uniprot_end'))
                    pdb_s = int(st.get('author_residue_number') or st.get('residue_number'))
                    pdb_e = int(en.get('author_residue_number') or en.get('residue_number'))
                except Exception:
                    continue
                out.append({'unp_start': unp_s, 'unp_end': unp_e, 'pdb_start': pdb_s, 'pdb_end': pdb_e})
        # merge overlapping UniProt ranges to continuous PDB spans
        out.sort(key=lambda d: (d['unp_start'], d['unp_end']))
        merged: List[Dict[str,int]] = []
        for m in out:
            if not merged:
                merged.append(m)
                continue
            prev = merged[-1]
            if m['unp_start'] <= prev['unp_end'] + 1:
                prev['unp_end'] = max(prev['unp_end'], m['unp_end'])
                prev['pdb_start'] = min(prev['pdb_start'], m['pdb_start'])
                prev['pdb_end'] = max(prev['pdb_end'], m['pdb_end'])
            else:
                merged.append(m)
        return merged
    except Exception:
        return []


def coverage_stats(uniprot_id: str, timeout: int = 20) -> Dict[str, Optional[float]]:
    """Compute coverage stats via PDBe best_structures for a UniProt ID.

    Returns dict with keys: num_pdb_structures, min_resolution, distinct_methods
    """
    try:
        js = _get_json(BEST_STRUCT_URL.format(uniprot_id), timeout=timeout, cache_key=f"best_structures_{uniprot_id}")
        if not js:
            return {'num_pdb_structures': 0, 'min_resolution': None, 'distinct_methods': 0}
        arr = None
        if isinstance(js, dict):
            arr = js.get(uniprot_id) or next((v for v in js.values() if isinstance(v, list)), None)
        elif isinstance(js, list):
            arr = js
        if not isinstance(arr, list):
            return {'num_pdb_structures': 0, 'min_resolution': None, 'distinct_methods': 0}
        n = 0
        min_res = None
        methods = set()
        for it in arr:
            if not isinstance(it, dict):
                continue
            n += 1
            # PDBe Graph may provide 'resolution' and 'experimental_method'
            try:
                res = it.get('resolution')
                if res is not None:
                    val = float(res)
                    min_res = val if min_res is None else min(min_res, val)
            except Exception:
                pass
            em = it.get('experimental_method') or it.get('experimental_methods')
            if isinstance(em, list):
                for m in em:
                    if m:
                        methods.add(str(m))
            elif isinstance(em, str):
                methods.add(em)
        return {'num_pdb_structures': n, 'min_resolution': min_res, 'distinct_methods': len(methods)}
    except Exception:
        return {'num_pdb_structures': 0, 'min_resolution': None, 'distinct_methods': 0}


def project_uniprot_range_to_pdb(unp_start0: int, unp_end0: int, mappings: List[Dict[str,int]]) -> Optional[Tuple[int,int]]:
    """Project a UniProt 0-based half-open range to an approximate PDB residue span.

    Returns (pdb_start, pdb_end) inclusive if any overlap exists, else None.
    """
    if unp_end0 <= unp_start0:
        return None
    # convert to 1-based inclusive
    u_s = unp_start0 + 1
    u_e = unp_end0
    pdb_min = None
    pdb_max = None
    for m in mappings:
        # overlap test with 1-based [u_s, u_e]
        a = max(u_s, m['unp_start'])
        b = min(u_e, m['unp_end'])
        if a <= b:
            if pdb_min is None:
                pdb_min = m['pdb_start']
                pdb_max = m['pdb_end']
            else:
                pdb_min = min(pdb_min, m['pdb_start'])
                pdb_max = max(pdb_max, m['pdb_end'])
    if pdb_min is None or pdb_max is None:
        return None
    return (int(pdb_min), int(pdb_max))
