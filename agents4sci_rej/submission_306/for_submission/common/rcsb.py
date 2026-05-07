import datetime as dt
from typing import Dict, Iterable, Optional

import requests


RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{}"


def fetch_release_date(pdb_id: str, timeout: int = 20) -> Optional[str]:
    """Return YYYY-MM-DD release date for a PDB ID via RCSB REST or None."""
    try:
        r = requests.get(RCSB_ENTRY.format(pdb_id.lower()), timeout=timeout)
        if r.status_code != 200:
            return None
        js = r.json()
        info = js.get('rcsb_accession_info', {})
        d = info.get('initial_release_date') or info.get('deposit_date')
        return d
    except Exception:
        return None


def fetch_release_dates(pdb_ids: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for pid in pdb_ids:
        out[pid] = fetch_release_date(pid)
    return out


def split_by_date(pdb_ids: Iterable[str], cutoff: str = "2019-01-01") -> str:
    """Return 'eval' if any PDB for the set is >= cutoff date, else 'dev' if all < cutoff, else 'unknown'."""
    ids = list(pdb_ids)
    if not ids:
        return 'unknown'
    dates = [fetch_release_date(pid) for pid in ids]
    try:
        cut = dt.datetime.strptime(cutoff, "%Y-%m-%d")
    except Exception:
        cut = dt.datetime(2019,1,1)
    has_eval = False
    all_dev = True
    for d in dates:
        if not d:
            continue
        try:
            dd = dt.datetime.strptime(d[:10], "%Y-%m-%d")
        except Exception:
            continue
        if dd >= cut:
            has_eval = True
        else:
            # dev candidate
            pass
        if dd >= cut:
            all_dev = False
    if has_eval:
        return 'eval'
    if all_dev:
        return 'dev'
    return 'unknown'

