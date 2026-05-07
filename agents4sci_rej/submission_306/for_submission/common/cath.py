import time
from typing import List, Optional, Tuple

import requests


SUBMIT_URL = "https://www.cathdb.info/search/by_funfhmmer"
CHECK_URL = "https://www.cathdb.info/search/by_funfhmmer/check/{}"
RES_URL = "https://www.cathdb.info/search/by_funfhmmer/results/{}"


def fetch_cath_domains_from_fasta(fasta_text: str, timeout: int = 30, poll_interval: float = 2.0, max_wait: float = 60.0) -> List[Tuple[int, int]]:
    """Submit FASTA to CATH/Gene3D FunFam HMMER and return domain ranges as [(start,end_exclusive)].

    Implements the documented flow:
      POST /search/by_funfhmmer with form key 'fasta'
      GET  /search/by_funfhmmer/check/{task_id} until status=='done'
      GET  /search/by_funfhmmer/results/{task_id} and parse signatures_by_id -> matches -> regions
    """
    try:
        headers = {'Accept': 'application/json'}
        # Submit sequence
        r = requests.post(SUBMIT_URL, data={'fasta': fasta_text}, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return []
        task = None
        try:
            j = r.json(); task = j.get('task_id')
        except Exception:
            task = None
        if not task:
            return []
        # Poll
        waited = 0.0
        done = False
        while waited < max_wait:
            cr = requests.get(CHECK_URL.format(task), headers=headers, timeout=timeout)
            if cr.status_code == 200:
                try:
                    cj = cr.json()
                    status = (cj.get('data') or {}).get('status') or cj.get('message')
                    if str(status).lower() in ('done','finished','success'):
                        done = True
                        break
                except Exception:
                    pass
            time.sleep(poll_interval)
            waited += poll_interval
        if not done:
            return []
        # Results
        rr = requests.get(RES_URL.format(task), headers=headers, timeout=timeout)
        if rr.status_code != 200:
            return []
        data = rr.json()
        ranges: List[Tuple[int,int]] = []

        # Robust parse: traverse nested dicts/lists and collect any start/end pairs
        def _collect(obj):
            if isinstance(obj, dict):
                if 'start' in obj and 'end' in obj:
                    try:
                        s1 = int(obj.get('start'))
                        e1 = int(obj.get('end'))
                        # CATH regions are 1-based inclusive; convert to 0-based half-open
                        if s1 > 0 and e1 >= s1:
                            s0 = s1 - 1
                            e0 = e1  # inclusive -> half-open
                            ranges.append((s0, e0))
                    except Exception:
                        pass
                for v in obj.values():
                    _collect(v)
            elif isinstance(obj, list):
                for v in obj:
                    _collect(v)

        _collect(data.get('signatures_by_id') or data)
        # de-duplicate and sort
        ranges = sorted(set(ranges))
        return ranges
    except Exception:
        return []


def iou_ranges(pred: List[Tuple[int,int]], truth: List[Tuple[int,int]]) -> float:
    def _intervals_to_mask(ranges: List[Tuple[int,int]]) -> set:
        s = set()
        for a,b in ranges:
            s.update(range(a, b))
        return s
    p = _intervals_to_mask(pred)
    t = _intervals_to_mask(truth)
    if not p or not t:
        return float('nan')
    inter = len(p & t)
    union = len(p | t)
    return inter / max(union, 1)
