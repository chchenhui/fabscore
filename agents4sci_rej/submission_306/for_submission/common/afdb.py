import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import requests
except Exception:  # requests is optional; offline mode supported
    requests = None  # type: ignore
import urllib.request
import urllib.error


AFDB_LOCAL_DIR = "/data/afdb"
AFDB_MODELS_DIR = os.path.join(AFDB_LOCAL_DIR, "alphafold_v4")
AFDB_PAE_DIR = os.path.join(AFDB_LOCAL_DIR, "alpha_pae_v4")

# Cache root (can be overridden via env var AFDB_CACHE_DIR)
_CACHE_ROOT = os.environ.get("AFDB_CACHE_DIR", os.path.join("data", "cache", "afdb"))
CACHE_DIR = _CACHE_ROOT
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class AFEntry:
    uniprot_id: str
    length: int
    pae: np.ndarray  # (L, L) float32 in Angstrom
    plddt: np.ndarray  # (L,) float32 0..100
    source: str  # 'local' or 'remote' or 'cache'


def _find_local_files(uniprot_id: str) -> Tuple[Optional[str], Optional[str]]:
    base = f"AF-{uniprot_id}-F1"
    # PAE json
    pae_json = os.path.join(AFDB_PAE_DIR, f"{base}-predicted_aligned_error_v4.json")
    # Model file (pdb preferred)
    pdb = os.path.join(AFDB_MODELS_DIR, f"{base}-model_v4.pdb")
    cif = os.path.join(AFDB_MODELS_DIR, f"{base}-model_v4.cif")
    model = pdb if os.path.exists(pdb) else (cif if os.path.exists(cif) else None)
    return (pae_json if os.path.exists(pae_json) else None, model)


def _cache_paths(uniprot_id: str) -> Tuple[str, str]:
    return (
        os.path.join(CACHE_DIR, f"{uniprot_id}_pae.json"),
        os.path.join(CACHE_DIR, f"{uniprot_id}_model.pdb"),
    )


def _find_existing_cache_any(uniprot_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Search for cached PAE/model across known locations (project cache and past run folders)."""
    # Primary project cache
    c_pae, c_model = _cache_paths(uniprot_id)
    if os.path.exists(c_pae) and os.path.exists(c_model):
        return c_pae, c_model
    # Past runs: runs/*/data/cache/afdb
    runs_root = os.path.join("runs")
    if os.path.isdir(runs_root):
        try:
            for run in os.listdir(runs_root):
                cand = os.path.join(runs_root, run, "data", "cache", "afdb")
                if not os.path.isdir(cand):
                    continue
                p = os.path.join(cand, f"{uniprot_id}_pae.json")
                m = os.path.join(cand, f"{uniprot_id}_model.pdb")
                if os.path.exists(p) and os.path.exists(m):
                    return p, m
        except Exception:
            pass
    return None, None


def _fetch_remote(uniprot_id: str, timeout: float = 30.0) -> Tuple[Optional[str], Optional[str]]:
    """Fetch via AFDB API to cache. Returns cached file paths or (None, None) on failure."""
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    pae_path, model_path = _cache_paths(uniprot_id)
    try:
        # Prefer requests if available; otherwise fall back to urllib
        if requests is not None:
            r = requests.get(api, timeout=timeout)
            if r.status_code != 200:
                return (None, None)
            arr = r.json()
        else:
            with urllib.request.urlopen(api, timeout=timeout) as resp:
                import json as _json
                arr = _json.loads(resp.read().decode('utf-8'))
        if not arr:
            return (None, None)
        meta = arr[0]
        pae_url = (
            meta.get("paeDoc", {}).get("url")
            or meta.get("paeUrl")
            or meta.get("paeDocUrl")
        )
        pdb_url = meta.get("pdbUrl") or meta.get("cifUrl") or meta.get("bcifUrl")

        if pae_url and not os.path.exists(pae_path):
            try:
                if requests is not None:
                    pr = requests.get(pae_url, timeout=timeout)
                    if pr.status_code == 200:
                        with open(pae_path, "wb") as f:
                            f.write(pr.content)
                else:
                    with urllib.request.urlopen(pae_url, timeout=timeout) as resp:
                        with open(pae_path, "wb") as f:
                            f.write(resp.read())
            except Exception:
                pass
        if pdb_url and not os.path.exists(model_path):
            try:
                if requests is not None:
                    mr = requests.get(pdb_url, timeout=timeout)
                    if mr.status_code == 200:
                        with open(model_path, "wb") as f:
                            f.write(mr.content)
                else:
                    with urllib.request.urlopen(pdb_url, timeout=timeout) as resp:
                        with open(model_path, "wb") as f:
                            f.write(resp.read())
            except Exception:
                pass
        return (
            pae_path if os.path.exists(pae_path) else None,
            model_path if os.path.exists(model_path) else None,
        )
    except Exception:
        return (None, None)


def _parse_pae_json(path: str) -> np.ndarray:
    with open(path, "r") as f:
        obj = json.load(f)
    # Common formats:
    # - {"predicted_aligned_error": [[...], ...]}
    # - [{"predicted_aligned_error": [[...], ...]}, ...] (AFDB API bundle)
    # - flat list of length L*L
    if isinstance(obj, dict) and "predicted_aligned_error" in obj:
        pae = np.array(obj["predicted_aligned_error"], dtype=np.float32)
    elif isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], dict) and "predicted_aligned_error" in obj[0]:
            pae = np.array(obj[0]["predicted_aligned_error"], dtype=np.float32)
        else:
            # some PAE JSON formats are a flat list length L*L
            L = int(np.sqrt(len(obj)))
            pae = np.array(obj, dtype=np.float32).reshape(L, L)
    else:
        raise ValueError("Unrecognized PAE JSON format")
    return pae


def _parse_plddt_from_pdb(path: str) -> Tuple[np.ndarray, int]:
    """Parse pLDDT from B-factor column in PDB. Returns (plddt, length)."""
    plddts: Dict[int, float] = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                res_seq = int(line[22:26])
                bfac = float(line[60:66])
            except Exception:
                continue
            # pLDDT stored as B-factor
            plddts[res_seq] = bfac
    if not plddts:
        return (np.array([], dtype=np.float32), 0)
    max_idx = max(plddts)
    arr = np.zeros(max_idx, dtype=np.float32)
    for i in range(1, max_idx + 1):
        arr[i - 1] = plddts.get(i, np.nan)
    return (arr, max_idx)


def load_entry(uniprot_id: str, prefer_local: bool = True) -> Optional[AFEntry]:
    """Load AFDB entry from local dir or cache/remote."""
    pae_json = None
    model_path = None
    source = ""
    if prefer_local:
        pae_json, model_path = _find_local_files(uniprot_id)
        if pae_json:
            source = "local"
    if not pae_json or not model_path:
        # check any cache (project cache or past runs)
        c_pae, c_model = _find_existing_cache_any(uniprot_id)
        if c_pae and c_model:
            pae_json, model_path, source = c_pae, c_model, "cache"
        else:
            # remote fetch
            r_pae, r_model = _fetch_remote(uniprot_id)
            if r_pae and r_model:
                pae_json, model_path, source = r_pae, r_model, "remote"
    if not pae_json:
        return None
    pae = _parse_pae_json(pae_json)
    plddt = np.array([], dtype=np.float32)
    length = 0
    if model_path and os.path.exists(model_path):
        plddt, length = _parse_plddt_from_pdb(model_path)
    # If pLDDT missing, fill with NaNs to match PAE size
    L = pae.shape[0]
    if plddt.size == 0:
        plddt = np.full(L, np.nan, dtype=np.float32)
    elif plddt.size != L:
        # truncate or pad
        if plddt.size > L:
            plddt = plddt[:L]
        else:
            pad = np.full(L - plddt.size, np.nan, dtype=np.float32)
            plddt = np.concatenate([plddt, pad])
    return AFEntry(uniprot_id=uniprot_id, length=L, pae=pae, plddt=plddt.astype(np.float32), source=source or "local")


def list_local_uniprot_ids(limit: Optional[int] = None) -> List[str]:
    ids: List[str] = []
    if not os.path.isdir(AFDB_PAE_DIR):
        return ids
    for name in os.listdir(AFDB_PAE_DIR):
        if not name.endswith("_v4.json"):
            continue
        m = re.match(r"AF-([A-Za-z0-9]+)-F1-predicted_aligned_error_v4.json", name)
        if m:
            ids.append(m.group(1))
    ids.sort()
    if limit:
        ids = ids[:limit]
    return ids


def symmetrize_pae(pae: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Return PAE processed according to symmetrization `mode` and clipped to [0, 31.75] Å.

    - mean: (pae + pae.T) / 2 (default)
    - min:  min(pae, pae.T)
    - max:  max(pae, pae.T)
    - asym: return pae unchanged (still clipped)
    """
    mode = (mode or "mean").lower()
    if mode == "min":
        pae_sym = np.minimum(pae, pae.T).astype(np.float32, copy=False)
    elif mode == "max":
        pae_sym = np.maximum(pae, pae.T).astype(np.float32, copy=False)
    elif mode == "asym":
        pae_sym = pae.astype(np.float32, copy=False)
    else:
        pae_sym = ((pae + pae.T) / 2.0).astype(np.float32)
    # Expected PAE range is [0, 31.75] Å in AFDB
    return np.clip(pae_sym, 0.0, 31.75, out=pae_sym)


def pae_asymmetry(pae: np.ndarray, delta: int = 7) -> float:
    """Return a simple asymmetry metric: mean absolute (PAE - PAE^T) off-diagonal beyond `delta`."""
    L = pae.shape[0]
    m = mask_near_diagonal(L, delta)
    diff = np.abs(pae.astype(np.float32) - pae.T.astype(np.float32))
    if not np.any(m):
        return float(np.mean(diff))
    return float(np.mean(diff[m]))


def mask_near_diagonal(L: int, delta: int = 7) -> np.ndarray:
    """Return boolean mask of shape (L, L) marking pairs with |i-j| > delta."""
    i = np.arange(L)[:, None]
    j = np.arange(L)[None, :]
    return (np.abs(i - j) > delta)


def trimmed_mean(arr: np.ndarray, trim: float = 0.1) -> float:
    if arr.size == 0:
        return float("nan")
    arr = np.sort(arr.reshape(-1))
    n = arr.size
    k = int(n * trim)
    if 2 * k >= n:
        return float(np.mean(arr))
    return float(np.mean(arr[k:n - k]))
