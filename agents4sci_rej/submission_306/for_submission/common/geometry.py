import math
from typing import Dict, List, Tuple, Optional

import numpy as np


def parse_ca_coords_from_pdb(path: str) -> Dict[int, np.ndarray]:
    coords: Dict[int, np.ndarray] = {}
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                if line[12:16].strip() != 'CA':
                    continue
                try:
                    res_seq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception:
                    continue
                coords[res_seq] = np.array([x, y, z], dtype=np.float32)
    except Exception:
        pass
    return coords


def principal_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Center
    P = points - points.mean(axis=0, keepdims=True)
    # SVD
    try:
        u, s, vh = np.linalg.svd(P, full_matrices=False)
        axis = vh[0]
        axis = axis / (np.linalg.norm(axis) + 1e-6)
        return axis.astype(np.float32)
    except Exception:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def angle_between_axes(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    ang = math.degrees(math.acos(c))
    # angles are symmetric: prefer acute (<90)
    if ang > 90:
        ang = 180 - ang
    return ang

def parse_backbone_from_pdb(path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """Parse N, CA, C, CB coordinates per residue when present.

    Returns mapping: res_index -> {'N': vec, 'CA': vec, 'C': vec, 'CB': vec?}
    """
    out: Dict[int, Dict[str, np.ndarray]] = {}
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                name = line[12:16].strip()
                if name not in ('N','CA','C','CB'):
                    continue
                try:
                    res_seq = int(line[22:26])
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except Exception:
                    continue
                out.setdefault(res_seq, {})[name] = np.array([x,y,z], dtype=np.float32)
    except Exception:
        pass
    return out

def ensure_cb(backbone: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, np.ndarray]:
    """Return Cβ coordinates per residue; approximate when missing (e.g., GLY) using N, CA, C.
    Approximation: cb = ca - normalize((n-ca)+(c-ca)) * 1.5 Å
    """
    cb: Dict[int, np.ndarray] = {}
    for idx, atoms in backbone.items():
        if 'CB' in atoms:
            cb[idx] = atoms['CB']
        elif all(k in atoms for k in ('N','CA','C')):
            n, ca, c = atoms['N'], atoms['CA'], atoms['C']
            v = (n - ca) + (c - ca)
            norm = np.linalg.norm(v) + 1e-6
            cb[idx] = ca - (v / norm) * 1.5
    return cb

def dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Return torsion angle (degrees) between (p1,p2,p3,p4)."""
    b0 = p2 - p1; b1 = p3 - p2; b2 = p4 - p3
    b1n = b1 / (np.linalg.norm(b1) + 1e-6)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    ang = math.degrees(math.atan2(y, x))
    return ang

def compute_nc_distance(ca_coords: Dict[int, np.ndarray], length: int) -> Optional[float]:
    if not ca_coords:
        return None
    a = ca_coords.get(1)
    b = ca_coords.get(length)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(b - a))

def compute_terminal_orientation(ca_coords: Dict[int, np.ndarray], length: int) -> Optional[float]:
    """Angle between terminal unit vectors using centroid windows.

    u^N: centroid(1..3) -> centroid(5..7)
    u^C: centroid(L-6..L-4) -> centroid(L-2..L)
    """
    if not ca_coords or length < 8:
        return None
    def centroid(indices: List[int]) -> Optional[np.ndarray]:
        pts = [ca_coords.get(i) for i in indices if i in ca_coords]
        if not pts:
            return None
        return np.stack(pts, axis=0).mean(axis=0)
    n1 = centroid([1,2,3]); n2 = centroid([5,6,7])
    c1 = centroid([length-6, length-5, length-4])
    c2 = centroid([length-2, length-1, length])
    if any(x is None for x in (n1,n2,c1,c2)):
        return None
    uN = n2 - n1; uC = c2 - c1
    if np.linalg.norm(uN) < 1e-6 or np.linalg.norm(uC) < 1e-6:
        return None
    return angle_between_axes(uN, uC)

def find_latch_pairs(ca_coords: Dict[int, np.ndarray], cb_coords: Dict[int, np.ndarray], block1: Tuple[int,int], block2: Tuple[int,int], top_k: int = 8) -> List[Tuple[int,int]]:
    """Identify candidate latch residue pairs between blocks.

    Criteria: 5.0–6.5 Å Cβ–Cβ distance and pseudo-dihedral near 180±30° using CA neighbors.
    Returns up to top_k pairs (1-based residue indices), prioritized by distance closeness to 5.75 Å.
    """
    s1,e1 = block1; s2,e2 = block2
    cand: List[Tuple[float,int,int]] = []  # (score, i, j)
    for i in range(s1, e1):
        ci = cb_coords.get(i+1)
        if ci is None:
            continue
        for j in range(s2, e2):
            cj = cb_coords.get(j+1)
            if cj is None:
                continue
            d = float(np.linalg.norm(cj - ci))
            if d < 5.0 or d > 6.5:
                continue
            # pseudo-dihedral using CA neighbors when available
            ca_im1 = ca_coords.get(i)  # (i-1)
            ca_i = ca_coords.get(i+1)
            ca_j = ca_coords.get(j+1)
            ca_jp1 = ca_coords.get(j+2)
            dih_ok = True
            if ca_im1 is not None and ca_i is not None and ca_j is not None and ca_jp1 is not None:
                ang = abs(dihedral(ca_im1, ca_i, ca_j, ca_jp1))
                # near-planar ~180±30 => ang in [150, 210]
                dih_ok = (150.0 <= ang <= 210.0)
            if not dih_ok:
                continue
            # score by closeness to 5.75 Å
            score = abs(d - 5.75)
            cand.append((score, i+1, j+1))
    cand.sort(key=lambda x: x[0])
    out = [(i,j) for _,i,j in cand[:top_k]]
    return out
