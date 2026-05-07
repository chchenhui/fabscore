from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


@dataclass
class Blocks:
    blocks: List[Tuple[int, int]]  # inclusive start, exclusive end [(s,e), ...]

    def to_mask(self, L: int) -> np.ndarray:
        mask = np.zeros((L,), dtype=bool)
        for s, e in self.blocks:
            mask[s:e] = True
        return mask


def plddt_blocks(plddt: np.ndarray, threshold: float = 70.0, min_len: int = 30) -> Blocks:
    """Contiguous segments with pLDDT >= threshold."""
    L = plddt.size
    mask = np.isfinite(plddt) & (plddt >= threshold)
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < L:
        if mask[i]:
            j = i
            while j < L and mask[j]:
                j += 1
            if j - i >= min_len:
                blocks.append((i, j))
            i = j
        else:
            i += 1
    return Blocks(blocks)


def spectral_bipartition_from_pae(pae: np.ndarray, sigma: float = 10.0) -> Blocks:
    """Split residues into two contiguous blocks via spectral sign of the Fiedler vector.

    Applies a light contiguity heuristic and guards against degenerate results.
    """
    L = pae.shape[0]
    # Guard PAE magnitude and numerical stability
    pae = np.clip(pae.astype(np.float32, copy=False), 0.0, 31.75)
    # Build similarity matrix; avoid diag dominance
    S = np.exp(-pae / float(sigma))
    np.fill_diagonal(S, 0.0)
    # Graph Laplacian
    Lmat = csgraph.laplacian(S, normed=True)
    # Smallest non-zero eigenvector (Fiedler)
    # Use eigsh on symmetric matrix
    try:
        vals, vecs = eigsh(Lmat, k=2, which="SM")
        fiedler = vecs[:, 1]
    except Exception:
        # fallback: random split
        idx = np.arange(L)
        mid = L // 2
        return Blocks([(0, mid), (mid, L)])
    sign = fiedler >= 0
    # Convert sign vector to contiguous runs
    runs: List[Tuple[int, int, bool]] = []  # (start, end, label)
    i = 0
    while i < L:
        lbl = bool(sign[i])
        j = i + 1
        while j < L and bool(sign[j]) == lbl:
            j += 1
        runs.append((i, j, lbl))
        i = j
    # If only one run, split at midpoint
    if len(runs) == 1:
        mid = L // 2
        return Blocks([(0, mid), (mid, L)])
    # Merge runs of the same label that are separated by tiny gaps (<=2 residues)
    merged: List[Tuple[int, int, bool]] = []
    for s, e, lbl in runs:
        if not merged:
            merged.append((s, e, lbl))
            continue
        ps, pe, pl = merged[-1]
        if lbl == pl and (s - pe) <= 2:
            merged[-1] = (ps, e, pl)
        else:
            merged.append((s, e, lbl))
    # Choose cut between two largest contiguous segments with different labels
    # Heuristic: find the largest boundary (gap) between adjacent segments of different labels
    best_cut = None
    best_span = -1
    for a, b in zip(merged[:-1], merged[1:]):
        if a[2] == b[2]:
            continue
        span = (a[1] - a[0]) + (b[1] - b[0])
        cut = (a[1], b[0])
        if span > best_span:
            best_span = span
            best_cut = cut
    if not best_cut:
        mid = L // 2
        return Blocks([(0, mid), (mid, L)])
    c0, c1 = best_cut
    c0 = max(0, min(c0, L))
    c1 = max(0, min(c1, L))
    # Return two contiguous blocks
    return Blocks(sorted([(0, c0), (c1, L)], key=lambda x: x[0]))


def partition_k_spectral(pae: np.ndarray, k: int = 2, sigma: float = 10.0, min_block_len: int = 30) -> Blocks:
    """Partition residues into k contiguous blocks via recursive spectral bipartition with size checks."""
    L = pae.shape[0]
    if k <= 1:
        return Blocks([(0, L)])
    # initial split
    b2 = spectral_bipartition_from_pae(pae, sigma=sigma).blocks
    blocks = list(b2)
    while len(blocks) < k:
        # pick largest block and split
        idx = int(np.argmax([e - s for s, e in blocks]))
        s, e = blocks.pop(idx)
        sub = pae[s:e, s:e]
        sub_split = spectral_bipartition_from_pae(sub, sigma=sigma).blocks
        # adjust indices to global coords
        blocks.extend([(s + ss, s + ee) for (ss, ee) in sub_split])
        # safety: stop if no progress
        if len(sub_split) < 2:
            break
    # sort by start
    blocks = sorted(blocks, key=lambda x: x[0])
    # If overshoot, merge smallest to nearest
    while len(blocks) > k:
        sizes = [e - s for s, e in blocks]
        j = int(np.argmin(sizes))
        if j == 0:
            blocks[1] = (blocks[0][0], blocks[1][1])
            blocks.pop(0)
        else:
            blocks[j - 1] = (blocks[j - 1][0], blocks[j][1])
            blocks.pop(j)
    # Enforce minimum block length by merging tiny blocks into nearest neighbor
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        sizes = [e - s for s, e in blocks]
        j = int(np.argmin(sizes))
        if sizes[j] < max(1, min_block_len):
            # merge with neighbor yielding less size imbalance
            if j == 0:
                blocks[1] = (blocks[0][0], blocks[1][1])
                blocks.pop(0)
            elif j == len(blocks) - 1:
                blocks[-2] = (blocks[-2][0], blocks[-1][1])
                blocks.pop()
            else:
                left_size = blocks[j][1] - blocks[j - 1][0]
                right_size = blocks[j + 1][1] - blocks[j][0]
                if left_size <= right_size:
                    blocks[j - 1] = (blocks[j - 1][0], blocks[j][1])
                    blocks.pop(j)
                else:
                    blocks[j + 1] = (blocks[j][0], blocks[j + 1][1])
                    blocks.pop(j)
            changed = True
    # Final sort to ensure ordering
    blocks = sorted(blocks, key=lambda x: x[0])
    return Blocks(blocks)
