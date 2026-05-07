from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .afdb import mask_near_diagonal, symmetrize_pae, trimmed_mean
from .segmentation import Blocks


@dataclass
class ContrastStats:
    mu_intra: float
    mu_inter: float
    n_intra: int
    n_inter: int


def compute_contrast(pae_sym: np.ndarray, blocks: Blocks, delta: int = 7, trim: float = 0.1) -> ContrastStats:
    L = pae_sym.shape[0]
    m = mask_near_diagonal(L, delta)
    # Build masks for intra and inter entries
    intra_vals = []
    inter_vals = []
    for (s1, e1) in blocks.blocks:
        # intra block values
        bmask = np.zeros((L, L), dtype=bool)
        bmask[s1:e1, s1:e1] = True
        sel = bmask & m
        v = pae_sym[sel]
        if v.size:
            intra_vals.append(v)
    # inter-block values: pairs across different blocks
    if len(blocks.blocks) >= 2:
        (s1, e1), (s2, e2) = blocks.blocks[:2]
        imask = np.zeros((L, L), dtype=bool)
        imask[s1:e1, s2:e2] = True
        imask[s2:e2, s1:e1] = True
        sel = imask & m
        v = pae_sym[sel]
        if v.size:
            inter_vals.append(v)
    intra_concat = np.concatenate(intra_vals) if intra_vals else np.array([], dtype=np.float32)
    inter_concat = np.concatenate(inter_vals) if inter_vals else np.array([], dtype=np.float32)
    mu_intra = trimmed_mean(intra_concat, trim=trim)
    mu_inter = trimmed_mean(inter_concat, trim=trim)
    return ContrastStats(mu_intra=mu_intra, mu_inter=mu_inter, n_intra=int(intra_concat.size), n_inter=int(inter_concat.size))


def rsi_score(pae: np.ndarray, blocks: Blocks, delta: int = 7, trim: float = 0.1, sym_mode: str = "mean") -> float:
    pae_sym = symmetrize_pae(pae, mode=sym_mode)
    cs = compute_contrast(pae_sym, blocks, delta=delta, trim=trim)
    if not np.isfinite(cs.mu_intra) or not np.isfinite(cs.mu_inter):
        return float("nan")
    return float(cs.mu_inter - cs.mu_intra)


def _intra_inter_values(pae_sym: np.ndarray, blocks: Blocks, delta: int) -> Tuple[np.ndarray, np.ndarray]:
    L = pae_sym.shape[0]
    m = mask_near_diagonal(L, delta)
    intra_vals = []
    for (s1, e1) in blocks.blocks:
        bmask = np.zeros((L, L), dtype=bool)
        bmask[s1:e1, s1:e1] = True
        sel = bmask & m
        v = pae_sym[sel]
        if v.size:
            intra_vals.append(v)
    inter_vals = []
    if len(blocks.blocks) >= 2:
        (s1, e1), (s2, e2) = blocks.blocks[:2]
        imask = np.zeros((L, L), dtype=bool)
        imask[s1:e1, s2:e2] = True
        imask[s2:e2, s1:e1] = True
        sel = imask & m
        v = pae_sym[sel]
        if v.size:
            inter_vals.append(v)
    intra_concat = np.concatenate(intra_vals) if intra_vals else np.array([], dtype=np.float32)
    inter_concat = np.concatenate(inter_vals) if inter_vals else np.array([], dtype=np.float32)
    return intra_concat, inter_concat


def bcr_score(
    pae: np.ndarray,
    blocks: Blocks,
    delta: int = 7,
    trim: float = 0.1,
    n_perm: int = 50,
    rng: np.random.Generator | None = None,
    null_mode: str = "perm",
    sym_mode: str = "mean",
    return_p: bool = False,
) -> Tuple[float, float] | Tuple[float, float, float]:
    """Difference-based BCR (inter - intra) and Z vs a null; optional permutation p.

    Stability: clips PAE and guards against vanishing null variance. Z is not clamped.
    """
    pae_sym = symmetrize_pae(pae, mode=sym_mode)
    cs = compute_contrast(pae_sym, blocks, delta=delta, trim=trim)
    bcr = float(cs.mu_inter - cs.mu_intra)
    if rng is None:
        rng = np.random.default_rng(0)
    L = pae.shape[0]
    sizes = [e - s for (s, e) in blocks.blocks]
    if sum(sizes) != L and len(sizes) == 2:
        sizes = [sizes[0], L - sizes[0]]
    null_vals: list[float] = []
    for _ in range(n_perm):
        if null_mode == "rotation":
            shift = int(rng.integers(0, L))
            order = np.roll(np.arange(L), shift)
        else:
            order = rng.permutation(L)
        s0 = 0
        perm_blocks = []
        for sz in sizes:
            perm_blocks.append((s0, s0 + sz))
            s0 += sz
        perm = pae_sym[np.ix_(order, order)]
        csp = compute_contrast(perm, Blocks(perm_blocks), delta=delta, trim=trim)
        null_vals.append(float(csp.mu_inter - csp.mu_intra))
    null_arr = np.array(null_vals, dtype=np.float32)
    if null_arr.size == 0:
        return (bcr, float("nan"), float("nan")) if return_p else (bcr, float("nan"))
    mu = float(np.mean(null_arr))
    sd = float(np.std(null_arr, ddof=1))
    # Guard against near-zero variance; fall back to robust IQR-based scale
    if sd < 1e-6:
        q75, q25 = np.quantile(null_arr, 0.75), np.quantile(null_arr, 0.25)
        iqr = float(q75 - q25)
        sd = (iqr / 1.349) if iqr > 0 else 1e-3
    z = float((bcr - mu) / (sd + 1e-6))
    # Permutation p-value (one-sided): Pr[null >= observed]
    r = int(np.sum(null_arr >= bcr))
    p_perm = (r + 1.0) / (null_arr.size + 1.0)
    return (bcr, z, float(p_perm)) if return_p else (bcr, z)


def bcr_q_ratio(
    pae: np.ndarray,
    blocks: Blocks,
    delta: int = 7,
    q_inter: float = 0.75,
    q_intra: float = 0.25,
    n_perm: int = 50,
    rng: np.random.Generator | None = None,
    null_mode: str = "perm",
    sym_mode: str = "mean",
    return_p: bool = False,
) -> Tuple[float, float] | Tuple[float, float, float]:
    """Quantile-based BCR log-ratio with robust Z; optional permutation p.

    stat = log((Q_inter + eps) / (Q_intra + eps)) with eps = max(Q05_intra, 1e-3)
    Z is computed against a null (perm/rotation) using median/MAD.
    Per-protein PAE is normalized by Q95(off-diagonal) to stabilize scale.
    Returns (stat, robust_z).
    """
    pae_sym = symmetrize_pae(pae, mode=sym_mode)
    # Per-protein normalization by Q95 of off-diagonal pairs to stabilize scale
    L = pae_sym.shape[0]
    m_all = mask_near_diagonal(L, delta)
    scale = float(np.quantile(pae_sym[m_all], 0.95)) if np.any(m_all) else 1.0
    # Avoid amplifying noise for uniformly low-PAE proteins
    scale = max(scale, 5.0)
    pae_sym = pae_sym / scale
    intra, inter = _intra_inter_values(pae_sym, blocks, delta)
    if inter.size == 0 or intra.size == 0:
        return (float("nan"), float("nan"), float("nan")) if return_p else (float("nan"), float("nan"))
    q_inter_v = float(np.quantile(inter, q_inter))
    q_intra_v = float(np.quantile(intra, q_intra))
    q05_intra = float(np.quantile(intra, 0.05))
    eps = max(q05_intra, 1e-3)
    stat = float(np.log((q_inter_v + eps) / (q_intra_v + eps)))
    if rng is None:
        rng = np.random.default_rng(0)
    L = pae.shape[0]
    sizes = [e - s for (s, e) in blocks.blocks]
    if sum(sizes) != L and len(sizes) == 2:
        sizes = [sizes[0], L - sizes[0]]
    null_vals: list[float] = []
    for _ in range(n_perm):
        if null_mode == "rotation":
            shift = int(rng.integers(0, L))
            order = np.roll(np.arange(L), shift)
        else:
            order = rng.permutation(L)
        s0 = 0
        perm_blocks = []
        for sz in sizes:
            perm_blocks.append((s0, s0 + sz))
            s0 += sz
        perm = pae_sym[np.ix_(order, order)]
        intra_p, inter_p = _intra_inter_values(perm, Blocks(perm_blocks), delta)
        if inter_p.size == 0 or intra_p.size == 0:
            continue
        qi = float(np.quantile(inter_p, q_inter))
        qj = float(np.quantile(intra_p, q_intra))
        null_vals.append(float(np.log((qi + eps) / (qj + eps))))
    null_arr = np.array(null_vals, dtype=np.float32)
    if null_arr.size == 0:
        return (stat, float("nan"), float("nan")) if return_p else (stat, float("nan"))
    med = float(np.median(null_arr))
    mad = float(np.median(np.abs(null_arr - med)))
    robust_sd = 1.4826 * mad
    if not np.isfinite(robust_sd) or robust_sd < 1e-6:
        # fallback to IQR/1.349, else a conservative floor
        q75, q25 = np.quantile(null_arr, 0.75), np.quantile(null_arr, 0.25)
        iqr = float(q75 - q25)
        robust_sd = (iqr / 1.349) if iqr > 0 else 1e-3
    z = float((stat - med) / (robust_sd + 1e-6))
    # Permutation p-value (one-sided): Pr[null >= observed]
    r = int(np.sum(null_arr >= stat))
    p_perm = (r + 1.0) / (null_arr.size + 1.0)
    return (stat, z, float(p_perm)) if return_p else (stat, z)


def rsi_n_score(pae: np.ndarray, blocks: Blocks, delta: int = 7, trim: float = 0.1, sym_mode: str = "mean") -> float:
    """Size/robustness-normalized RSI: (mu_inter - mu_intra) / robust_scale.
    robust_scale approximated via IQR/1.349 of pooled (inter ∪ intra).
    """
    pae_sym = symmetrize_pae(pae, mode=sym_mode)
    intra, inter = _intra_inter_values(pae_sym, blocks, delta)
    if inter.size == 0 or intra.size == 0:
        return float("nan")
    mu_intra = trimmed_mean(intra, trim)
    mu_inter = trimmed_mean(inter, trim)
    pooled = np.concatenate([intra, inter])
    q75 = np.quantile(pooled, 0.75)
    q25 = np.quantile(pooled, 0.25)
    iqr = float(q75 - q25)
    robust_sd = iqr / 1.349 if iqr > 0 else float(np.std(pooled) + 1e-6)
    return float((mu_inter - mu_intra) / (robust_sd + 1e-6))


def quartiles_used_normalized(
    pae: np.ndarray,
    blocks: Blocks,
    delta: int = 7,
    q_inter: float = 0.75,
    q_intra: float = 0.25,
) -> tuple[float, float]:
    """Return (Q25_intra, Q75_inter) after per-protein Q95 normalization and symmetrization.

    Mirrors the normalization used by bcr_q_ratio to ensure consistency with the BCR quantile statistic.
    """
    pae_sym = symmetrize_pae(pae, mode="mean")
    L = pae_sym.shape[0]
    m_all = mask_near_diagonal(L, delta)
    scale = float(np.quantile(pae_sym[m_all], 0.95)) if np.any(m_all) else 1.0
    scale = max(scale, 5.0)
    pae_sym = pae_sym / scale
    intra, inter = _intra_inter_values(pae_sym, blocks, delta)
    if inter.size == 0 or intra.size == 0:
        return float("nan"), float("nan")
    q_intra_v = float(np.quantile(intra, q_intra))
    q_inter_v = float(np.quantile(inter, q_inter))
    return q_intra_v, q_inter_v
