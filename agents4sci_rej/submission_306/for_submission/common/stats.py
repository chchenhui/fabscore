from __future__ import annotations

from typing import Iterable, List, Optional

import math


def p_from_z(z: float, two_sided: bool = True) -> float:
    """Compute p-value from a Z-score using erfc to avoid underflow.

    Clamps extreme |z| to a safe range and returns a finite float in (0,1].
    """
    if z is None or not (z == z):  # NaN check
        return float('nan')
    # Guard extreme values to avoid inf/NaN
    z = float(max(min(z, 1e3), -1e3))
    # one-sided tail: sf(z) = 0.5 * erfc(z / sqrt(2)) for z >= 0
    az = abs(z)
    tail = 0.5 * math.erfc(az / math.sqrt(2.0))
    if two_sided:
        return max(min(2.0 * tail, 1.0), 0.0)
    return max(min(tail, 1.0), 0.0)


def benjamini_hochberg(pvals: Iterable[float]) -> List[Optional[float]]:
    """Benjamini–Hochberg FDR adjustment.

    Returns q-values in the original order. NaNs in pvals propagate to NaN.
    Implements the standard monotone step-up procedure.
    """
    p_list = list(pvals)
    n = len(p_list)
    # Pair with original indices, filter finite p-values
    pairs = [(i, p) for i, p in enumerate(p_list) if p is not None and p == p]
    if not pairs:
        return [float('nan')] * n
    # Sort by p ascending
    pairs.sort(key=lambda t: t[1])
    qvals = [float('nan')] * n
    m = len(pairs)
    # Compute raw q = p * m / rank, then enforce monotonicity from largest rank downwards
    prev = 1.0
    for rank, (i, p) in enumerate(reversed(pairs), start=1):
        # rank here counts from largest to smallest, so use m - (rank-1) as the BH rank
        bh_rank = m - (rank - 1)
        q = (p * m) / max(bh_rank, 1)
        q = min(q, prev, 1.0)
        qvals[i] = q
        prev = q
    return qvals

