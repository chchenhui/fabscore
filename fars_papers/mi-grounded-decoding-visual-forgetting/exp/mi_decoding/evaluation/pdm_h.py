# Prompt Dependency Measure based on squared Hellinger distance (PDM-H).
# PDM-H(t) = H^2(p_c(.|x,c,y<t), p_u(.|x,y<t))
# Measures how much the next-token distribution depends on the image conditioning.
import numpy as np
import torch
import torch.nn.functional as F


def hellinger_squared(logits_c, logits_u):
    """Squared Hellinger distance between two distributions given their logits.

    H^2(p, q) = 0.5 * sum_k (sqrt(p_k) - sqrt(q_k))^2

    Args:
        logits_c: conditioned logits, shape (V,) - numpy or torch tensor
        logits_u: masked logits, shape (V,) - numpy or torch tensor

    Returns:
        float: H^2 in [0, 1]
    """
    if isinstance(logits_c, np.ndarray):
        logits_c = torch.from_numpy(logits_c).float()
    if isinstance(logits_u, np.ndarray):
        logits_u = torch.from_numpy(logits_u).float()

    p_c = F.softmax(logits_c, dim=-1)
    p_u = F.softmax(logits_u, dim=-1)

    sqrt_pc = torch.sqrt(p_c)
    sqrt_pu = torch.sqrt(p_u)

    h2 = 0.5 * torch.sum((sqrt_pc - sqrt_pu) ** 2).item()
    return h2


def compute_pdm_h_from_logits(logit_pairs):
    """Compute PDM-H at each saved generation step.

    Args:
        logit_pairs: dict mapping step (int) -> (logits_c, logits_u) numpy arrays

    Returns:
        dict mapping step (int) -> H^2 (float)
    """
    results = {}
    for step in sorted(logit_pairs.keys()):
        lc, lu = logit_pairs[step]
        results[step] = hellinger_squared(lc, lu)
    return results
