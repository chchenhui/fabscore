"""Landmark MDS (LMDS) for out-of-sample embedding from distance matrices.

Implements classical MDS on the landmark submatrix, then extends to all
non-landmark nodes via the LMDS triangulation formula:
  x_new = (1/2) * L_embed_pinv @ (mean_landmark_sq_dists - d_new_sq)

Uses scipy.linalg.eigh for eigendecomposition and numpy.linalg.pinv for
pseudoinverse.

Reference: de Silva & Tenenbaum, "Landmark Isomap" (NeurIPS 2003).
"""

import numpy as np
from scipy.linalg import eigh


def landmark_mds(
    distance_matrix_landmarks_to_all: np.ndarray,
    n_components: int,
    landmark_indices: np.ndarray | None = None,
) -> np.ndarray:
    L, n = distance_matrix_landmarks_to_all.shape

    if landmark_indices is None:
        landmark_indices = np.arange(L)

    D_ll = distance_matrix_landmarks_to_all[:, landmark_indices]
    D_ll_sq = D_ll ** 2

    H = np.eye(L) - np.ones((L, L)) / L
    B = -0.5 * H @ D_ll_sq @ H

    eigenvalues, eigenvectors = eigh(B)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_k = min(n_components, np.sum(eigenvalues > 1e-10))
    if top_k < n_components:
        pass

    lam = eigenvalues[:n_components]
    lam = np.maximum(lam, 0)
    V = eigenvectors[:, :n_components]

    L_embed = V * np.sqrt(lam)[np.newaxis, :]

    mean_landmark_sq_dists = np.mean(D_ll_sq, axis=1)

    L_embed_pinv = np.linalg.pinv(L_embed)

    D_all_sq = distance_matrix_landmarks_to_all ** 2

    non_landmark_mask = np.ones(n, dtype=bool)
    non_landmark_mask[landmark_indices] = False
    non_landmark_indices = np.where(non_landmark_mask)[0]

    embedding = np.zeros((n, n_components))
    embedding[landmark_indices] = L_embed

    if len(non_landmark_indices) > 0:
        D_nonlandmark_sq = D_all_sq[:, non_landmark_indices]
        delta = mean_landmark_sq_dists[:, np.newaxis] - D_nonlandmark_sq
        embedding[non_landmark_indices] = 0.5 * (L_embed_pinv @ delta).T

    return embedding
