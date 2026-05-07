# PCA-k top component removal (rank-matched control).
# Removes the top k principal components from embeddings, where k matches
# the number of INLP iterations, to test whether gains come from generic
# dimensionality reduction vs. targeted subject-identity removal.
# Uses efficient projection: Z_proj = Z - (Z @ V_k.T) @ V_k to avoid
# allocating a d x d matrix (infeasible for d=17600).

import numpy as np
from sklearn.decomposition import PCA


def pca_k_removal(Z_train, Z_test, k):
    pca = PCA(n_components=k)
    pca.fit(Z_train)
    V_k = pca.components_

    Z_train_proj = Z_train - (Z_train @ V_k.T) @ V_k
    Z_test_proj = Z_test - (Z_test @ V_k.T) @ V_k

    return Z_train_proj, Z_test_proj, V_k
