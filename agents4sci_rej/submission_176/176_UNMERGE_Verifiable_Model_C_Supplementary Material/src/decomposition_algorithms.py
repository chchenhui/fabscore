import torch
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.decomposition import SparseCoder
from scipy.optimize import nnls


class DecompositionAlgorithms:
    def __init__(self, dictionary_matrix):
        self.dictionary = dictionary_matrix.float()
        self.n_atoms, self.n_features = self.dictionary.shape

    def dot_product_similarity(self, target_vector, threshold=0.01):
        target_vector = target_vector.float()
        similarities = torch.matmul(self.dictionary, target_vector)
        coefficients = similarities / torch.norm(self.dictionary, dim=1)
        coefficients = torch.threshold(coefficients, threshold=threshold, value=0.0)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def lasso_regression(self, target_vector, alpha=0.01, max_iter=1000):
        target_vector = target_vector.float()
        lasso = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
        lasso.fit(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(lasso.coef_, dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def ridge_regression(self, target_vector, alpha=1.0):
        target_vector = target_vector.float()
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(ridge.coef_, dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def elastic_net_regression(
        self, target_vector, alpha=0.01, l1_ratio=0.5, max_iter=1000
    ):
        target_vector = target_vector.float()
        elastic_net = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=False
        )
        elastic_net.fit(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(elastic_net.coef_, dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def orthogonal_matching_pursuit(
        self, target_vector, n_nonzero_coefs=None, tol=1e-4
    ):
        target_vector = target_vector.float()
        if n_nonzero_coefs is None:
            n_nonzero_coefs = min(self.n_atoms, int(0.1 * self.n_atoms))

        coder = SparseCoder(
            dictionary=self.dictionary.T.numpy().transpose(),
            transform_algorithm="omp",
            transform_n_nonzero_coefs=n_nonzero_coefs,
            transform_alpha=tol,
        )
        coefficients = coder.transform(target_vector.numpy().reshape(1, -1))
        coefficients = torch.tensor(coefficients.flatten(), dtype=torch.float32)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction

    def non_negative_least_squares(self, target_vector, threshold=0.01):
        target_vector = target_vector.float()
        coefficients_np, _ = nnls(self.dictionary.T.numpy(), target_vector.numpy())
        coefficients = torch.tensor(coefficients_np, dtype=torch.float32)
        coefficients = torch.threshold(coefficients, threshold=threshold, value=0.0)
        reconstruction = torch.matmul(coefficients, self.dictionary)
        return coefficients, reconstruction
