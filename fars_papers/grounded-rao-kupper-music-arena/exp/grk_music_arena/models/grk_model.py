# Grounded Rao-Kupper (GRK) model: BOTH_BAD as outside option via fictitious anchor.
# Couples BOTH_BAD probability to skill through grounded scores exp(beta_k).
#
# Base model (use_gamma=False):
#   P(i wins) = phi_i / (phi_i + lam*phi_j + 1)
#   P(j wins) = phi_j / (phi_j + lam*phi_i + 1)
#   P(BOTH_BAD) = 1 / (1 + phi_i + phi_j)
#   P(TIE) = 1 - P(i) - P(j) - P(BOTH_BAD)
#
# Extended model (use_gamma=True): per-system tie propensity gamma_k.
#   lam_ij = base_lam * exp(gamma_i + gamma_j)
#   Same formulas as above but with lam_ij replacing lam.
#   L2 regularization on gamma to prevent overfitting.
#
# Probability renormalization ensures valid distributions when P(TIE) is clipped.

import numpy as np
from scipy.optimize import minimize

OUTCOME_MAP = {'A': 0, 'B': 1, 'TIE': 2, 'BOTH_BAD': 3}


class GRKModel:
    def __init__(self, use_gamma=False):
        self.system_to_idx = {}
        self.idx_to_system = {}
        self.beta = None
        self.lam = None
        self.gamma = None
        self.n_systems = 0
        self.use_gamma = use_gamma

    def _compute_probs(self, beta, lam, idx_a, idx_b, gamma=None, eps=1e-12):
        phi_a = np.exp(beta[idx_a])
        phi_b = np.exp(beta[idx_b])

        if gamma is not None:
            lam_pair = lam * np.exp(gamma[idx_a] + gamma[idx_b])
        else:
            lam_pair = lam

        p_a = phi_a / (phi_a + lam_pair * phi_b + 1.0)
        p_b = phi_b / (phi_b + lam_pair * phi_a + 1.0)
        p_bad = 1.0 / (1.0 + phi_a + phi_b)
        p_tie = 1.0 - p_a - p_b - p_bad
        p_tie = np.clip(p_tie, eps, None)

        probs = np.column_stack([p_a, p_b, p_tie, p_bad])
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def fit(self, battles_df, l2_beta=0.0, l2_gamma=0.1):
        systems = sorted(set(battles_df['system_a']) | set(battles_df['system_b']))
        self.system_to_idx = {s: i for i, s in enumerate(systems)}
        self.idx_to_system = {i: s for s, i in self.system_to_idx.items()}
        self.n_systems = len(systems)
        K = self.n_systems

        idx_a = battles_df['system_a'].map(self.system_to_idx).values
        idx_b = battles_df['system_b'].map(self.system_to_idx).values
        y = np.array([OUTCOME_MAP[p] for p in battles_df['preference'].values])
        N = len(y)

        if self.use_gamma:
            def neg_log_likelihood(params):
                beta = params[:K].copy()
                beta -= beta.mean()
                gamma = params[K:2*K].copy()
                gamma -= gamma.mean()
                log_lam_m1 = params[2*K]
                base_lam = 1.0 + np.exp(log_lam_m1)

                probs = self._compute_probs(beta, base_lam, idx_a, idx_b, gamma=gamma)
                log_probs = np.log(np.clip(probs[np.arange(N), y], 1e-15, None))
                nll = -log_probs.sum() / N
                if l2_beta > 0:
                    nll += l2_beta * np.sum(beta ** 2)
                nll += l2_gamma * np.sum(gamma ** 2)
                return nll

            x0 = np.zeros(2*K + 1)
            x0[2*K] = np.log(0.3)
        else:
            def neg_log_likelihood(params):
                beta = params[:K].copy()
                beta -= beta.mean()
                log_lam_m1 = params[K]
                lam = 1.0 + np.exp(log_lam_m1)

                probs = self._compute_probs(beta, lam, idx_a, idx_b)
                log_probs = np.log(np.clip(probs[np.arange(N), y], 1e-15, None))
                nll = -log_probs.sum() / N
                if l2_beta > 0:
                    nll += l2_beta * np.sum(beta ** 2)
                return nll

            x0 = np.zeros(K + 1)
            x0[K] = np.log(0.3)

        result = minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
        )

        if self.use_gamma:
            self.beta = result.x[:K].copy()
            self.beta -= self.beta.mean()
            self.gamma = result.x[K:2*K].copy()
            self.gamma -= self.gamma.mean()
            self.lam = 1.0 + np.exp(result.x[2*K])
        else:
            self.beta = result.x[:K].copy()
            self.beta -= self.beta.mean()
            self.lam = 1.0 + np.exp(result.x[K])
            self.gamma = None

        return self

    def predict_probs(self, system_i, system_j):
        idx_a = np.array([self.system_to_idx[system_i]])
        idx_b = np.array([self.system_to_idx[system_j]])
        probs = self._compute_probs(self.beta, self.lam, idx_a, idx_b, gamma=self.gamma)
        return probs[0]

    def predict_probs_batch(self, battles_df):
        idx_a = battles_df['system_a'].map(self.system_to_idx).values
        idx_b = battles_df['system_b'].map(self.system_to_idx).values
        return self._compute_probs(self.beta, self.lam, idx_a, idx_b, gamma=self.gamma)

    def get_scores(self):
        scores = {
            'beta': {self.idx_to_system[i]: float(self.beta[i]) for i in range(self.n_systems)},
            'lambda': float(self.lam),
        }
        if self.gamma is not None:
            scores['gamma'] = {self.idx_to_system[i]: float(self.gamma[i]) for i in range(self.n_systems)}
        return scores

    def get_acceptability(self):
        phi = np.exp(self.beta)
        phi_avg = np.exp(self.beta.mean())
        p_bad = 1.0 / (1.0 + phi + phi_avg)
        return {self.idx_to_system[i]: float(p_bad[i]) for i in range(self.n_systems)}
