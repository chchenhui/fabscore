# Bradley-Terry baseline: scalar skill parameters, ties treated as half-wins.
# MLE via L-BFGS with identifiability constraint sum(beta) = 0.
# predict_probs returns 4-way distribution with epsilon smoothing for TIE/BOTH_BAD.

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid


class BradleyTerryModel:
    def __init__(self):
        self.system_to_idx = {}
        self.idx_to_system = {}
        self.beta = None
        self.n_systems = 0

    def fit(self, battles_df):
        systems = sorted(set(battles_df['system_a']) | set(battles_df['system_b']))
        self.system_to_idx = {s: i for i, s in enumerate(systems)}
        self.idx_to_system = {i: s for s, i in self.system_to_idx.items()}
        self.n_systems = len(systems)

        idx_a = battles_df['system_a'].map(self.system_to_idx).values
        idx_b = battles_df['system_b'].map(self.system_to_idx).values
        prefs = battles_df['preference'].values

        y_a = np.zeros(len(prefs), dtype=np.float64)
        y_b = np.zeros(len(prefs), dtype=np.float64)
        for i, p in enumerate(prefs):
            if p == 'A':
                y_a[i] = 1.0
            elif p == 'B':
                y_b[i] = 1.0
            else:  # TIE or BOTH_BAD
                y_a[i] = 0.5
                y_b[i] = 0.5

        def neg_log_likelihood(beta):
            beta = beta - beta.mean()
            diff = beta[idx_a] - beta[idx_b]
            log_sig_pos = np.log(expit(diff) + 1e-15)
            log_sig_neg = np.log(expit(-diff) + 1e-15)
            nll = -np.sum(y_a * log_sig_pos + y_b * log_sig_neg)
            return nll

        def grad_nll(beta):
            beta = beta - beta.mean()
            diff = beta[idx_a] - beta[idx_b]
            sig = expit(diff)
            residual = y_a - sig
            grad = np.zeros(self.n_systems)
            np.add.at(grad, idx_a, -residual)
            np.add.at(grad, idx_b, residual)
            grad -= grad.mean()
            return grad

        beta0 = np.zeros(self.n_systems)
        result = minimize(
            neg_log_likelihood,
            beta0,
            jac=grad_nll,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12}
        )
        self.beta = result.x - result.x.mean()
        return self

    def predict_probs(self, system_i, system_j, eps=1e-8):
        beta_i = self.beta[self.system_to_idx[system_i]]
        beta_j = self.beta[self.system_to_idx[system_j]]
        p_a = float(expit(beta_i - beta_j))
        p_b = 1.0 - p_a
        p_a_smooth = max(p_a - eps, eps)
        p_b_smooth = max(p_b - eps, eps)
        probs = np.array([p_a_smooth, p_b_smooth, eps, eps])
        probs = probs / probs.sum()
        return probs

    def predict_probs_batch(self, battles_df, eps=1e-8):
        idx_a = battles_df['system_a'].map(self.system_to_idx).values
        idx_b = battles_df['system_b'].map(self.system_to_idx).values
        diff = self.beta[idx_a] - self.beta[idx_b]
        p_a = expit(diff)
        p_b = 1.0 - p_a
        p_a_smooth = np.clip(p_a - eps, eps, None)
        p_b_smooth = np.clip(p_b - eps, eps, None)
        probs = np.column_stack([p_a_smooth, p_b_smooth,
                                 np.full(len(diff), eps),
                                 np.full(len(diff), eps)])
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def get_scores(self):
        return {self.idx_to_system[i]: float(self.beta[i])
                for i in range(self.n_systems)}
