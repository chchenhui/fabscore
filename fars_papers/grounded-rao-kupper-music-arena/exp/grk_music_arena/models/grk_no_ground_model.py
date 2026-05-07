# GRK without grounding: replaces model-quality-dependent BOTH_BAD probability
# with a constant rate P(BOTH_BAD) = sigmoid(c). Tests whether the grounding
# mechanism (coupling BOTH_BAD to skill) drives GRK's improvement.
#
# P(BOTH_BAD) = sigmoid(c)  -- constant across all matchups
# P(i wins) = (1 - p_bad) * phi_i / (phi_i + lam*phi_j)
# P(j wins) = (1 - p_bad) * phi_j / (phi_j + lam*phi_i)
# P(TIE)    = 1 - P(i wins) - P(j wins) - P(BOTH_BAD)

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

OUTCOME_MAP = {'A': 0, 'B': 1, 'TIE': 2, 'BOTH_BAD': 3}


class GRKNoGroundModel:
    def __init__(self):
        self.system_to_idx = {}
        self.idx_to_system = {}
        self.beta = None
        self.lam = None
        self.c = None
        self.n_systems = 0

    def _compute_probs(self, beta, lam, c, idx_a, idx_b, eps=1e-12):
        phi_a = np.exp(beta[idx_a])
        phi_b = np.exp(beta[idx_b])

        p_bad = expit(c)

        denom_a = phi_a + lam * phi_b
        denom_b = phi_b + lam * phi_a

        rk_p_a = phi_a / denom_a
        rk_p_b = phi_b / denom_b

        p_a = (1.0 - p_bad) * rk_p_a
        p_b = (1.0 - p_bad) * rk_p_b
        p_tie = 1.0 - p_a - p_b - p_bad
        p_tie = np.clip(p_tie, eps, None)

        probs = np.column_stack([p_a, p_b, p_tie, np.full_like(p_a, p_bad)])
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def fit(self, battles_df, l2_beta=0.0):
        systems = sorted(set(battles_df['system_a']) | set(battles_df['system_b']))
        self.system_to_idx = {s: i for i, s in enumerate(systems)}
        self.idx_to_system = {i: s for s, i in self.system_to_idx.items()}
        self.n_systems = len(systems)
        K = self.n_systems

        idx_a = battles_df['system_a'].map(self.system_to_idx).values
        idx_b = battles_df['system_b'].map(self.system_to_idx).values
        y = np.array([OUTCOME_MAP[p] for p in battles_df['preference'].values])
        N = len(y)

        def neg_log_likelihood(params):
            beta = params[:K].copy()
            beta -= beta.mean()
            log_lam_m1 = params[K]
            lam = 1.0 + np.exp(log_lam_m1)
            c = params[K + 1]

            probs = self._compute_probs(beta, lam, c, idx_a, idx_b)
            log_probs = np.log(np.clip(probs[np.arange(N), y], 1e-15, None))
            nll = -log_probs.sum() / N
            if l2_beta > 0:
                nll += l2_beta * np.sum(beta ** 2)
            return nll

        x0 = np.zeros(K + 2)
        x0[K] = np.log(0.3)
        x0[K + 1] = 0.0

        result = minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
        )

        self.beta = result.x[:K].copy()
        self.beta -= self.beta.mean()
        self.lam = 1.0 + np.exp(result.x[K])
        self.c = result.x[K + 1]

        return self

    def predict_probs(self, system_i, system_j):
        idx_a = np.array([self.system_to_idx[system_i]])
        idx_b = np.array([self.system_to_idx[system_j]])
        probs = self._compute_probs(self.beta, self.lam, self.c, idx_a, idx_b)
        return probs[0]

    def predict_probs_batch(self, battles_df):
        idx_a = battles_df['system_a'].map(self.system_to_idx).values
        idx_b = battles_df['system_b'].map(self.system_to_idx).values
        return self._compute_probs(self.beta, self.lam, self.c, idx_a, idx_b)

    def get_scores(self):
        return {
            'beta': {self.idx_to_system[i]: float(self.beta[i]) for i in range(self.n_systems)},
            'lambda': float(self.lam),
            'c': float(self.c),
            'p_bothbad_constant': float(expit(self.c)),
        }
