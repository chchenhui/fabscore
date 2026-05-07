# AB-MNL (Absolute-Badness Multinomial Logit) baseline: decoupled skill and badness.
# 4-way softmax over (win_A, win_B, tie, both_bad) with per-system rho parameters.
# Logits: u_A=beta_i, u_B=beta_j, u_tie=tau+0.5*(beta_i+beta_j), u_bad=kappa+0.5*(rho_i+rho_j).
# Regularized MLE: NLL + l2_rho * sum(rho_k^2). Identifiability: center beta and rho.

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

OUTCOME_MAP = {'A': 0, 'B': 1, 'TIE': 2, 'BOTH_BAD': 3}


class ABMNLModel:
    def __init__(self):
        self.system_to_idx = {}
        self.idx_to_system = {}
        self.beta = None
        self.rho = None
        self.tau = None
        self.kappa = None
        self.n_systems = 0

    def _pack(self, beta, rho, tau, kappa):
        return np.concatenate([beta, rho, [tau, kappa]])

    def _unpack(self, params):
        K = self.n_systems
        beta = params[:K]
        rho = params[K:2*K]
        tau = params[2*K]
        kappa = params[2*K + 1]
        beta = beta - beta.mean()
        rho = rho - rho.mean()
        return beta, rho, tau, kappa

    def _compute_logits(self, beta, rho, tau, kappa, idx_a, idx_b):
        u_A = beta[idx_a]
        u_B = beta[idx_b]
        u_tie = tau + 0.5 * (beta[idx_a] + beta[idx_b])
        u_bad = kappa + 0.5 * (rho[idx_a] + rho[idx_b])
        return np.column_stack([u_A, u_B, u_tie, u_bad])

    def fit(self, battles_df, l2_rho=0.0):
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
            beta, rho, tau, kappa = self._unpack(params)
            logits = self._compute_logits(beta, rho, tau, kappa, idx_a, idx_b)
            log_probs = log_softmax(logits, axis=1)
            nll = -log_probs[np.arange(N), y].sum() / N
            reg = l2_rho * np.sum(rho ** 2)
            return nll + reg

        def grad_nll(params):
            beta, rho, tau, kappa = self._unpack(params)
            logits = self._compute_logits(beta, rho, tau, kappa, idx_a, idx_b)
            probs = softmax(logits, axis=1)

            residual = -np.zeros_like(probs)
            residual[np.arange(N), y] = 1.0
            diff = residual - probs

            grad_beta = np.zeros(K)
            grad_rho = np.zeros(K)

            np.add.at(grad_beta, idx_a, diff[:, 0])
            np.add.at(grad_beta, idx_b, diff[:, 1])
            np.add.at(grad_beta, idx_a, 0.5 * diff[:, 2])
            np.add.at(grad_beta, idx_b, 0.5 * diff[:, 2])

            np.add.at(grad_rho, idx_a, 0.5 * diff[:, 3])
            np.add.at(grad_rho, idx_b, 0.5 * diff[:, 3])

            grad_tau = diff[:, 2].sum()
            grad_kappa = diff[:, 3].sum()

            grad_beta = -grad_beta / N
            grad_rho = -grad_rho / N + 2 * l2_rho * rho
            grad_tau = -grad_tau / N
            grad_kappa = -grad_kappa / N

            grad_beta -= grad_beta.mean()
            grad_rho -= grad_rho.mean()

            return np.concatenate([grad_beta, grad_rho, [grad_tau, grad_kappa]])

        x0 = np.zeros(2 * K + 2)
        result = minimize(
            neg_log_likelihood,
            x0,
            jac=grad_nll,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-12}
        )
        self.beta, self.rho, self.tau, self.kappa = self._unpack(result.x)
        return self

    def predict_probs(self, system_i, system_j):
        idx_a = np.array([self.system_to_idx[system_i]])
        idx_b = np.array([self.system_to_idx[system_j]])
        logits = self._compute_logits(self.beta, self.rho, self.tau, self.kappa, idx_a, idx_b)
        return softmax(logits, axis=1)[0]

    def predict_probs_batch(self, battles_df):
        idx_a = battles_df['system_a'].map(self.system_to_idx).values
        idx_b = battles_df['system_b'].map(self.system_to_idx).values
        logits = self._compute_logits(self.beta, self.rho, self.tau, self.kappa, idx_a, idx_b)
        return softmax(logits, axis=1)

    def get_scores(self):
        return {
            'beta': {self.idx_to_system[i]: float(self.beta[i]) for i in range(self.n_systems)},
            'rho': {self.idx_to_system[i]: float(self.rho[i]) for i in range(self.n_systems)},
            'tau': float(self.tau),
            'kappa': float(self.kappa),
        }
