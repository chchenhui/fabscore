# Overlap-resampled L-BFGS optimizer for PINNs with collocation resampling.
# Implements multi-batch L-BFGS (Berahas, Nocedal & Takac, 2016) with:
#   - Two-loop recursion on overlap-set curvature pairs y_k = g_{k+1}^{O_k} - g_k^{O_k}
#   - Strong Wolfe line search on the current full batch S_k
#   - Cautious update rule (Berahas & Takac, 2017): skip (s,y) if y^T s too small
# Closures contract: callable() -> (loss_val, loss_e_val, loss_d_val).
#   Must zero_grad, forward, backward internally. Gradients read from p.grad.
# All computations in float64.

import torch
import math
from collections import deque


class OverlapLBFGS:

    def __init__(self, params, history_size=20, c1=1e-4, c2=0.9,
                 max_ls=20, cautious_eps=1e-6):
        self.params = list(params)
        self.history_size = history_size
        self.c1 = c1
        self.c2 = c2
        self.max_ls = max_ls
        self.cautious_eps = cautious_eps

        self.S = deque(maxlen=history_size)
        self.Y = deque(maxlen=history_size)
        self.rho = deque(maxlen=history_size)

        self._prev_overlap_grad = None
        self._step_count = 0
        self.diagnostics = {
            "line_search_evals": 0,
            "overlap_grad_evals": 0,
            "cautious_skips": 0,
            "line_search_failures": 0,
        }

    def _gather_flat_params(self):
        views = []
        for p in self.params:
            views.append(p.data.view(-1))
        return torch.cat(views)

    def _gather_flat_grad(self):
        views = []
        for p in self.params:
            if p.grad is not None:
                views.append(p.grad.data.view(-1))
            else:
                views.append(torch.zeros_like(p.data.view(-1)))
        return torch.cat(views)

    def _set_flat_params(self, flat):
        offset = 0
        for p in self.params:
            numel = p.data.numel()
            p.data.copy_(flat[offset:offset + numel].view_as(p.data))
            offset += numel

    def _two_loop(self, g):
        q = g.clone()
        m = len(self.S)
        if m == 0:
            return -q

        alphas = []
        for i in range(m - 1, -1, -1):
            a = self.rho[i] * self.S[i].dot(q)
            alphas.append(a)
            q.add_(self.Y[i], alpha=-a.item())
        alphas.reverse()

        sk = self.S[-1]
        yk = self.Y[-1]
        gamma = sk.dot(yk) / yk.dot(yk)
        r = gamma * q

        for i in range(m):
            b = self.rho[i] * self.Y[i].dot(r)
            r.add_(self.S[i], alpha=(alphas[i] - b).item())

        return -r

    def _strong_wolfe_line_search(self, full_closure, x0, f0, g0_flat, d,
                                  budget_tracker):
        dphi0 = g0_flat.dot(d).item()

        if dphi0 >= 0:
            return None, 0

        alpha_lo = 0.0
        alpha_hi = 2.0
        alpha = 1.0

        phi_lo = f0
        ls_evals = 0
        best_alpha = None
        best_f = f0

        for iteration in range(self.max_ls):
            if budget_tracker.exhausted():
                break

            self._set_flat_params(x0 + alpha * d)
            fi_tuple = full_closure()
            fi = fi_tuple[0]
            ls_evals += 1
            gi_flat = self._gather_flat_grad()
            dphi_i = gi_flat.dot(d).item()

            if not math.isfinite(fi):
                alpha_hi = alpha
                alpha = 0.5 * (alpha_lo + alpha_hi)
                continue

            if fi > f0 + self.c1 * alpha * dphi0 or (iteration > 0 and fi >= phi_lo):
                alpha_hi = alpha
                alpha = 0.5 * (alpha_lo + alpha_hi)
                continue

            if abs(dphi_i) <= -self.c2 * dphi0:
                return alpha, ls_evals

            if fi < best_f:
                best_f = fi
                best_alpha = alpha

            if dphi_i >= 0:
                alpha_hi = alpha
                alpha = 0.5 * (alpha_lo + alpha_hi)
            else:
                alpha_lo = alpha
                phi_lo = fi
                if alpha_hi <= alpha_lo:
                    alpha = min(2.0 * alpha, 10.0)
                else:
                    alpha = 0.5 * (alpha_lo + alpha_hi)

            if abs(alpha_hi - alpha_lo) < 1e-16:
                break

        if best_alpha is not None:
            self._set_flat_params(x0 + best_alpha * d)
            return best_alpha, ls_evals

        return None, ls_evals

    def step(self, full_closure, overlap_closure, budget_tracker):
        f_tuple = full_closure()
        f0 = f_tuple[0]
        budget_tracker.increment(1)
        self.diagnostics["line_search_evals"] += 1

        x0 = self._gather_flat_params().clone()
        g0 = self._gather_flat_grad().clone()
        g0_norm = g0.norm().item()

        step_info = {
            "loss": f0,
            "loss_e": f_tuple[1],
            "loss_d": f_tuple[2],
            "grad_norm": g0_norm,
            "ls_evals": 1,
            "overlap_evals": 0,
            "cautious_skip": False,
            "ls_failed": False,
            "alpha": 0.0,
            "nan_detected": False,
        }

        if not math.isfinite(f0) or not math.isfinite(g0_norm):
            step_info["nan_detected"] = True
            step_info["ls_failed"] = True
            return step_info

        if self._step_count == 0 and self._prev_overlap_grad is None:
            overlap_closure()
            budget_tracker.increment(1)
            self.diagnostics["overlap_grad_evals"] += 1
            step_info["overlap_evals"] += 1
            self._prev_overlap_grad = self._gather_flat_grad().clone()
            self._set_flat_params(x0)
            for p in self.params:
                if p.grad is not None:
                    p.grad.zero_()

        d = self._two_loop(g0)
        if d.dot(g0) >= 0:
            d = -g0

        alpha, ls_count = self._strong_wolfe_line_search(
            full_closure, x0, f0, g0, d, budget_tracker
        )
        budget_tracker.increment(ls_count)
        self.diagnostics["line_search_evals"] += ls_count
        step_info["ls_evals"] += ls_count

        if alpha is None:
            self.diagnostics["line_search_failures"] += 1
            step_info["ls_failed"] = True
            alpha = 1e-3
            self._set_flat_params(x0 + alpha * d)

        step_info["alpha"] = alpha
        x1 = self._gather_flat_params().clone()
        s_new = x1 - x0
        step_info["s_norm"] = s_new.norm().item()
        step_info["ys_value"] = None
        step_info["y_norm"] = None

        if not budget_tracker.exhausted():
            overlap_closure()
            budget_tracker.increment(1)
            self.diagnostics["overlap_grad_evals"] += 1
            step_info["overlap_evals"] += 1
            g_overlap_new = self._gather_flat_grad().clone()

            if not math.isfinite(g_overlap_new.norm().item()):
                self._prev_overlap_grad = None
            elif self._prev_overlap_grad is not None:
                y_new = g_overlap_new - self._prev_overlap_grad
                ys = y_new.dot(s_new).item()
                ss = s_new.dot(s_new).item()
                threshold = self.cautious_eps * ss * g0_norm

                step_info["ys_value"] = ys
                step_info["y_norm"] = y_new.norm().item()

                if ys > threshold and math.isfinite(ys):
                    self.S.append(s_new)
                    self.Y.append(y_new)
                    self.rho.append(1.0 / ys)
                else:
                    self.diagnostics["cautious_skips"] += 1
                    step_info["cautious_skip"] = True

                self._prev_overlap_grad = g_overlap_new
            else:
                self._prev_overlap_grad = g_overlap_new

        self._step_count += 1
        return step_info

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
