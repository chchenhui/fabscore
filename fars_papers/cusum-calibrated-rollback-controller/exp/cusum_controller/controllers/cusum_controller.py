# FIR-CUSUM-epsilon sequential test controller (proposed method).
# Accumulates evidence of sustained upward drift via one-sided CUSUM statistic:
#   S_t = max(0, S_{t-1} + r_t - k), rollback when S_t > h.
# r_t = (nu_t - mu_0) / (sigma_0 + 1e-8) is the standardized innovation.
# Uses Fast Initial Response (FIR): on rollback S_t resets to h*reset_fraction
# instead of 0, so the CUSUM re-triggers quickly during sustained perturbation.
# On accept: update y_hat via EMA, save new snapshot.

from cusum_controller.controllers.base_controller import BaseController


class CUSUMController(BaseController):
    def __init__(self, alpha=0.1, k=0.5, h=None, mu_0=0.0, sigma_0=1.0,
                 reset_fraction=0.5):
        if h is None:
            raise ValueError("h must be provided (run calibration first)")
        self.alpha = alpha
        self.k = k
        self.h = h
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.reset_fraction = reset_fraction
        self.y_hat = None
        self.S_t = 0.0

    def initialize(self, y0):
        self.y_hat = y0
        self.S_t = 0.0

    def decide(self, y_proposed):
        nu = y_proposed - self.y_hat
        r_t = (nu - self.mu_0) / (self.sigma_0 + 1e-8)
        self.S_t = max(0.0, self.S_t + r_t - self.k)

        if self.S_t > self.h:
            accepted = False
            self.S_t = self.h * self.reset_fraction
        else:
            accepted = True
            self.y_hat = (1.0 - self.alpha) * self.y_hat + self.alpha * y_proposed

        info = {
            "y_hat": self.y_hat,
            "innovation": nu,
            "r_t": r_t,
            "S_t": self.S_t,
            "accepted": accepted,
        }

        return accepted, info
