# Or-epsilon one-step innovation threshold controller (Or 2026 baseline).
# Triggers rollback when innovation nu_t = y(theta_proposed) - y_hat_t > epsilon.
# Snapshot save/restore is handled by the training loop, not this controller.

from cusum_controller.controllers.base_controller import BaseController


class OrController(BaseController):
    def __init__(self, alpha=0.1, epsilon=None):
        if epsilon is None:
            raise ValueError("epsilon must be provided (run calibration first)")
        self.alpha = alpha
        self.epsilon = epsilon
        self.y_hat = None

    def initialize(self, y0):
        self.y_hat = y0

    def decide(self, y_proposed):
        nu = y_proposed - self.y_hat
        accepted = nu <= self.epsilon

        info = {
            "y_hat": self.y_hat,
            "innovation": nu,
            "accepted": accepted,
        }

        if accepted:
            self.y_hat = (1.0 - self.alpha) * self.y_hat + self.alpha * y_proposed

        return accepted, info
