# No-controller pass-through baseline.
# Always accepts optimizer updates; no rollback mechanism.
# Still computes probe loss innovation and EMA for post-hoc calibration.

from cusum_controller.controllers.base_controller import BaseController


class NoController(BaseController):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.y_hat = None

    def initialize(self, y0):
        self.y_hat = y0

    def decide(self, y_proposed):
        nu = y_proposed - self.y_hat
        info = {
            "y_hat": self.y_hat,
            "innovation": nu,
            "accepted": True,
        }
        self.y_hat = (1.0 - self.alpha) * self.y_hat + self.alpha * y_proposed
        return True, info
