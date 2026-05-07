# Utility-gated curriculum scheduler following OSNIP's optimization strategy.
# Dynamically modulates privacy and diversity loss weights based on:
#   1. Time-based linear warmup (0->1 over warmup_steps)
#   2. Utility-gated safety: relaxes constraints when utility loss is too high

import torch


class UtilityGatedScheduler:
    def __init__(
        self,
        lambda1_base=1.0,
        lambda2_base=0.5,
        warmup_steps=1000,
        tau_low=0.005,
        tau_high=0.05,
    ):
        self.lambda1_base = lambda1_base
        self.lambda2_base = lambda2_base
        self.warmup_steps = warmup_steps
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.step = 0

    def get_weights(self, util_loss_val):
        """Compute effective lambda1, lambda2 given current step and utility loss.

        Args:
            util_loss_val: float, current utility loss value
        Returns:
            (lambda1_eff, lambda2_eff, w_time, w_safe) for logging
        """
        w_time = min(1.0, self.step / max(self.warmup_steps, 1))

        if util_loss_val >= self.tau_high:
            w_safe = 0.0
        elif util_loss_val <= self.tau_low:
            w_safe = 1.0
        else:
            w_safe = (self.tau_high - util_loss_val) / (self.tau_high - self.tau_low)
        w_safe = max(0.0, min(1.0, w_safe))

        lambda1_eff = self.lambda1_base * w_time * w_safe
        lambda2_eff = self.lambda2_base * w_time * w_safe

        return lambda1_eff, lambda2_eff, w_time, w_safe

    def advance(self):
        self.step += 1
