"""
Validation-based measurement probe.

Implements an external measurement signal based on loss evaluated
over a fixed validation (probe) dataset.
"""

import torch
import torch.nn as nn

from runtime_stability_controller.probes.base import Probe


class ValidationProbe(Probe):
    """
    Validation loss probe.

    Evaluates model state using loss computed on a small, fixed
    validation batch that is not used for training.
    """

    def __init__(self, dataloader, loss_fn=None, device=None):
        """
        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader providing the validation probe data.
            Typically very small (e.g. 8–32 samples).
        loss_fn : callable, optional
            Loss function used for evaluation.
            Defaults to nn.MSELoss.
        device : torch.device or str, optional
            Device on which evaluation should be performed.
            If None, uses the model's current device.
        """
        self.dataloader = dataloader
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.device = device

    def evaluate(self, model):
        model.eval()

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, y in self.dataloader:
                if self.device is not None:
                    x = x.to(self.device)
                    y = y.to(self.device)

                pred = model(x)
                loss = self.loss_fn(pred, y)

                total_loss += float(loss.item())
                count += 1

        model.train()

        if count == 0:
            raise RuntimeError("ValidationProbe dataloader is empty.")

        return total_loss / count
