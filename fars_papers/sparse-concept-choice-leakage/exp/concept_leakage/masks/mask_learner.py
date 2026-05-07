"""SPARSE hard-concrete mask learning for concept-aware embedding privacy.
Jointly trains a binary hard-concrete mask m in [0,1]^d with a 2-layer MLP
classifier to separate D_plus vs D_minus embeddings. L0 sparsity regularization
encourages the mask to focus on concept-relevant dimensions."""

import torch
import torch.nn as nn
import numpy as np


class HardConcreteMask(nn.Module):
    """Hard-concrete stochastic gate (Louizos et al., 2018).
    xi=1.1, gamma=-0.1 stretch the sigmoid to allow exact 0/1 values."""

    def __init__(self, dim: int = 768, xi: float = 1.1, gamma: float = -0.1):
        super().__init__()
        self.dim = dim
        self.xi = xi
        self.gamma = gamma
        self.log_alpha = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.full((dim,), 2.0 / 3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-8, 1.0 - 1e-8)
            s = torch.sigmoid((torch.log(u / (1.0 - u)) + self.log_alpha) / self.beta)
            s_bar = s * (self.xi - self.gamma) + self.gamma
            mask = s_bar.clamp(0.0, 1.0)
        else:
            mask = torch.sigmoid(self.log_alpha) * (self.xi - self.gamma) + self.gamma
            mask = mask.clamp(0.0, 1.0)
        return x * mask

    def get_mask(self) -> torch.Tensor:
        with torch.no_grad():
            mask = torch.sigmoid(self.log_alpha) * (self.xi - self.gamma) + self.gamma
            return mask.clamp(0.0, 1.0)

    def l0_regularization(self) -> torch.Tensor:
        return torch.sigmoid(
            self.log_alpha - self.beta * torch.log(torch.tensor(-self.gamma / self.xi))
        ).mean()


class ConceptMLP(nn.Module):
    """2-hidden-layer MLP: 768 -> 256 -> 128 -> 1."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class MaskLearner(nn.Module):
    """Composes hard-concrete mask with MLP classifier."""

    def __init__(self, dim: int = 768, lambda_l0: float = 0.001):
        super().__init__()
        self.mask = HardConcreteMask(dim=dim)
        self.mlp = ConceptMLP(input_dim=dim)
        self.lambda_l0 = lambda_l0
        self.bce = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked = self.mask(x)
        return self.mlp(masked)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        pred = self.forward(x)
        cls_loss = self.bce(pred, y)
        l0_loss = self.mask.l0_regularization()
        total = cls_loss + self.lambda_l0 * l0_loss
        return {"total": total, "cls": cls_loss, "l0": l0_loss, "pred": pred}


def extract_mask(model: MaskLearner, dim: int = 768, delta: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Extract deterministic mask, normalize to trace=d, build diagonal Sigma."""
    mask = model.mask.get_mask().cpu().numpy().astype(np.float64)
    mask_sum = mask.sum()
    if mask_sum > 0:
        mask = mask * (dim / mask_sum)
    else:
        mask = np.ones(dim, dtype=np.float64)
    sigma = mask + delta
    return mask.astype(np.float32), sigma.astype(np.float32)
