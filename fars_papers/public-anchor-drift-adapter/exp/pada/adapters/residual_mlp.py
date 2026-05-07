"""Residual MLP drift adapter.

Maps f_new embeddings toward f_old space via a lightweight residual network:
  g(x) = x + W2 * Dropout(GELU(W1 * x + b1)) + b2
Input and output are L2-normalized 768-dim embeddings.
Supports optional dropout (default 0.0 for backward compat, paper uses 0.1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPAdapter(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(embed_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.W2(self.dropout(F.gelu(self.W1(x))))
        out = F.normalize(out, p=2, dim=-1)
        return out
