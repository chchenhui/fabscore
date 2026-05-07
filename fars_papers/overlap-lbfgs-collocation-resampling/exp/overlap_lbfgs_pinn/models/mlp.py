# MLP with tanh activations for PINN.
# Small networks (10^4-10^5 parameters) operating in FP64.
# Architecture follows pinn_clusters: 6 hidden layers of 20 units, tanh, Xavier init.

import torch
import torch.nn as nn


class PINNMLP(nn.Module):

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = [1, 20, 20, 20, 20, 20, 20, 3]
        linears = []
        for i in range(len(layers) - 1):
            lin = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            linears.append(lin)
        self.linears = nn.ModuleList(linears)
        self.to(torch.float64)

    def forward(self, x):
        for i, lin in enumerate(self.linears[:-1]):
            x = torch.tanh(lin(x))
        x = self.linears[-1](x)
        return x
