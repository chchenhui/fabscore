# MLP probe for activation-based harmful-request monitoring.
# Architecture: Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1) -> Sigmoid
# Following Activation Monitoring paper: hidden_dim=32.

import torch
import torch.nn as nn


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
