# OSNIP-like key-conditioned encryptor MLP architecture.
# Input: token embedding h_t (R^d) concatenated with per-request Gaussian key k (R^d_k).
# Output: encrypted embedding z_t via iso-norm projection preserving ||h_t||.
# Architecture: 2 hidden layers [d+d_k -> 2d -> d -> d] with GELU activations.

import torch
import torch.nn as nn


class KeyConditionedEncryptor(nn.Module):
    def __init__(self, hidden_dim=3584, key_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        input_dim = hidden_dim + key_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, h, k):
        """
        Args:
            h: (batch, seq_len, hidden_dim) - token embeddings
            k: (batch, key_dim) - per-request Gaussian key
        Returns:
            z: (batch, seq_len, hidden_dim) - encrypted embeddings (iso-norm projected)
        """
        batch, seq_len, _ = h.shape
        k_expanded = k.unsqueeze(1).expand(batch, seq_len, self.key_dim)
        hk = torch.cat([h, k_expanded], dim=-1)

        delta = self.mlp(hk)

        z = self._iso_norm_project(h, delta)
        return z

    @staticmethod
    def _iso_norm_project(h, delta):
        """Iso-norm projection: z = (h + delta) * ||h|| / ||h + delta||"""
        h_plus_delta = h + delta
        h_norm = torch.norm(h, dim=-1, keepdim=True).clamp(min=1e-8)
        hd_norm = torch.norm(h_plus_delta, dim=-1, keepdim=True).clamp(min=1e-8)
        z = h_plus_delta * (h_norm / hd_norm)
        return z

    @staticmethod
    def sample_key(batch_size, key_dim=128, device="cuda", dtype=torch.float32):
        return torch.randn(batch_size, key_dim, device=device, dtype=dtype)
