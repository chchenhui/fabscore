"""UniTalker-Base-style non-autoregressive TCN motion decoder.

Architecture: N temporal conv blocks with residual connections and LayerNorm,
plus a learnable per-speaker identity embedding concatenated at each frame.
Final linear head maps to PCA dimension.
"""
import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        """x: (B, T, C)"""
        residual = x
        out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C)
        out = self.norm(out)
        out = self.act(out)
        return out + residual


class TCNMotionDecoder(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=512, n_blocks=5,
                 kernel_size=3, num_speakers=6, speaker_embed_dim=64):
        super().__init__()
        self.speaker_embed = nn.Embedding(num_speakers, speaker_embed_dim)
        self.input_proj = nn.Linear(hidden_dim + speaker_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size) for _ in range(n_blocks)
        ])
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, speaker_id):
        """
        Args:
            x: (B, T, hidden_dim) features from frequency adaptor
            speaker_id: (B,) integer speaker indices

        Returns:
            pca_coeffs: (B, T, output_dim) predicted PCA coefficients
        """
        B, T, _ = x.shape
        spk_emb = self.speaker_embed(speaker_id)  # (B, speaker_embed_dim)
        spk_emb = spk_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, speaker_embed_dim)

        x = torch.cat([x, spk_emb], dim=-1)  # (B, T, hidden_dim + speaker_embed_dim)
        x = self.input_proj(x)  # (B, T, hidden_dim)

        for block in self.blocks:
            x = block(x)

        return self.output_head(x)  # (B, T, output_dim)
