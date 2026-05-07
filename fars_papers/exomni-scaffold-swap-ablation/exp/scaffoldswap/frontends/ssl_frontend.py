"""Condition A: SSL frontend using WavLM-base-plus (frozen) + prosody features.

Produces (T_50hz, hidden_dim) features from raw audio waveform.
WavLM last hidden state (768-dim) is concatenated with F0+energy (2-dim)
then projected via a trainable linear layer.
"""
import torch
import torch.nn as nn
from transformers import WavLMModel


class SSLFrontend(nn.Module):
    def __init__(self, ssl_model_name="microsoft/wavlm-base-plus", ssl_dim=768,
                 prosody_dim=2, hidden_dim=256, cache_dir=None):
        super().__init__()
        self.ssl_model = WavLMModel.from_pretrained(
            ssl_model_name, cache_dir=cache_dir
        )
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        self.ssl_model.eval()

        self.proj = nn.Linear(ssl_dim + prosody_dim, hidden_dim)
        self.ssl_dim = ssl_dim
        self.prosody_dim = prosody_dim
        self.hidden_dim = hidden_dim

    def forward(self, audio, prosody, attention_mask=None):
        """
        Args:
            audio: (B, n_samples) raw waveform at 16kHz
            prosody: (B, T_50hz, 2) F0 + energy features
            attention_mask: (B, n_samples) optional mask for padding

        Returns:
            features: (B, T_50hz, hidden_dim)
        """
        with torch.no_grad():
            ssl_out = self.ssl_model(
                audio, attention_mask=attention_mask
            ).last_hidden_state  # (B, T_50hz, 768)

        T_ssl = ssl_out.shape[1]
        T_pros = prosody.shape[1]
        T = min(T_ssl, T_pros)
        ssl_out = ssl_out[:, :T, :]
        prosody = prosody[:, :T, :]

        combined = torch.cat([ssl_out, prosody], dim=-1)  # (B, T, 770)
        features = self.proj(combined)  # (B, T, hidden_dim)
        return features

    @property
    def output_rate(self):
        return 50  # Hz
