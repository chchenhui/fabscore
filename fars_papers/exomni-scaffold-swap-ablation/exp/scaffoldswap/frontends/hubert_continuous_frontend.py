"""HuBERT-continuous frontend: frozen HuBERT-base-ls960 + prosody, NO discretization.

Same architecture as SSLFrontend (Condition A) but uses HuBERT instead of WavLM.
This isolates the effect of discretization when compared against Condition B (discrete units).
"""
import torch
import torch.nn as nn
from transformers import HubertModel


class HuBERTContinuousFrontend(nn.Module):
    def __init__(self, ssl_model_name="facebook/hubert-base-ls960", ssl_dim=768,
                 prosody_dim=2, hidden_dim=256, cache_dir=None):
        super().__init__()
        self.ssl_model = HubertModel.from_pretrained(
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
        with torch.no_grad():
            ssl_out = self.ssl_model(
                audio, attention_mask=attention_mask
            ).last_hidden_state

        T_ssl = ssl_out.shape[1]
        T_pros = prosody.shape[1]
        T = min(T_ssl, T_pros)
        ssl_out = ssl_out[:, :T, :]
        prosody = prosody[:, :T, :]

        combined = torch.cat([ssl_out, prosody], dim=-1)
        features = self.proj(combined)
        return features

    @property
    def output_rate(self):
        return 50
