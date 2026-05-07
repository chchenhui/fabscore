"""Condition B: Discrete unit frontend using HuBERT + k-means quantization.

Extracts HuBERT-base-ls960 last_hidden_state (768-dim, 50 Hz), quantizes each frame
to the nearest k-means centroid (K=200), maps unit IDs through a learnable embedding,
concatenates with prosody (F0+energy), and projects to decoder hidden dim.
"""
import pickle

import numpy as np
import torch
import torch.nn as nn
from transformers import HubertModel


class UnitFrontend(nn.Module):
    def __init__(self, hubert_model_name="facebook/hubert-base-ls960",
                 kmeans_path="pretrained_models/hubert_kmeans_biwi_K200.pkl",
                 n_clusters=200, embed_dim=256, prosody_dim=2, hidden_dim=256,
                 cache_dir=None):
        super().__init__()
        self.hubert_model = HubertModel.from_pretrained(
            hubert_model_name, cache_dir=cache_dir
        )
        for param in self.hubert_model.parameters():
            param.requires_grad = False
        self.hubert_model.eval()

        with open(kmeans_path, "rb") as f:
            self.kmeans = pickle.load(f)
        self.centroids = torch.from_numpy(self.kmeans.cluster_centers_).float()

        self.unit_embedding = nn.Embedding(n_clusters, embed_dim)
        self.proj = nn.Linear(embed_dim + prosody_dim, hidden_dim)
        self.n_clusters = n_clusters
        self.embed_dim = embed_dim
        self.prosody_dim = prosody_dim
        self.hidden_dim = hidden_dim

    def _quantize(self, features):
        """Quantize features to nearest k-means centroid using sklearn predict.

        Args:
            features: (B, T, 768) HuBERT hidden states

        Returns:
            unit_ids: (B, T) long tensor of cluster IDs
        """
        B, T, D = features.shape
        flat = features.reshape(-1, D).cpu().numpy()
        ids = self.kmeans.predict(flat)
        unit_ids = torch.from_numpy(ids).long().reshape(B, T).to(features.device)
        return unit_ids

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
            hubert_out = self.hubert_model(
                audio, attention_mask=attention_mask
            ).last_hidden_state  # (B, T_50hz, 768)

        unit_ids = self._quantize(hubert_out)  # (B, T)
        unit_embeds = self.unit_embedding(unit_ids)  # (B, T, embed_dim)

        T_unit = unit_embeds.shape[1]
        T_pros = prosody.shape[1]
        T = min(T_unit, T_pros)
        unit_embeds = unit_embeds[:, :T, :]
        prosody = prosody[:, :T, :]

        combined = torch.cat([unit_embeds, prosody], dim=-1)  # (B, T, embed_dim+2)
        features = self.proj(combined)  # (B, T, hidden_dim)
        return features

    @property
    def output_rate(self):
        return 50  # Hz
