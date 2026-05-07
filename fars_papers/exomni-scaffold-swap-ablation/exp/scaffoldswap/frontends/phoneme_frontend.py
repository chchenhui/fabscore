"""Condition C: Phoneme + timing frontend using MFA alignments.

Encodes phoneme IDs via learned embedding, concatenates with within-phoneme
position p in [0,1], duration d, and prosody (F0, energy). No pretrained
audio encoder -- all features are precomputed from MFA alignment + prosody extraction.
"""
import torch
import torch.nn as nn


class PhonemeFrontend(nn.Module):
    def __init__(self, n_phonemes=67, embed_dim=256, prosody_dim=2,
                 timing_dim=2, hidden_dim=256):
        super().__init__()
        self.phoneme_embed = nn.Embedding(n_phonemes, embed_dim)
        self.proj = nn.Linear(embed_dim + timing_dim + prosody_dim, hidden_dim)
        self.n_phonemes = n_phonemes
        self.embed_dim = embed_dim
        self.prosody_dim = prosody_dim
        self.timing_dim = timing_dim
        self.hidden_dim = hidden_dim

    def forward(self, phoneme_ids, phoneme_pos, phoneme_dur, prosody):
        """
        Args:
            phoneme_ids: (B, T_50hz) long tensor of phoneme indices
            phoneme_pos: (B, T_50hz) float, within-phoneme position in [0,1]
            phoneme_dur: (B, T_50hz) float, phoneme duration in seconds
            prosody: (B, T_50hz, 2) F0 + energy

        Returns:
            features: (B, T_50hz, hidden_dim)
        """
        emb = self.phoneme_embed(phoneme_ids)  # (B, T, embed_dim)

        T_emb = emb.shape[1]
        T_pros = prosody.shape[1]
        T = min(T_emb, T_pros)
        emb = emb[:, :T, :]
        prosody = prosody[:, :T, :]

        if self.timing_dim > 0:
            phoneme_pos = phoneme_pos[:, :T]
            phoneme_dur = phoneme_dur[:, :T]
            timing = torch.stack([phoneme_pos, phoneme_dur], dim=-1)  # (B, T, 2)
            combined = torch.cat([emb, timing, prosody], dim=-1)
        else:
            combined = torch.cat([emb, prosody], dim=-1)

        features = self.proj(combined)  # (B, T, hidden_dim)
        return features

    @property
    def output_rate(self):
        return 50  # Hz
