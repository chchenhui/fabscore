"""Full ScaffoldSwap model: frontend + frequency_adaptor + TCN decoder.

Unified interface for all conditions. Frontend is swappable (SSL / Unit / Phoneme).
Conditions A/B: audio -> frontend -> adaptor -> decoder -> PCA
Condition C:    phoneme features -> frontend -> adaptor -> decoder -> PCA
"""
import torch
import torch.nn as nn

from scaffoldswap.model.frequency_adaptor import FrequencyAdaptor
from scaffoldswap.model.decoder import TCNMotionDecoder


class ScaffoldModel(nn.Module):
    def __init__(self, frontend, source_rate=50, target_fps=25,
                 hidden_dim=256, output_dim=512, n_decoder_blocks=5,
                 decoder_kernel_size=3, num_speakers=6, speaker_embed_dim=64):
        super().__init__()
        self.frontend = frontend
        self.adaptor = FrequencyAdaptor(source_rate=source_rate, target_fps=target_fps)
        self.decoder = TCNMotionDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_blocks=n_decoder_blocks,
            kernel_size=decoder_kernel_size,
            num_speakers=num_speakers,
            speaker_embed_dim=speaker_embed_dim,
        )

    def forward(self, audio, prosody, speaker_id, n_target_frames=None,
                audio_mask=None, phoneme_ids=None, phoneme_pos=None,
                phoneme_dur=None, shuffle_temporal=False):
        """
        Args (Conditions A/B):
            audio: (B, n_samples) raw waveform at 16kHz
            prosody: (B, T_50hz, 2) F0 + energy
            audio_mask: optional attention mask for SSL model

        Args (Condition C -- phoneme_ids not None):
            phoneme_ids: (B, T_50hz) long
            phoneme_pos: (B, T_50hz) float
            phoneme_dur: (B, T_50hz) float
            prosody: (B, T_50hz, 2) F0 + energy

        Common:
            speaker_id: (B,) speaker indices
            n_target_frames: explicit target length
            shuffle_temporal: if True, randomly permute feature frames per sample

        Returns:
            pca_pred: (B, T_target, output_dim)
        """
        if phoneme_ids is not None:
            features = self.frontend(phoneme_ids, phoneme_pos, phoneme_dur, prosody)
        else:
            features = self.frontend(audio, prosody, attention_mask=audio_mask)

        if shuffle_temporal:
            for i in range(features.shape[0]):
                perm = torch.randperm(features.shape[1], device=features.device)
                features[i] = features[i][perm]

        features = self.adaptor(features, n_target_frames=n_target_frames)
        pca_pred = self.decoder(features, speaker_id)
        return pca_pred
