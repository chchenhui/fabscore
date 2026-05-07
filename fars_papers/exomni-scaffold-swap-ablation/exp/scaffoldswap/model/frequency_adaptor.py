"""Frequency adaptor: temporal linear interpolation from encoder rate to target fps.

Placed after the audio frontend, before the decoder. Uses torch interpolate
to resample the feature sequence from source_rate (e.g., 50 Hz) to target_fps
(e.g., 25 fps for BIWI, 30 fps for VOCASET).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAdaptor(nn.Module):
    def __init__(self, source_rate=50, target_fps=25):
        super().__init__()
        self.source_rate = source_rate
        self.target_fps = target_fps

    def forward(self, x, n_target_frames=None):
        """
        Args:
            x: (B, T_source, D) features at source_rate
            n_target_frames: optional explicit target length; if None, computed
                             from ratio target_fps / source_rate

        Returns:
            out: (B, T_target, D) resampled features
        """
        if n_target_frames is None:
            T_source = x.shape[1]
            n_target_frames = int(T_source * self.target_fps / self.source_rate)
            n_target_frames = max(1, n_target_frames)

        x_t = x.permute(0, 2, 1)  # (B, D, T_source)
        out = F.interpolate(x_t, size=n_target_frames, mode="linear", align_corners=False)
        return out.permute(0, 2, 1)  # (B, T_target, D)
