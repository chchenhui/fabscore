# Collocation point samplers: fixed, resampled, and overlap-resampled.
# 1D sampling convention from pinn_clusters: uniform [0,1] then cubed (bias toward x=0).
# 2D samplers use uniform sampling on [0,1]^2 (no cubing).

import torch


def _sample_cubed(n, device="cpu"):
    pts = torch.rand(n, 1, dtype=torch.float64, device=device)
    return pts ** 3


class FixedSampler:

    def __init__(self, N, seed=0, device="cpu"):
        gen = torch.Generator(device=device).manual_seed(seed)
        pts = torch.rand(N, 1, dtype=torch.float64, device=device, generator=gen)
        self.points = pts ** 3

    def sample(self):
        return self.points


class ResampleSampler:

    def __init__(self, N, device="cpu"):
        self.N = N
        self.device = device

    def sample(self):
        return _sample_cubed(self.N, self.device)


class OverlapResampleSampler:

    def __init__(self, N, overlap_frac=0.5, device="cpu"):
        self.N = N
        self.n_keep = int(N * overlap_frac)
        self.n_fresh = N - self.n_keep
        self.device = device
        self.prev_points = _sample_cubed(N, device)

    def sample(self):
        perm = torch.randperm(self.N, device=self.device)
        keep = self.prev_points[perm[:self.n_keep]]
        fresh = _sample_cubed(self.n_fresh, self.device)
        full_set = torch.cat([keep, fresh], dim=0)
        overlap_set = keep.clone()
        self.prev_points = full_set
        return full_set, overlap_set


class ResampleSampler2D:

    def __init__(self, N, device="cpu"):
        self.N = N
        self.device = device

    def sample(self):
        return torch.rand(self.N, 2, dtype=torch.float64, device=self.device)


class FixedSampler2D:

    def __init__(self, N, seed=0, device="cpu"):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.points = torch.rand(N, 2, dtype=torch.float64, device=device, generator=gen)

    def sample(self):
        return self.points


class OverlapResampleSampler2D:

    def __init__(self, N, overlap_frac=0.5, device="cpu"):
        self.N = N
        self.n_keep = int(N * overlap_frac)
        self.n_fresh = N - self.n_keep
        self.device = device
        self.prev_points = torch.rand(N, 2, dtype=torch.float64, device=device)

    def sample(self):
        perm = torch.randperm(self.N, device=self.device)
        keep = self.prev_points[perm[:self.n_keep]]
        fresh = torch.rand(self.n_fresh, 2, dtype=torch.float64, device=self.device)
        full_set = torch.cat([keep, fresh], dim=0)
        overlap_set = keep.clone()
        self.prev_points = full_set
        return full_set, overlap_set
