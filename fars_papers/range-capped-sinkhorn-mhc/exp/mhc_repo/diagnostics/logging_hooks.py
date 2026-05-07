"""
DiagnosticLogger for mHC experiments.

Collects per-step diagnostics every `diag_interval` steps:
  - H_res_logits gradient L2 norms (per layer + aggregates)
  - Sinkhorn input conditioning (log-range r = max(Z)-min(Z) per layer)
  - Doubly-stochastic error (max row/col sum deviation from 1)
  - H_res row entropy
  - Global gradient norm (passed in from training loop)

Writes CSV to `{out_dir}/diagnostics.csv` and saves initial H_res_logits
snapshot to `{out_dir}/h_res_logits_init.pt` at step 0.
"""

import csv
import os
from pathlib import Path

import torch
import torch.nn as nn


def _find_hc_modules(model):
    hcs = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'H_res_logits') and hasattr(mod, 'mhc') and mod.mhc:
            hcs.append((name, mod))
    return hcs


def _sinkhorn_log(logits, num_iters, tau, r_cap=None):
    n = logits.shape[-1]
    Z = logits / tau
    if r_cap is not None:
        r = Z.max() - Z.min()
        s = torch.clamp(r_cap / (r + 1e-8), max=1.0)
        Z = s * Z
    log_marginal = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    u = torch.zeros(logits.shape[:-1], device=Z.device, dtype=Z.dtype)
    v = torch.zeros_like(u)
    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1)
        v = log_marginal - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2)
    return torch.exp(Z + u.unsqueeze(-1) + v.unsqueeze(-2))


class DiagnosticLogger:
    def __init__(self, out_dir: str, condition: str, seed: int, diag_interval: int = 10, r_cap: float = 30.0, mhc_rrcs: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.condition = condition
        self.seed = seed
        self.diag_interval = diag_interval
        self.r_cap = r_cap
        self.mhc_rrcs = mhc_rrcs
        self._csv_path = self.out_dir / "diagnostics.csv"
        self._init_snapshot_path = self.out_dir / "h_res_logits_init.pt"
        self._csv_file = None
        self._csv_writer = None
        self._initialized = False
        self._hc_cache = None

    def _get_hcs(self, model):
        if self._hc_cache is None:
            self._hc_cache = _find_hc_modules(model)
        return self._hc_cache

    def _init_csv(self, fieldnames):
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()
        self._initialized = True

    def _save_init_snapshot(self, model):
        hcs = self._get_hcs(model)
        snapshot = {}
        for name, hc in hcs:
            snapshot[name] = hc.H_res_logits.detach().cpu().clone()
        torch.save(snapshot, self._init_snapshot_path)

    @torch.no_grad()
    def log_step(self, model, iter_num, grad_norm=None):
        if iter_num == 0:
            self._save_init_snapshot(model)

        if iter_num % self.diag_interval != 0:
            return

        hcs = self._get_hcs(model)
        if not hcs:
            return

        h_res_grad_norms = []
        sinkhorn_ranges = []
        ds_row_errors = []
        ds_col_errors = []
        entropies = []

        for name, hc in hcs:
            logits = hc.H_res_logits
            tau = getattr(hc, 'sinkhorn_tau', None) or getattr(hc, 'mhc_tau', 0.05)
            num_iters = getattr(hc, 'sinkhorn_iters', None) or getattr(hc, 'mhc_num_iters', 10)

            if logits.grad is not None:
                gn = logits.grad.detach().float().norm(2).item()
            else:
                gn = 0.0
            h_res_grad_norms.append(gn)

            Z = logits.detach().float() / tau
            r = (Z.max() - Z.min()).item()
            sinkhorn_ranges.append(r)

            rrcs_r_cap = self.r_cap if self.mhc_rrcs else None
            H_res = _sinkhorn_log(logits.detach().float(), num_iters, tau, r_cap=rrcs_r_cap)
            row_sums = H_res.sum(dim=-1)
            col_sums = H_res.sum(dim=-2)
            ds_row_err = (row_sums - 1.0).abs().max().item()
            ds_col_err = (col_sums - 1.0).abs().max().item()
            ds_row_errors.append(ds_row_err)
            ds_col_errors.append(ds_col_err)

            n = H_res.shape[-1]
            ent = -(H_res * torch.log(H_res + 1e-10)).sum() / n
            entropies.append(ent.item())

        rrcs_s_values = []
        for r in sinkhorn_ranges:
            s_val = min(1.0, self.r_cap / (r + 1e-8))
            rrcs_s_values.append(s_val)

        grad_norms_t = torch.tensor(h_res_grad_norms)
        ranges_t = torch.tensor(sinkhorn_ranges)
        rrcs_s_t = torch.tensor(rrcs_s_values)

        row = {
            "iter": iter_num,
            "grad_norm_global": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm if grad_norm is not None else 0.0),
            "h_res_grad_median": grad_norms_t.median().item(),
            "h_res_grad_mean": grad_norms_t.mean().item(),
            "h_res_grad_max": grad_norms_t.max().item(),
            "h_res_grad_min": grad_norms_t.min().item(),
            "sinkhorn_range_mean": ranges_t.mean().item(),
            "sinkhorn_range_max": ranges_t.max().item(),
            "sinkhorn_range_p25": ranges_t.quantile(0.25).item(),
            "sinkhorn_range_p50": ranges_t.median().item(),
            "sinkhorn_range_p75": ranges_t.quantile(0.75).item(),
            "ds_row_error_mean": sum(ds_row_errors) / len(ds_row_errors),
            "ds_row_error_max": max(ds_row_errors),
            "ds_col_error_mean": sum(ds_col_errors) / len(ds_col_errors),
            "ds_col_error_max": max(ds_col_errors),
            "entropy_mean": sum(entropies) / len(entropies),
            "entropy_min": min(entropies),
            "entropy_max": max(entropies),
            "rrcs_s_mean": rrcs_s_t.mean().item(),
            "rrcs_s_min": rrcs_s_t.min().item(),
            "rrcs_s_frac_active": (rrcs_s_t < 1.0).float().mean().item(),
        }

        for i, (name, hc) in enumerate(hcs):
            row[f"h_res_grad_L{i}"] = h_res_grad_norms[i]
            row[f"sinkhorn_range_L{i}"] = sinkhorn_ranges[i]
            row[f"ds_row_err_L{i}"] = ds_row_errors[i]
            row[f"ds_col_err_L{i}"] = ds_col_errors[i]
            row[f"entropy_L{i}"] = entropies[i]
            row[f"rrcs_s_L{i}"] = rrcs_s_values[i]

        if not self._initialized:
            self._init_csv(list(row.keys()))

        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
