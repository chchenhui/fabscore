"""
Shared diagnostics logging module for orthostochastic mHC experiments.

Provides three diagnostic trackers:
1. GradientSpikeTracker - gradient spike ratio r_t = g_t / median(g_{t-100:t-1})
2. DSErrorLogger - doubly-stochastic constraint error for H_res matrices
3. OrthogonalityResidualLogger - ||O O^T - I||_F for Newton-Schulz orthogonal matrices

All diagnostics use torch.no_grad() and do not affect training gradients.
Results are saved to JSON files in a specified output directory.
"""

import json
import os
from collections import deque

import torch
import numpy as np


class GradientSpikeTracker:
    """Track gradient norms and compute spike ratio r_t = g_t / median(g_{t-100:t-1}).

    Spike ratio is computed for t > 200 (after warmup). r_max = max_t r_t is
    reported at the end of training.
    """

    def __init__(self, warmup_steps=200, window_size=100):
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.grad_norms = []
        self.spike_ratios = []
        self.history = deque(maxlen=window_size)

    @torch.no_grad()
    def step(self, iter_num, grad_norm):
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        self.grad_norms.append({"iter": iter_num, "grad_norm": grad_norm})

        r_t = None
        if iter_num > self.warmup_steps and len(self.history) >= self.window_size:
            median_val = float(np.median(list(self.history)))
            if median_val > 0:
                r_t = grad_norm / median_val
                self.spike_ratios.append({"iter": iter_num, "r_t": r_t})

        self.history.append(grad_norm)
        return r_t

    def get_r_max(self):
        if not self.spike_ratios:
            return None
        return max(entry["r_t"] for entry in self.spike_ratios)

    def save(self, out_dir, prefix=""):
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"{prefix}gradient_spikes.json")
        result = {
            "grad_norms": self.grad_norms,
            "spike_ratios": self.spike_ratios,
            "r_max": self.get_r_max(),
        }
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        return fname


class DSErrorLogger:
    """Compute doubly-stochastic error for H_res matrices.

    For each layer's H_res: max_i |sum_j H_res[i,j] - 1| (row error)
    and max_j |sum_i H_res[i,j] - 1| (col error). Logs the maximum over
    all layers.
    """

    def __init__(self):
        self.records = []

    @torch.no_grad()
    def step(self, iter_num, model, mhc_h_res_proj, sinkhorn_iters, sinkhorn_tau,
             ns_steps, ns_eps, ns_coeffs):
        from hyper_connections.hyper_connections import sinkhorn_log, orthostochastic_project

        max_row_err = 0.0
        max_col_err = 0.0
        per_layer = []

        for i, block in enumerate(model.transformer.h):
            for name, hc in [("attn", block.hc_attn), ("mlp", block.hc_mlp)]:
                if not hasattr(hc, "H_res_logits"):
                    continue
                logits = hc.H_res_logits

                if mhc_h_res_proj == "orthostochastic":
                    H_res = orthostochastic_project(
                        logits, ns_steps=ns_steps, ns_eps=ns_eps, ns_coeffs=ns_coeffs
                    )
                else:
                    H_res = sinkhorn_log(logits, sinkhorn_iters, sinkhorn_tau)

                row_err = (H_res.sum(dim=-1) - 1.0).abs().max().item()
                col_err = (H_res.sum(dim=-2) - 1.0).abs().max().item()

                max_row_err = max(max_row_err, row_err)
                max_col_err = max(max_col_err, col_err)

                per_layer.append({
                    "layer": i, "sub": name,
                    "row_err": row_err, "col_err": col_err,
                })

        self.records.append({
            "iter": iter_num,
            "max_row_err": max_row_err,
            "max_col_err": max_col_err,
            "per_layer": per_layer,
        })
        return max_row_err, max_col_err

    def save(self, out_dir, prefix=""):
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"{prefix}ds_error.json")
        with open(fname, "w") as f:
            json.dump(self.records, f, indent=2)
        return fname


class OrthogonalityResidualLogger:
    """Compute ||O O^T - I||_F for the Newton-Schulz orthogonal matrix O (before squaring).

    Only applicable for orthostochastic runs.
    """

    def __init__(self):
        self.records = []

    @torch.no_grad()
    def step(self, iter_num, model, ns_steps, ns_eps, ns_coeffs):
        from hyper_connections.hyper_connections import zeropower_via_newtonschulz

        per_layer = []
        max_residual = 0.0

        for i, block in enumerate(model.transformer.h):
            for name, hc in [("attn", block.hc_attn), ("mlp", block.hc_mlp)]:
                if not hasattr(hc, "H_res_logits"):
                    continue
                logits = hc.H_res_logits
                O = zeropower_via_newtonschulz(
                    logits, steps=ns_steps, eps=ns_eps, coeffs=ns_coeffs
                )
                n = O.shape[0]
                I = torch.eye(n, device=O.device, dtype=O.dtype)
                residual = torch.norm(O @ O.T - I, p="fro").item()
                max_residual = max(max_residual, residual)
                per_layer.append({
                    "layer": i, "sub": name, "orth_residual": residual,
                })

        self.records.append({
            "iter": iter_num,
            "max_orth_residual": max_residual,
            "per_layer": per_layer,
        })
        return max_residual

    def save(self, out_dir, prefix=""):
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"{prefix}orth_residual.json")
        with open(fname, "w") as f:
            json.dump(self.records, f, indent=2)
        return fname


class DiagnosticsManager:
    """Convenience wrapper that manages all three diagnostic trackers.

    Usage in training loop:
        diag = DiagnosticsManager(out_dir="logs/run_name", is_orthostochastic=True, ...)
        # after each training step:
        diag.step(iter_num, grad_norm, model)
        # after training:
        diag.save()
    """

    def __init__(self, out_dir, is_orthostochastic=False,
                 mhc_h_res_proj="sinkhorn", sinkhorn_iters=10, sinkhorn_tau=0.05,
                 ns_steps=10, ns_eps=1e-7, ns_coeffs=(3.0, -3.2, 1.2),
                 log_interval=10, warmup_steps=200, window_size=100):
        self.out_dir = out_dir
        self.is_orthostochastic = is_orthostochastic
        self.mhc_h_res_proj = mhc_h_res_proj
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tau = sinkhorn_tau
        self.ns_steps = ns_steps
        self.ns_eps = ns_eps
        self.ns_coeffs = ns_coeffs
        self.log_interval = log_interval

        self.grad_tracker = GradientSpikeTracker(warmup_steps=warmup_steps,
                                                  window_size=window_size)
        self.ds_logger = DSErrorLogger()
        self.orth_logger = OrthogonalityResidualLogger() if is_orthostochastic else None

    @torch.no_grad()
    def step(self, iter_num, grad_norm, model):
        self.grad_tracker.step(iter_num, grad_norm)

        if iter_num % self.log_interval == 0:
            self.ds_logger.step(
                iter_num, model, self.mhc_h_res_proj,
                self.sinkhorn_iters, self.sinkhorn_tau,
                self.ns_steps, self.ns_eps, self.ns_coeffs,
            )
            if self.orth_logger is not None:
                self.orth_logger.step(
                    iter_num, model, self.ns_steps, self.ns_eps, self.ns_coeffs,
                )

    def save(self, prefix=""):
        paths = {}
        paths["gradient_spikes"] = self.grad_tracker.save(self.out_dir, prefix)
        paths["ds_error"] = self.ds_logger.save(self.out_dir, prefix)
        if self.orth_logger is not None:
            paths["orth_residual"] = self.orth_logger.save(self.out_dir, prefix)
        summary = {
            "r_max": self.grad_tracker.get_r_max(),
            "final_ds_error": self.ds_logger.records[-1] if self.ds_logger.records else None,
        }
        if self.orth_logger and self.orth_logger.records:
            summary["final_orth_residual"] = self.orth_logger.records[-1]
        summary_path = os.path.join(self.out_dir, f"{prefix}diagnostics_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        paths["summary"] = summary_path
        return paths

    def get_summary(self):
        summary = {"r_max": self.grad_tracker.get_r_max()}
        if self.ds_logger.records:
            last = self.ds_logger.records[-1]
            summary["final_max_row_err"] = last["max_row_err"]
            summary["final_max_col_err"] = last["max_col_err"]
        if self.orth_logger and self.orth_logger.records:
            summary["final_max_orth_residual"] = self.orth_logger.records[-1]["max_orth_residual"]
        return summary
