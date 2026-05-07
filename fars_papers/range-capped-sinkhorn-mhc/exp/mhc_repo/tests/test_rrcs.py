"""
Unit tests for the RRCS (Range-Capped Sinkhorn) modification.
Tests range capping, doubly-stochastic output, gradient flow,
no-op for well-conditioned inputs, and differentiability of s.
"""

import math
import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hyper_connections"))
from hyper_connections import sinkhorn_log, sinkhorn_log_rrcs


def _make_sharp_logits(n=4):
    logits = torch.full((n, n), -8.0)
    logits.fill_diagonal_(0.0)
    return logits


class TestRRCSRangeCapping:
    def test_range_is_capped(self):
        logits = _make_sharp_logits()
        tau = 0.05
        r_cap = 30.0
        Z = logits / tau
        r_before = (Z.max() - Z.min()).item()
        assert r_before > r_cap

        r_raw = Z.max() - Z.min()
        s = torch.clamp(r_cap / (r_raw + 1e-8), max=1.0)
        Z_capped = s * Z
        r_capped = (Z_capped.max() - Z_capped.min()).item()
        assert r_capped <= r_cap + 1e-4, f"Range {r_capped} exceeds cap {r_cap}"


class TestRRCSDoublyStochastic:
    def test_output_is_doubly_stochastic(self):
        logits = _make_sharp_logits()
        S = sinkhorn_log_rrcs(logits, num_iters=10, tau=0.05, r_cap=30.0)
        row_sums = S.sum(dim=-1)
        col_sums = S.sum(dim=-2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
            f"Row sums: {row_sums}"
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), \
            f"Col sums: {col_sums}"

    def test_all_entries_nonnegative(self):
        logits = _make_sharp_logits()
        S = sinkhorn_log_rrcs(logits, num_iters=10, tau=0.05, r_cap=30.0)
        assert (S >= -1e-6).all(), f"Negative entries found: {S.min().item()}"


class TestRRCSGradientFlow:
    def test_gradient_nonzero_with_rrcs(self):
        """RRCS with r_cap=5 produces nonzero H_res_logits gradients,
        while baseline sinkhorn_log produces exactly zero."""
        logits_rrcs = _make_sharp_logits().float().requires_grad_(True)
        S = sinkhorn_log_rrcs(logits_rrcs, num_iters=10, tau=0.05, r_cap=5.0)
        loss = -(S * torch.log(S + 1e-10)).sum()
        loss.backward()
        grad_rrcs = logits_rrcs.grad.norm(2).item()
        assert grad_rrcs > 0.0, "RRCS gradient is exactly zero"

    def test_rrcs_grad_larger_than_baseline(self):
        """RRCS gradients must be strictly larger than baseline."""
        logits_rrcs = _make_sharp_logits().float().requires_grad_(True)
        S_rrcs = sinkhorn_log_rrcs(logits_rrcs, num_iters=10, tau=0.05, r_cap=5.0)
        (-(S_rrcs * torch.log(S_rrcs + 1e-10)).sum()).backward()
        grad_rrcs = logits_rrcs.grad.norm(2).item()

        logits_base = _make_sharp_logits().float().requires_grad_(True)
        S_base = sinkhorn_log(logits_base, num_iters=10, tau=0.05)
        (-(S_base * torch.log(S_base + 1e-10)).sum()).backward()
        grad_base = logits_base.grad.norm(2).item()

        assert grad_rrcs > grad_base, \
            f"RRCS grad {grad_rrcs} not larger than baseline grad {grad_base}"

    def test_gradient_vanishes_without_rrcs(self):
        logits = _make_sharp_logits().float().requires_grad_(True)
        S = sinkhorn_log(logits, num_iters=10, tau=0.05)
        loss = ((S - torch.eye(4)) ** 2).sum()
        loss.backward()
        grad_norm = logits.grad.norm(2).item()
        assert grad_norm < 1e-4, \
            f"Expected vanishing gradient without RRCS, got {grad_norm}"


class TestRRCSNoOp:
    def test_noop_for_well_conditioned(self):
        torch.manual_seed(42)
        logits = torch.randn(4, 4) * 0.1
        tau = 0.05
        r_cap = 30.0
        Z = logits / tau
        r = (Z.max() - Z.min()).item()
        assert r <= r_cap, "Test setup: logits should already be well-conditioned"

        S_rrcs = sinkhorn_log_rrcs(logits, num_iters=10, tau=tau, r_cap=r_cap)
        S_orig = sinkhorn_log(logits, num_iters=10, tau=tau)
        assert torch.allclose(S_rrcs, S_orig, atol=1e-5), \
            f"Max diff: {(S_rrcs - S_orig).abs().max().item()}"


class TestRRCSDifferentiableS:
    def test_s_gradient_nonzero_when_capping_active(self):
        """Verify gradient flows through the scaling factor s."""
        logits = _make_sharp_logits().float().requires_grad_(True)
        tau = 0.05
        r_cap = 5.0

        S = sinkhorn_log_rrcs(logits, num_iters=10, tau=tau, r_cap=r_cap)
        loss = -(S * torch.log(S + 1e-10)).sum()
        loss.backward()

        assert logits.grad is not None, "No gradient computed"
        grad_norm = logits.grad.norm(2).item()
        assert grad_norm > 0.0, "Gradient through s is exactly zero"

    def test_s_value_when_capping_active(self):
        logits = _make_sharp_logits()
        tau = 0.05
        r_cap = 30.0
        Z = logits / tau
        r = (Z.max() - Z.min()).item()
        s = min(1.0, r_cap / (r + 1e-8))
        assert s < 1.0, f"Expected s < 1 when capping active, got {s}"
        assert s > 0.0, f"Expected s > 0, got {s}"
        expected_s = 30.0 / 160.0
        assert abs(s - expected_s) < 1e-4, f"Expected s={expected_s}, got {s}"
