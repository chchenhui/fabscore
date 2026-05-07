import pytest

import torch
from torch import nn


@pytest.mark.parametrize("num_fracs", (1, 4))
@pytest.mark.parametrize("disable", (False, True))
def test_readme(num_fracs, disable):
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    branch = nn.Linear(512, 512)
    residual = torch.randn(2, 1024, 512)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(
            4, num_fracs=num_fracs, disable=disable
        )
    )

    hyper_conn_branch = init_hyper_conn(dim=512, branch=branch)

    residual = expand_stream(residual)
    residual = hyper_conn_branch(residual)
    residual = reduce_stream(residual)

    assert residual.shape == (2, 1024, 512)


def test_manual():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    branch = nn.Linear(512, 512)
    residual = torch.randn(2, 1024, 512)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4)
    )

    hyper_conn = init_hyper_conn(dim=512)

    residual = expand_stream(residual)
    branch_input, add_residual = hyper_conn(residual)
    branch_output = branch(branch_input)
    residual = add_residual(branch_output)
    residual = reduce_stream(residual)

    assert residual.shape == (2, 1024, 512)


@pytest.mark.parametrize("disable", (False, True))
def test_residual_transform(disable):
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    branch = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1), nn.SiLU(), nn.Conv2d(512, 256, 3, padding=1)
    )
    residual_fn = nn.Conv2d(512, 256, 1)

    residual = torch.randn(2, 512, 16, 16)
    before_residual = branch(residual) + residual_fn(residual)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=disable)
    )

    hyper_conn_branch = init_hyper_conn(
        dim=512, branch=branch, channel_first=True, residual_transform=residual_fn
    )

    residual = expand_stream(residual)
    residual = hyper_conn_branch(residual)
    after_residual = reduce_stream(residual)

    assert before_residual.shape == after_residual.shape


def test_disable_matches_residual():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    torch.manual_seed(0)

    branch = nn.Linear(32, 32)
    residual = torch.randn(2, 16, 32)
    expected = branch(residual) + residual

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4, disable=True)
    )

    hyper_conn_branch = init_hyper_conn(dim=32, branch=branch)
    output = reduce_stream(hyper_conn_branch(expand_stream(residual)))

    torch.testing.assert_close(output, expected)


def test_decorate_matches_manual():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    torch.manual_seed(0)

    branch = nn.Linear(32, 32)
    residual = torch.randn(2, 16, 32)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4)
    )

    hyper_conn = init_hyper_conn(dim=32)
    hyper_conn_branch = hyper_conn.decorate_branch(branch)

    expanded = expand_stream(residual)
    output_decorated = reduce_stream(hyper_conn_branch(expanded))

    branch_input, add_residual = hyper_conn(expanded)
    output_manual = reduce_stream(add_residual(branch(branch_input)))

    torch.testing.assert_close(output_decorated, output_manual)


def test_backward_smoke():
    from hyper_connections import get_init_and_expand_reduce_stream_functions

    torch.manual_seed(0)

    branch = nn.Linear(16, 16)
    residual = torch.randn(2, 8, 16, requires_grad=True)

    init_hyper_conn, expand_stream, reduce_stream = (
        get_init_and_expand_reduce_stream_functions(4)
    )

    hyper_conn_branch = init_hyper_conn(dim=16, branch=branch)

    output = reduce_stream(hyper_conn_branch(expand_stream(residual)))
    loss = output.sum()
    loss.backward()

    assert residual.grad is not None


def test_mhc_H_res_constraints():
    from hyper_connections.hyper_connections import HyperConnections, sinkhorn_log

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_res = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)

    assert H_res.min().item() >= 0
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )


def test_mhc_orthostochastic_H_res_constraints():
    from hyper_connections.hyper_connections import (
        HyperConnections,
        orthostochastic_project,
    )

    hc = HyperConnections(
        num_residual_streams=4,
        dim=64,
        mhc=True,
        mhc_h_res_proj="orthostochastic",
    )

    H_res = orthostochastic_project(
        hc.H_res_logits,
        ns_steps=hc.ns_steps,
        ns_eps=hc.ns_eps,
        ns_coeffs=hc.ns_coeffs,
    )

    assert H_res.min().item() >= 0
    # Orthostochastic => doubly-stochastic when O is orthogonal; Newton-Schulz
    # gives an approximation, so use a looser tolerance than Sinkhorn.
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=5e-2,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=5e-2,
    )


def test_mhc_identity_mix_orthostochastic_H_res_constraints():
    from hyper_connections.hyper_connections import (
        HyperConnections,
        orthostochastic_project,
    )

    hc = HyperConnections(
        num_residual_streams=4,
        dim=64,
        mhc=True,
        mhc_h_res_proj="orthostochastic",
        mhc_residual_identity_mix=True,
        mhc_residual_alpha=0.01,
    )

    S = orthostochastic_project(
        hc.H_res_logits,
        ns_steps=hc.ns_steps,
        ns_eps=hc.ns_eps,
        ns_coeffs=hc.ns_coeffs,
    )

    alpha = torch.sigmoid(hc.H_res_alpha_logit)
    I = torch.eye(4, device=S.device, dtype=S.dtype)
    H_res = (1.0 - alpha) * I + alpha * S

    assert H_res.min().item() >= 0
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=5e-2,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=5e-2,
    )


def test_mhc_rejects_multiple_fracs_and_views():
    from hyper_connections.hyper_connections import HyperConnections

    with pytest.raises(AssertionError):
        HyperConnections(num_residual_streams=4, dim=64, mhc=True, num_fracs=2)

    with pytest.raises(AssertionError):
        HyperConnections(num_residual_streams=4, dim=64, mhc=True, num_input_views=2)


def test_mhc_H_pre_H_post_constraints():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    H_pre = torch.softmax(hc.H_pre_logits, dim=-1)
    H_post = torch.softmax(hc.H_post_logits, dim=-1)

    assert H_pre.min().item() >= 0
    assert H_post.min().item() >= 0
    assert torch.allclose(
        H_pre.sum(),
        torch.ones((), device=H_pre.device, dtype=H_pre.dtype),
        atol=1e-6,
    )
    assert torch.allclose(
        H_post.sum(),
        torch.ones((), device=H_post.device, dtype=H_post.dtype),
        atol=1e-6,
    )


def test_mhc_forward_shapes():
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 4, 64, 2, 8
    hc = HyperConnections(num_residual_streams=streams, dim=dim, mhc=True)
    x = torch.randn(batch * streams, seq, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch * streams, seq, dim)


def test_mhc_gradients_flow():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(num_residual_streams=4, dim=64, mhc=True)
    x = torch.randn(8, 8, 64, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.H_res_logits.grad is not None
    assert hc.H_pre_logits.grad is not None
    assert hc.H_post_logits.grad is not None
    assert not torch.isnan(hc.H_res_logits.grad).any()
    assert not torch.isnan(hc.H_pre_logits.grad).any()
    assert not torch.isnan(hc.H_post_logits.grad).any()


def test_mhc_identity_mix_H_res_constraints():
    from hyper_connections.hyper_connections import HyperConnections, sinkhorn_log

    hc = HyperConnections(
        num_residual_streams=4, dim=64, mhc=True, mhc_residual_identity_mix=True
    )
    S = sinkhorn_log(hc.H_res_logits, hc.sinkhorn_iters, hc.sinkhorn_tau)
    alpha = torch.sigmoid(hc.H_res_alpha_logit)
    I = torch.eye(4, device=S.device, dtype=S.dtype)
    H_res = (1 - alpha) * I + alpha * S

    assert H_res.min().item() >= 0
    assert torch.allclose(
        H_res.sum(dim=-1),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )
    assert torch.allclose(
        H_res.sum(dim=-2),
        torch.ones(4, device=H_res.device, dtype=H_res.dtype),
        atol=1e-3,
    )


def test_mhc_identity_mix_alpha_init():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=4,
        dim=64,
        mhc=True,
        mhc_residual_identity_mix=True,
        mhc_residual_alpha=0.01,
    )
    alpha = torch.sigmoid(hc.H_res_alpha_logit)
    assert torch.allclose(alpha, torch.tensor(0.01), atol=1e-3)


def test_mhc_identity_mix_gradients_flow():
    from hyper_connections.hyper_connections import HyperConnections

    hc = HyperConnections(
        num_residual_streams=4, dim=64, mhc=True, mhc_residual_identity_mix=True
    )
    x = torch.randn(8, 8, 64, requires_grad=True)

    branch_input, add_residual = hc(x)
    out = add_residual(branch_input)
    out.sum().backward()

    assert hc.H_res_logits.grad is not None
    assert hc.H_pre_logits.grad is not None
    assert hc.H_post_logits.grad is not None
    assert hc.H_res_alpha_logit.grad is not None
    assert not torch.isnan(hc.H_res_alpha_logit.grad).any()


def test_mhc_identity_mix_forward_shapes():
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 4, 64, 2, 8
    hc = HyperConnections(
        num_residual_streams=streams,
        dim=dim,
        mhc=True,
        mhc_residual_identity_mix=True,
    )
    x = torch.randn(batch * streams, seq, dim)

    branch_input, add_residual = hc(x)
    assert branch_input.shape == (batch, seq, dim)

    branch_output = torch.randn(batch, seq, dim)
    out = add_residual(branch_output)
    assert out.shape == (batch * streams, seq, dim)


def test_hc_width_connection_applies_residual_mixing():
    from hyper_connections.hyper_connections import HyperConnections

    torch.manual_seed(0)

    streams, dim, batch, seq = 2, 4, 1, 3
    hc = HyperConnections(num_residual_streams=streams, dim=dim)

    x0 = torch.zeros(batch, seq, dim)
    x1 = torch.ones(batch, seq, dim)
    x = torch.cat((x0, x1), dim=0)

    with torch.no_grad():
        hc.static_alpha.zero_()
        hc.static_alpha[1, 1] = 1.0  # residual stream 0 output <- stream 1
        hc.static_alpha[0, 2] = 1.0  # residual stream 1 output <- stream 0

    branch_input, residuals, _ = hc.width_connection(x)

    torch.testing.assert_close(branch_input, torch.zeros(batch, seq, dim))
    torch.testing.assert_close(residuals, torch.cat((x1, x0), dim=0))


def test_hc_depth_connection_applies_branch_distribution():
    from hyper_connections.hyper_connections import HyperConnections

    streams, dim, batch, seq = 2, 4, 1, 3
    hc = HyperConnections(num_residual_streams=streams, dim=dim)

    branch_output = torch.randn(batch, seq, dim)
    residuals = torch.zeros(batch * streams, seq, dim)

    # beta has shape (b, seq, f1, s, f2); for num_fracs=1 => f1=f2=1
    beta = torch.zeros(batch, seq, 1, streams, 1)
    beta[..., 0, :] = 1.0

    out = hc.depth_connection(branch_output, residuals, beta=beta)
    out0, out1 = out.chunk(2, dim=0)

    torch.testing.assert_close(out0, branch_output)
    torch.testing.assert_close(out1, torch.zeros_like(branch_output))


def test_hc_channel_first_width_connection_applies_residual_mixing():
    from hyper_connections.hyper_connections_channel_first import HyperConnections

    torch.manual_seed(0)

    streams, dim, batch, h, w = 2, 4, 1, 2, 2
    hc = HyperConnections(num_residual_streams=streams, dim=dim)

    x0 = torch.zeros(batch, dim, h, w)
    x1 = torch.ones(batch, dim, h, w)
    x = torch.cat((x0, x1), dim=0)

    with torch.no_grad():
        hc.static_alpha.zero_()
        hc.static_alpha[1, 1] = 1.0  # residual stream 0 output <- stream 1
        hc.static_alpha[0, 2] = 1.0  # residual stream 1 output <- stream 0

    branch_input, residuals, _ = hc.width_connection(x)

    torch.testing.assert_close(branch_input, torch.zeros(batch, dim, h, w))
    torch.testing.assert_close(residuals, torch.cat((x1, x0), dim=0))
