# 2D Poisson forward problem: -Laplacian(u) = f on [0,1]^2 with Dirichlet BCs.
# Standard PINN benchmark for testing optimizer and sampler behavior on a
# canonical elliptic PDE. Implements exact solution, forcing term, and loss.
# Exact solution: u*(x,y) = sin(pi*x)*sin(pi*y), f(x,y) = 2*pi^2*sin(pi*x)*sin(pi*y).

import torch
import math


def exact_solution(xy):
    return torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])


def forcing(xy):
    return 2.0 * math.pi**2 * torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])


def generate_boundary_points(N_per_edge=200, device="cpu"):
    t = torch.linspace(0.0, 1.0, N_per_edge, dtype=torch.float64, device=device).unsqueeze(-1)
    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)
    bottom = torch.cat([t, zeros], dim=1)
    top = torch.cat([t, ones], dim=1)
    left = torch.cat([zeros, t], dim=1)
    right = torch.cat([ones, t], dim=1)
    return torch.cat([bottom, top, left, right], dim=0)


def pde_residual(model, xy_coll):
    xy = xy_coll.detach().requires_grad_(True)
    u = model(xy)
    grad_u = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    laplacian = u_xx + u_yy
    f = forcing(xy)
    return laplacian + f


def bc_loss(model, xy_bc):
    u_pred = model(xy_bc)
    return torch.mean(u_pred ** 2)


def total_loss(model, xy_coll, xy_bc, lambda_bc=1.0):
    residual = pde_residual(model, xy_coll)
    loss_pde = torch.mean(residual ** 2)
    loss_bc = bc_loss(model, xy_bc)
    loss = loss_pde + lambda_bc * loss_bc
    return loss, loss_pde, loss_bc


@torch.no_grad()
def eval_rel_l2(model, N_grid=100, device="cpu"):
    x = torch.linspace(0.0, 1.0, N_grid, dtype=torch.float64, device=device)
    y = torch.linspace(0.0, 1.0, N_grid, dtype=torch.float64, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    u_pred = model(xy)
    u_exact = exact_solution(xy)
    rel_l2 = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    return rel_l2.item()
