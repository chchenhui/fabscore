# Ice-shelf inverse problem (1D Shallow Shelf Approximation)
# Re-implements the ice-shelf PINN from YaoGroup/pinn_clusters in PyTorch FP64.
# PDE: (2*nu_star*B)^n * u_x - h^n = 0 (momentum balance, drop mass balance).
# Inverse problem: infer B(x) from noisy u(x), h(x) observations.
# Constants ported verbatim from pinn_clusters formulations/_formulations.py.

import torch
import numpy as np

SPY = 60.0 * 60.0 * 24.0 * 365.25
RHOI = 910.0
RHOW = 1028.0
G = 9.81
H0 = 1.0e3
B0 = 1.4688e8
N_GLEN = 3

DELTA = 1.0 - RHOI / RHOW
A_ACC = 0.3 / SPY
Q0 = 4.0e5 / SPY
Z0 = A_ACC ** (1.0 / (N_GLEN + 1)) * (4.0 * B0) ** (N_GLEN / (N_GLEN + 1)) / (RHOI * G * DELTA) ** (N_GLEN / (N_GLEN + 1))
U0 = 400.0 / SPY
LX = U0 * Z0 / A_ACC
H0_ND = H0 / Z0
Q0_ND = Q0 / (U0 * Z0)
NU_STAR = (2.0 * B0) / (RHOI * G * DELTA * Z0) * (U0 / LX) ** (1.0 / N_GLEN)
A0 = (A_ACC * LX) / (U0 * Z0)


def analytic_h(x_np):
    return ((A0 * H0_ND ** (N_GLEN + 1) * (A0 * x_np + Q0_ND) ** (N_GLEN + 1)) /
            (A0 * Q0_ND ** (N_GLEN + 1) - (Q0_ND * H0_ND) ** (N_GLEN + 1) +
             (H0_ND * (A0 * x_np + Q0_ND)) ** (N_GLEN + 1))) ** (1.0 / (N_GLEN + 1))


def analytic_u(x_np):
    return (A0 * x_np + Q0_ND) / analytic_h(x_np)


def generate_data(N_ob=401, noise_level=0.3, seed=0, device="cpu"):
    rng = np.random.RandomState(seed)
    x_np = np.linspace(0.0, 1.0, N_ob)
    u_true_np = analytic_u(x_np)
    h_true_np = analytic_h(x_np)
    B_true_np = np.ones_like(x_np)

    u_noisy_np = u_true_np + noise_level * rng.randn(N_ob)
    h_noisy_np = h_true_np + noise_level * rng.randn(N_ob)

    x_data = torch.tensor(x_np, dtype=torch.float64, device=device).unsqueeze(-1)
    u_data = torch.tensor(u_noisy_np, dtype=torch.float64, device=device).unsqueeze(-1)
    h_data = torch.tensor(h_noisy_np, dtype=torch.float64, device=device).unsqueeze(-1)
    u_true = torch.tensor(u_true_np, dtype=torch.float64, device=device).unsqueeze(-1)
    h_true = torch.tensor(h_true_np, dtype=torch.float64, device=device).unsqueeze(-1)
    B_true = torch.tensor(B_true_np, dtype=torch.float64, device=device).unsqueeze(-1)

    return {
        "x_data": x_data, "u_data": u_data, "h_data": h_data,
        "u_true": u_true, "h_true": h_true, "B_true": B_true,
    }


def pde_residual(model, x_coll):
    x = x_coll.detach().requires_grad_(True)
    out = model(x)
    u = out[:, 0:1]
    h = out[:, 1:2]
    B = out[:, 2:3]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    residual = (2.0 * NU_STAR * B) ** N_GLEN * u_x - h ** N_GLEN
    return residual


def data_loss(model, x_data, u_data, h_data):
    out = model(x_data)
    u_pred = out[:, 0:1]
    h_pred = out[:, 1:2]
    loss_u = torch.mean((u_pred - u_data) ** 2)
    loss_h = torch.mean((h_pred - h_data) ** 2)
    return loss_u + loss_h


def total_loss(model, x_coll, x_data, u_data, h_data, gamma):
    residual = pde_residual(model, x_coll)
    loss_e = torch.mean(residual ** 2)
    loss_d = data_loss(model, x_data, u_data, h_data)
    loss = gamma * loss_e + (1.0 - gamma) * loss_d
    return loss, loss_e, loss_d
