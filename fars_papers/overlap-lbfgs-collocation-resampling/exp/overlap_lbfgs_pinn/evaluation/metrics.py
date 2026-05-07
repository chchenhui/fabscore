# Evaluation metrics: relative L2 error for B, u, h predictions.
# Error formula from task spec: err = (1/k) * sum((pred_i - true_i)^2).

import torch


@torch.no_grad()
def relative_l2_errors(model, x_eval, u_true, h_true, B_true):
    out = model(x_eval)
    u_pred = out[:, 0:1]
    h_pred = out[:, 1:2]
    B_pred = out[:, 2:3]

    k = x_eval.shape[0]
    B_err = torch.sum((B_pred - B_true) ** 2).item() / k
    u_err = torch.sum((u_pred - u_true) ** 2).item() / k
    h_err = torch.sum((h_pred - h_true) ** 2).item() / k

    return {"B_err": B_err, "u_err": u_err, "h_err": h_err}
