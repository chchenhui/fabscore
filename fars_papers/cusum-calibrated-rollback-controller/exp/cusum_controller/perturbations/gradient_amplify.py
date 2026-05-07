# Gradient amplification perturbation utilities.
# Implements step and ramp gradient amplification for controlled destabilization.
# Applied in-place to param.grad after loss.backward() and before optimizer.step().


def apply_perturbation(model, step, perturb_type, zeta=300, window_start=120, window_len=10):
    if perturb_type == "nominal" or perturb_type is None:
        return False

    window_end = window_start + window_len
    if step < window_start or step >= window_end:
        return False

    i = step - window_start

    if perturb_type == "step":
        scale = zeta
    elif perturb_type == "ramp":
        scale = 1.0 + (zeta - 1.0) * i / (window_len - 1)
    else:
        raise ValueError(f"Unknown perturb_type: {perturb_type}")

    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)

    return True
