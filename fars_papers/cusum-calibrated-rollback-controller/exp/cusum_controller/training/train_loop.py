# Main training loop with controller integration.
# Supports all three conditions: no-controller, Or-epsilon, CUSUM-epsilon.
# Handles accept/rollback decisions, metric logging, WandB integration, and snapshot saving.

import copy
import numpy as np
import torch
import torch.nn as nn
import wandb

from cusum_controller.data.cifar10 import get_datasets, get_train_loader, sample_probe_set
from cusum_controller.models.resnet import create_resnet18, probe_loss
from cusum_controller.perturbations.gradient_amplify import apply_perturbation


def run_single(
    seed,
    perturb_type,
    controller,
    controller_name="no_controller",
    num_steps=250,
    batch_size=128,
    lr=1e-3,
    weight_decay=0.01,
    alpha=0.1,
    zeta=300,
    window_start=120,
    window_len=10,
    probe_size=16,
    data_root="./data",
    device=None,
    use_wandb=True,
    wandb_project=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    train_dataset, test_dataset = get_datasets(data_root=data_root)
    train_loader = get_train_loader(train_dataset, batch_size=batch_size, seed=seed)
    probe_images, probe_labels = sample_probe_set(test_dataset, probe_size=probe_size, seed=seed)

    model = create_resnet18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    y0 = probe_loss(model, probe_images, probe_labels, device)
    controller.initialize(y0)

    snapshot_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snapshot_optim_state = copy.deepcopy(optimizer.state_dict())

    run_name = f"{controller_name}_seed{seed}_{perturb_type}"
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=run_name,
            tags=[f"seed={seed}", f"perturb={perturb_type}", f"controller={controller_name}"],
            config={
                "seed": seed,
                "perturb_type": perturb_type,
                "controller": controller_name,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "alpha": alpha,
                "zeta": zeta,
                "window_start": window_start,
                "window_len": window_len,
                "probe_size": probe_size,
            },
            reinit=True,
        )

    probe_losses = [y0]
    train_losses = []
    innovations = [0.0]
    ema_refs = [y0]
    decisions = [True]
    perturbation_active = [False]
    standardized_innovations = [0.0]
    cusum_stats = [0.0]
    has_cusum = False

    model.train()
    step = 0
    data_iter = iter(train_loader)

    while step < num_steps:
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        perturbed = apply_perturbation(
            model, step, perturb_type, zeta=zeta,
            window_start=window_start, window_len=window_len,
        )

        snapshot_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        snapshot_optim_state = copy.deepcopy(optimizer.state_dict())

        optimizer.step()

        y_proposed = probe_loss(model, probe_images, probe_labels, device)
        accepted, info = controller.decide(y_proposed)

        if accepted:
            pass
        else:
            model.load_state_dict(snapshot_model_state)
            optimizer.load_state_dict(snapshot_optim_state)
            y_proposed = probe_loss(model, probe_images, probe_labels, device)

        train_losses.append(loss.item())
        probe_losses.append(y_proposed)
        innovations.append(info["innovation"])
        ema_refs.append(info["y_hat"])
        decisions.append(accepted)
        perturbation_active.append(perturbed)

        if "r_t" in info:
            has_cusum = True
            standardized_innovations.append(info["r_t"])
            cusum_stats.append(info["S_t"])

        log_dict = {
            "step": step,
            "probe_loss": y_proposed,
            "training_loss": loss.item(),
            "innovation": info["innovation"],
            "ema_reference": info["y_hat"],
            "perturbation_active": int(perturbed),
            "accepted": int(accepted),
        }
        if "r_t" in info:
            log_dict["standardized_innovation"] = info["r_t"]
            log_dict["cusum_statistic"] = info["S_t"]

        if use_wandb:
            wandb.log(log_dict, step=step)

        if step % 50 == 0 or step == num_steps - 1:
            print(
                f"  [{run_name}] step={step:03d} "
                f"train_loss={loss.item():.4f} "
                f"probe={y_proposed:.4f} "
                f"innov={info['innovation']:.4f} "
                f"accepted={accepted}"
            )

        step += 1

    if use_wandb:
        wandb.finish()

    result = {
        "seed": seed,
        "perturb_type": perturb_type,
        "controller": controller_name,
        "probe_losses": np.array(probe_losses, dtype=np.float64),
        "train_losses": np.array(train_losses, dtype=np.float64),
        "innovations": np.array(innovations, dtype=np.float64),
        "ema_refs": np.array(ema_refs, dtype=np.float64),
        "decisions": np.array(decisions, dtype=bool),
        "perturbation_active": np.array(perturbation_active, dtype=bool),
    }
    if has_cusum:
        result["standardized_innovations"] = np.array(standardized_innovations, dtype=np.float64)
        result["cusum_stats"] = np.array(cusum_stats, dtype=np.float64)
    return result
