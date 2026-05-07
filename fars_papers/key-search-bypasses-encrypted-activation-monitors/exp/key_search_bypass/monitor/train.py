# Monitor training loop: trains the MLP probe on extracted activations
# with BCE loss, Adam optimizer. Includes wandb logging, NaN/explosion guard,
# best checkpoint tracking, and layer sweep.

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score

from .probe import MLPProbe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_probe(
    train_acts,
    train_labels,
    val_acts,
    val_labels,
    input_dim=3584,
    hidden_dim=32,
    lr=1e-4,
    weight_decay=1.0,
    epochs=5000,
    seed=42,
    run_name="probe",
    use_wandb=True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    probe = MLPProbe(input_dim, hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    X_train = torch.tensor(train_acts, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_val = torch.tensor(val_acts, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "key-search-bypasses-encrypted-activation-monitors"),
            name=run_name,
            config={
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "seed": seed,
                "train_size": len(train_labels),
                "val_size": len(val_labels),
            },
            reinit=True,
        )

    best_auroc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        probe.train()
        optimizer.zero_grad()
        preds = probe(X_train)
        loss = criterion(preds, y_train)

        if math.isnan(loss.item()) or loss.item() > 10:
            msg = f"Training collapsed at epoch {epoch}: loss={loss.item()}"
            print(msg)
            if wandb_run:
                import wandb
                wandb.log({"error": msg, "epoch": epoch})
                wandb.finish()
            raise RuntimeError(msg)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            probe.eval()
            with torch.no_grad():
                val_preds = probe(X_val).cpu().numpy().flatten()
                val_loss = criterion(probe(X_val), y_val).item()
            auroc = roc_auc_score(val_labels, val_preds)
            train_loss = loss.item()

            if use_wandb and wandb_run:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auroc": auroc,
                })

            if epoch % 100 == 0 or epoch == 1:
                print(f"  Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auroc={auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                best_epoch = epoch

    if best_state is not None:
        probe.load_state_dict(best_state)
    print(f"  Best val AUROC: {best_auroc:.4f} at epoch {best_epoch}")

    if wandb_run:
        import wandb
        wandb.log({"best_val_auroc": best_auroc, "best_epoch": best_epoch})
        wandb.finish()

    return probe, best_auroc, best_epoch


def layer_sweep(
    activation_dir,
    layers,
    input_dim=3584,
    hidden_dim=32,
    epochs=5000,
    seed=42,
    use_wandb=True,
):
    activation_dir = Path(activation_dir)
    train_labels = np.load(activation_dir / "train_labels.npy")
    test_labels = np.load(activation_dir / "test_labels.npy")

    results = {}
    best_layer = None
    best_auroc = 0.0

    for layer_idx in layers:
        layer_dir = activation_dir / f"layer_{layer_idx}"
        train_acts = np.load(layer_dir / "train.npy")
        test_acts = np.load(layer_dir / "test.npy")

        print(f"\n=== Layer {layer_idx} ===")
        probe, auroc, best_epoch = train_probe(
            train_acts, train_labels,
            test_acts, test_labels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            epochs=epochs,
            seed=seed,
            run_name=f"layer_sweep_L{layer_idx}",
            use_wandb=use_wandb,
        )
        results[layer_idx] = {"auroc": auroc, "best_epoch": best_epoch}

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer_idx

    print(f"\nBest layer: {best_layer} with AUROC={best_auroc:.4f}")
    return best_layer, results


def train_multi_seed(
    activation_dir,
    layer_idx,
    seeds,
    output_dir,
    input_dim=3584,
    hidden_dim=32,
    epochs=5000,
    use_wandb=True,
):
    activation_dir = Path(activation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_dir = activation_dir / f"layer_{layer_idx}"
    train_acts = np.load(layer_dir / "train.npy")
    train_labels = np.load(activation_dir / "train_labels.npy")
    test_acts = np.load(layer_dir / "test.npy")
    test_labels = np.load(activation_dir / "test_labels.npy")

    probes = []
    for seed in seeds:
        print(f"\n=== Seed {seed}, Layer {layer_idx} ===")
        probe, auroc, best_epoch = train_probe(
            train_acts, train_labels,
            test_acts, test_labels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            epochs=epochs,
            seed=seed,
            run_name=f"unencrypted_L{layer_idx}_s{seed}",
            use_wandb=use_wandb,
        )
        ckpt_path = output_dir / f"probe_L{layer_idx}_s{seed}.pt"
        torch.save(probe.state_dict(), ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")
        probes.append((probe, seed, auroc, best_epoch))

    return probes
