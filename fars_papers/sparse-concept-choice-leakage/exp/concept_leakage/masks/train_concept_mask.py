"""CLI entry point for training a hard-concrete mask + MLP on one concept with one seed.
Loads precomputed embeddings (D_plus label=1, D_minus label=0), trains MaskLearner,
saves best checkpoint by val loss, extracts final mask and sigma."""

import argparse
import json
import os
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concept_leakage.masks.mask_learner import MaskLearner, extract_mask

BASE_DIR = Path(__file__).resolve().parent.parent
EMB_DIR = BASE_DIR / "outputs" / "embeddings"
CKPT_DIR = BASE_DIR / "checkpoints"
DIM = 768


def load_split_embeddings(concept: str, split: str):
    dplus = np.load(EMB_DIR / concept / f"{split}_dplus.npy")
    dminus = np.load(EMB_DIR / concept / f"{split}_dminus.npy")
    x = np.concatenate([dplus, dminus], axis=0)
    y = np.concatenate([np.ones(len(dplus)), np.zeros(len(dminus))]).astype(np.float32)
    return x, y


def train_mask(
    concept: str,
    seed: int = 42,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 64,
    lambda_l0: float = 0.001,
    eval_every: int = 5,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    wandb_run = None
    try:
        import wandb
        wandb_project = os.environ.get("WANDB_PROJECT", "sparse-concept-choice-leakage")
        wandb_run = wandb.init(
            project=wandb_project,
            name=f"mask_{concept}_seed{seed}",
            config={"concept": concept, "seed": seed, "epochs": epochs,
                    "lr": lr, "batch_size": batch_size, "lambda_l0": lambda_l0},
            reinit=True,
        )
    except Exception as e:
        print(f"WandB init failed: {e}, continuing without logging")

    x_train, y_train = load_split_embeddings(concept, "train")
    x_val, y_val = load_split_embeddings(concept, "val")

    print(f"[{concept}/seed{seed}] Train: {x_train.shape}, Val: {x_val.shape}")

    x_train_t = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = MaskLearner(dim=DIM, lambda_l0=lambda_l0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    n_train = x_train_t.shape[0]
    n_val = x_val_t.shape[0]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_cls_loss = 0.0
        epoch_l0_loss = 0.0
        epoch_total = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]

            losses = model.compute_loss(xb, yb)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            epoch_cls_loss += losses["cls"].item()
            epoch_l0_loss += losses["l0"].item()
            epoch_total += losses["total"].item()
            n_batches += 1

        avg_cls = epoch_cls_loss / n_batches
        avg_l0 = epoch_l0_loss / n_batches
        avg_total = epoch_total / n_batches

        log_dict = {"train/cls_loss": avg_cls, "train/l0_loss": avg_l0,
                     "train/total_loss": avg_total, "epoch": epoch}

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_losses = model.compute_loss(x_val_t, y_val_t)
                val_total = val_losses["total"].item()
                val_cls = val_losses["cls"].item()
                val_pred = val_losses["pred"]
                val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()

            log_dict.update({"val/total_loss": val_total, "val/cls_loss": val_cls,
                              "val/accuracy": val_acc})

            if val_total < best_val_loss:
                best_val_loss = val_total
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

            print(f"  Epoch {epoch:3d}: train_loss={avg_total:.4f} (cls={avg_cls:.4f}, l0={avg_l0:.4f}) "
                  f"| val_loss={val_total:.4f}, val_acc={val_acc:.4f} {'*' if epoch == best_epoch else ''}")
        else:
            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}: train_loss={avg_total:.4f} (cls={avg_cls:.4f}, l0={avg_l0:.4f})")

        if wandb_run:
            wandb_run.log(log_dict)

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")

    mask_vec, sigma_vec = extract_mask(model, dim=DIM)
    sparsity = float((mask_vec > 0.5).mean())
    active_dims = int((mask_vec > 0.5).sum())
    print(f"  Mask sparsity: {sparsity:.4f} ({active_dims}/{DIM} dims with m>0.5)")

    out_dir = CKPT_DIR / concept / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mask.npy", mask_vec)
    np.save(out_dir / "sigma.npy", sigma_vec)
    torch.save(best_state, out_dir / "model.pt")

    meta = {
        "concept": concept, "seed": seed, "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss), "mask_sparsity": sparsity,
        "active_dims": active_dims, "dim": DIM,
        "mask_trace": float(mask_vec.sum()),
        "sigma_trace": float(sigma_vec.sum()),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved to {out_dir}")

    if wandb_run:
        wandb_run.log({"mask_sparsity": sparsity, "active_dims": active_dims,
                        "best_epoch": best_epoch, "best_val_loss": best_val_loss})
        wandb_run.finish()

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", required=True, choices=["weekdays", "months", "countries", "gender", "cities"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lambda_l0", type=float, default=0.001)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_mask(
        concept=args.concept, seed=args.seed, epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, lambda_l0=args.lambda_l0,
        eval_every=args.eval_every, device=args.device,
    )


if __name__ == "__main__":
    main()
