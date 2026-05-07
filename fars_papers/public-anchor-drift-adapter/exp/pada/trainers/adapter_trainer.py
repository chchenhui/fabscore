"""Adapter training loop with MSE loss, early stopping, and checkpoint saving.

Trains a ResidualMLPAdapter to map f_new embeddings to f_old embeddings using
paired anchor embeddings. Supports reproducible seeding, 80/20 train/val split,
cosine LR scheduling, and saves best checkpoint by validation loss.
"""

import pathlib
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from pada.adapters.residual_mlp import ResidualMLPAdapter

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_adapter(
    source_embeds: np.ndarray,
    target_embeds: np.ndarray,
    save_dir: str | pathlib.Path,
    embed_dim: int = 768,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 5,
    seed: int = 0,
    device: str | None = None,
    use_cosine_schedule: bool = False,
) -> dict:
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(source_embeds)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    src_train = torch.tensor(source_embeds[train_idx], dtype=torch.float32)
    tgt_train = torch.tensor(target_embeds[train_idx], dtype=torch.float32)
    src_val = torch.tensor(source_embeds[val_idx], dtype=torch.float32)
    tgt_val = torch.tensor(target_embeds[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(src_train, tgt_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(src_val, tgt_val), batch_size=batch_size, shuffle=False
    )

    model = ResidualMLPAdapter(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.01) if use_cosine_schedule else None
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for src_batch, tgt_batch in train_loader:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
            pred = model(src_batch)
            loss = criterion(pred, tgt_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
                pred = model(src_batch)
                loss = criterion(pred, tgt_batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        current_lr = scheduler.get_last_lr()[0] if scheduler else lr
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(current_lr)
        logger.info(f"Epoch {epoch+1}/{max_epochs} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f} | lr={current_lr:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_dir / "best_adapter.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_dir / "best_adapter.pt", map_location=device, weights_only=True))
    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "history": history,
    }
