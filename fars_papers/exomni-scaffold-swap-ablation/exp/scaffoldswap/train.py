"""Training entry point for ScaffoldSwap ablation experiments.

Config-driven: loads YAML config specifying condition, dataset, seed, hyperparams.
Supports multi-seed runs. Logs to WandB (offline) and saves best checkpoint by val loss.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from scaffoldswap.frontends.ssl_frontend import SSLFrontend
from scaffoldswap.frontends.unit_frontend import UnitFrontend
from scaffoldswap.frontends.phoneme_frontend import PhonemeFrontend
from scaffoldswap.frontends.hubert_continuous_frontend import HuBERTContinuousFrontend
from scaffoldswap.model.scaffold_model import ScaffoldModel


def get_dataset_classes(dataset_name):
    if dataset_name == "vocaset":
        from scaffoldswap.data.vocaset.dataset import (
            VOCASETDataset as DS, vocaset_collate_fn as collate,
            VOCASETPhonemeDataset as PhonDS, vocaset_phoneme_collate_fn as phon_collate,
        )
    else:
        from scaffoldswap.data.biwi.dataset import (
            BIWIDataset as DS, biwi_collate_fn as collate,
            BIWIPhonemeDataset as PhonDS, biwi_phoneme_collate_fn as phon_collate,
        )
    return DS, collate, PhonDS, phon_collate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, device):
    model_cfg = cfg["model"]
    if cfg["condition"] == "A":
        frontend = SSLFrontend(
            ssl_model_name=model_cfg["ssl_model"],
            ssl_dim=model_cfg["ssl_dim"],
            prosody_dim=model_cfg["prosody_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            cache_dir=cfg.get("cache_dir"),
        )
    elif cfg["condition"] == "B":
        frontend = UnitFrontend(
            hubert_model_name=model_cfg["hubert_model"],
            kmeans_path=model_cfg["kmeans_path"],
            n_clusters=model_cfg["n_clusters"],
            embed_dim=model_cfg["embed_dim"],
            prosody_dim=model_cfg["prosody_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            cache_dir=cfg.get("cache_dir"),
        )
    elif cfg["condition"] == "C":
        frontend = PhonemeFrontend(
            n_phonemes=model_cfg["n_phonemes"],
            embed_dim=model_cfg["embed_dim"],
            prosody_dim=model_cfg["prosody_dim"],
            timing_dim=model_cfg.get("timing_dim", 2),
            hidden_dim=model_cfg["hidden_dim"],
        )
    elif cfg["condition"] == "hubert_continuous":
        frontend = HuBERTContinuousFrontend(
            ssl_model_name=model_cfg["ssl_model"],
            ssl_dim=model_cfg["ssl_dim"],
            prosody_dim=model_cfg["prosody_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            cache_dir=cfg.get("cache_dir"),
        )
    else:
        raise ValueError(f"Unknown condition: {cfg['condition']}")

    model = ScaffoldModel(
        frontend=frontend,
        source_rate=50,
        target_fps=cfg["data"]["target_fps"],
        hidden_dim=model_cfg["hidden_dim"],
        output_dim=model_cfg["pca_output_dim"],
        n_decoder_blocks=model_cfg["decoder_blocks"],
        decoder_kernel_size=model_cfg["decoder_kernel_size"],
        num_speakers=model_cfg["num_speakers"],
        speaker_embed_dim=model_cfg.get("speaker_embed_dim", 64),
    )
    return model.to(device)


def compute_loss(pca_pred, pca_target, target_mask, vertex_scaling):
    """MSE loss on PCA coefficients with optional vertex scaling."""
    mask = target_mask.unsqueeze(-1)  # (B, T, 1)
    diff = (pca_pred - pca_target) * mask * vertex_scaling
    loss = (diff ** 2).sum() / mask.sum() / pca_pred.shape[-1]
    return loss


def evaluate(model, dataloader, device, vertex_scaling, condition="A"):
    model.eval()
    total_loss = 0.0
    total_frames = 0
    with torch.no_grad():
        for batch in dataloader:
            prosody = batch["prosody"].to(device)
            pca_target = batch["pca_target"].to(device)
            target_mask = batch["target_mask"].to(device)
            speaker_id = batch["subject_id"].to(device)
            max_target_frames = pca_target.shape[1]

            if condition == "C":
                pca_pred = model(
                    None, prosody, speaker_id,
                    n_target_frames=max_target_frames,
                    phoneme_ids=batch["phoneme_ids"].to(device),
                    phoneme_pos=batch["phoneme_pos"].to(device),
                    phoneme_dur=batch["phoneme_dur"].to(device),
                )
            else:
                audio = batch["audio"].to(device)
                audio_mask = batch["audio_mask"].to(device)
                pca_pred = model(audio, prosody, speaker_id,
                                 n_target_frames=max_target_frames,
                                 audio_mask=audio_mask)

            T = min(pca_pred.shape[1], pca_target.shape[1])
            pca_pred = pca_pred[:, :T]
            pca_target_t = pca_target[:, :T]
            mask_t = target_mask[:, :T]

            loss = compute_loss(pca_pred, pca_target_t, mask_t, vertex_scaling)
            n_frames = mask_t.sum().item()
            total_loss += loss.item() * n_frames
            total_frames += n_frames

    return total_loss / max(total_frames, 1)


def train_one_seed(cfg, seed, device):
    set_seed(seed)
    output_dir = os.path.join(cfg["output_dir"], f"seed{seed}")
    os.makedirs(output_dir, exist_ok=True)

    wandb_run = None
    try:
        import wandb
        from dotenv import load_dotenv
        load_dotenv()
        wandb_project = os.environ.get("WANDB_PROJECT", "scaffoldswap")
        wandb_mode = os.environ.get("WANDB_MODE", "offline")
        wandb_name = os.environ.get(
            "WANDB_RUN_NAME",
            f"cond{cfg['condition']}_{cfg['dataset']}_seed{seed}",
        )
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=cfg,
            mode=wandb_mode,
            dir=output_dir,
        )
    except Exception as e:
        print(f"WandB init failed: {e}, continuing without logging")

    condition = cfg["condition"]
    train_cfg = cfg["training"]

    DS, collate_fn, PhonDS, phon_collate_fn = get_dataset_classes(cfg.get("dataset", "biwi"))

    if condition == "C":
        train_ds = PhonDS(cfg["data"]["data_dir"], split="train")
        val_ds = PhonDS(cfg["data"]["data_dir"], split="val")
        collate_fn = phon_collate_fn
    else:
        train_ds = DS(cfg["data"]["data_dir"], split="train")
        val_ds = DS(cfg["data"]["data_dir"], split="val")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"],
        shuffle=True, collate_fn=collate_fn,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True,
    )

    model = build_model(cfg, device)
    opt_name = train_cfg.get("optimizer", "adam").lower()
    opt_params = [p for p in model.parameters() if p.requires_grad]
    weight_decay = train_cfg.get("weight_decay", 0.0)
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(opt_params, lr=train_cfg["lr"], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(opt_params, lr=train_cfg["lr"])

    scheduler = None
    sched_type = train_cfg.get("scheduler", "none")
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    if sched_type == "cosine":
        cosine_epochs = train_cfg["epochs"] - warmup_epochs
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(cosine_epochs, 1),
            eta_min=train_cfg.get("scheduler_eta_min", 1e-6),
        )
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs],
            )
        else:
            scheduler = cosine_sched

    vertex_scaling = cfg["data"].get("vertex_scaling", 1.0)
    best_val_loss = float("inf")
    eval_every = train_cfg.get("eval_every_epochs", 10)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Seed {seed}] Trainable: {trainable:,} / Total: {total:,}")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        if hasattr(model.frontend, 'ssl_model'):
            model.frontend.ssl_model.eval()
        elif hasattr(model.frontend, 'hubert_model'):
            model.frontend.hubert_model.eval()
        epoch_loss = 0.0
        epoch_frames = 0
        t0 = time.time()

        for batch in train_loader:
            prosody = batch["prosody"].to(device)
            pca_target = batch["pca_target"].to(device)
            target_mask = batch["target_mask"].to(device)
            speaker_id = batch["subject_id"].to(device)
            max_target_frames = pca_target.shape[1]

            shuffle_flag = cfg.get("shuffle_temporal", False)
            if condition == "C":
                pca_pred = model(
                    None, prosody, speaker_id,
                    n_target_frames=max_target_frames,
                    phoneme_ids=batch["phoneme_ids"].to(device),
                    phoneme_pos=batch["phoneme_pos"].to(device),
                    phoneme_dur=batch["phoneme_dur"].to(device),
                    shuffle_temporal=shuffle_flag,
                )
            else:
                audio = batch["audio"].to(device)
                audio_mask = batch["audio_mask"].to(device)
                pca_pred = model(audio, prosody, speaker_id,
                                 n_target_frames=max_target_frames,
                                 audio_mask=audio_mask,
                                 shuffle_temporal=shuffle_flag)

            T = min(pca_pred.shape[1], pca_target.shape[1])
            pca_pred = pca_pred[:, :T]
            pca_target_t = pca_target[:, :T]
            mask_t = target_mask[:, :T]

            loss = compute_loss(pca_pred, pca_target_t, mask_t, vertex_scaling)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_frames = mask_t.sum().item()
            epoch_loss += loss.item() * n_frames
            epoch_frames += n_frames

        avg_train_loss = epoch_loss / max(epoch_frames, 1)
        elapsed = time.time() - t0

        log_msg = f"[Seed {seed}] Epoch {epoch}/{train_cfg['epochs']}: train_loss={avg_train_loss:.6f}, time={elapsed:.1f}s"

        if epoch % eval_every == 0 or epoch == train_cfg["epochs"]:
            val_loss = evaluate(model, val_loader, device, vertex_scaling, condition=condition)
            log_msg += f", val_loss={val_loss:.6f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(output_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                    "seed": seed,
                }, ckpt_path)
                log_msg += " *best*"

            if wandb_run:
                wandb_run.log({"val_loss": val_loss, "epoch": epoch})

        print(log_msg)
        if wandb_run:
            log_dict = {"train_loss": avg_train_loss, "epoch": epoch}
            if scheduler is not None:
                log_dict["lr"] = scheduler.get_last_lr()[0]
            wandb_run.log(log_dict)

        if scheduler is not None:
            scheduler.step()

        if np.isnan(avg_train_loss) or avg_train_loss > 1e6:
            print(f"Training collapsed at epoch {epoch}, stopping.")
            break

    if wandb_run:
        wandb_run.finish()

    print(f"[Seed {seed}] Best val_loss: {best_val_loss:.6f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override single seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shuffle_temporal", action="store_true",
                        help="Randomly permute temporal order of features (negative control)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.shuffle_temporal:
        cfg["shuffle_temporal"] = True
        print("[SHUFFLE] Temporal shuffle enabled -- features will be randomly permuted per utterance per batch")

    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [42])
    device = torch.device(args.device)

    for seed in seeds:
        print(f"\n{'='*60}\nTraining with seed {seed}\n{'='*60}")
        train_one_seed(cfg, seed, device)


if __name__ == "__main__":
    main()
