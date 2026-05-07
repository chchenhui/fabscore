"""Evaluation script for ScaffoldSwap experiments.

Loads a trained checkpoint, runs inference on the test set, reconstructs
full vertex displacements from PCA coefficients, and computes:
  - LVE (Lip Vertex Error): max L2 per lip vertex per frame, averaged
  - MVE (Mean Vertex Error): mean L2 across all vertices per frame, averaged
  - UFVE (Upper Face Vertex Error): mean L2 across upper-face vertices
  - FDD (Face Dynamics Deviation): std of frame-to-frame vertex displacement
"""
import argparse
import json
import os

import numpy as np
import torch
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


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

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

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, cfg, ckpt


def inverse_pca(pca_coeffs, pca_model):
    """Reconstruct vertex displacements from PCA coefficients.

    Args:
        pca_coeffs: (T, 512) PCA coefficients
        pca_model: dict with 'components_to_data' (512, V*3) and 'original_data_mean' (V*3,)

    Returns:
        vertices: (T, V, 3) reconstructed vertex displacements
    """
    components = pca_model["components_to_data"]  # (512, V*3)
    mean = pca_model["original_data_mean"]  # (V*3,)
    n_verts = components.shape[1] // 3

    reconstructed = pca_coeffs @ components + mean.unsqueeze(0)  # (T, V*3)
    return reconstructed.reshape(-1, n_verts, 3)  # (T, V, 3)


def compute_lve(pred_verts, gt_verts, lip_indices):
    """Lip Vertex Error: for each frame, max squared L2 among lip vertices, then averaged.

    LVE = (1/N) * sum_i max_v ||x_v - x_hat_v||^2  (standard definition)
    """
    pred_lip = pred_verts[:, lip_indices]  # (T, n_lip, 3)
    gt_lip = gt_verts[:, lip_indices]
    sq_l2_per_vert = ((pred_lip - gt_lip) ** 2).sum(dim=-1)  # (T, n_lip)
    max_per_frame = sq_l2_per_vert.max(dim=-1).values  # (T,)
    return max_per_frame.mean().item()


def compute_mve(pred_verts, gt_verts):
    """Mean Vertex Error: mean L2 across all vertices per frame, then averaged."""
    l2_per_vert = torch.norm(pred_verts - gt_verts, dim=-1)  # (T, V)
    return l2_per_vert.mean().item()


def compute_ufve(pred_verts, gt_verts, upper_face_indices):
    """Upper Face Vertex Error: mean L2 across upper-face vertices."""
    pred_uf = pred_verts[:, upper_face_indices]
    gt_uf = gt_verts[:, upper_face_indices]
    l2_per_vert = torch.norm(pred_uf - gt_uf, dim=-1)
    return l2_per_vert.mean().item()


def compute_fdd(pred_verts, gt_verts, upper_face_indices):
    """Face Dynamics Deviation: difference in temporal dynamics (std of velocity)."""
    pred_uf = pred_verts[:, upper_face_indices]
    gt_uf = gt_verts[:, upper_face_indices]

    pred_vel = pred_uf[1:] - pred_uf[:-1]
    gt_vel = gt_uf[1:] - gt_uf[:-1]

    pred_vel_std = pred_vel.std(dim=0)
    gt_vel_std = gt_vel.std(dim=0)

    return torch.norm(pred_vel_std - gt_vel_std).item() / len(upper_face_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, ckpt = load_model_from_checkpoint(args.checkpoint, device)

    pca_model = torch.load(os.path.join(args.data_dir, "pca_model.pt"), weights_only=False)
    for k in pca_model:
        pca_model[k] = pca_model[k].to(device)

    regions = torch.load(os.path.join(args.data_dir, "region_indices.pt"), weights_only=False)
    lip_indices = regions["lip"]
    upper_face_indices = regions["upper_face"]

    condition = cfg["condition"]
    DS, collate_fn, PhonDS, phon_collate_fn = get_dataset_classes(cfg.get("dataset", "biwi"))

    if condition == "C":
        test_ds = PhonDS(args.data_dir, split="test")
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size,
            shuffle=False, collate_fn=phon_collate_fn,
            num_workers=2,
        )
    else:
        test_ds = DS(args.data_dir, split="test")
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=2,
        )

    all_lve, all_mve, all_ufve, all_fdd = [], [], [], []

    with torch.no_grad():
        for batch in test_loader:
            prosody = batch["prosody"].to(device)
            pca_target = batch["pca_target"].to(device)
            target_mask = batch["target_mask"].to(device)
            speaker_id = batch["subject_id"].to(device)
            n_frames_list = batch["n_frames"]
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

            for i in range(pca_pred.shape[0]):
                T = n_frames_list[i]
                pred_pca = pca_pred[i, :T]
                gt_pca = pca_target[i, :T]

                pred_verts = inverse_pca(pred_pca, pca_model)
                gt_verts = inverse_pca(gt_pca, pca_model)

                lve = compute_lve(pred_verts, gt_verts, lip_indices)
                mve = compute_mve(pred_verts, gt_verts)
                ufve = compute_ufve(pred_verts, gt_verts, upper_face_indices)
                fdd = compute_fdd(pred_verts, gt_verts, upper_face_indices) if T > 1 else 0.0

                all_lve.append(lve)
                all_mve.append(mve)
                all_ufve.append(ufve)
                all_fdd.append(fdd)

    results = {
        "condition": cfg["condition"],
        "dataset": cfg["dataset"],
        "seed": ckpt["seed"],
        "best_epoch": ckpt["epoch"],
        "best_val_loss": ckpt["val_loss"],
        "LVE": float(np.mean(all_lve)),
        "MVE": float(np.mean(all_mve)),
        "UFVE": float(np.mean(all_ufve)),
        "FDD": float(np.mean(all_fdd)),
        "LVE_std": float(np.std(all_lve)),
        "n_test_sequences": len(all_lve),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults for Condition {cfg['condition']}, seed {ckpt['seed']}:")
    print(f"  LVE:  {results['LVE']:.6f}")
    print(f"  MVE:  {results['MVE']:.6f}")
    print(f"  UFVE: {results['UFVE']:.6f}")
    print(f"  FDD:  {results['FDD']:.6f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
