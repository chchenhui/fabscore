# Extract frozen CBraMod embeddings from EA-aligned EEG epochs.
# Loads pretrained CBraMod encoder (proj_out replaced by Identity),
# passes EEG patches through the frozen model, and saves two pooling
# variants (avgpool -> (n,200), flatten -> (n,17600)) per subject.

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "CBraMod"))

from models.cbramod import CBraMod

ALIGNED_DIR = PROJECT_ROOT / "project" / "outputs" / "aligned_eeg"
EMB_DIR = PROJECT_ROOT / "project" / "outputs" / "embeddings"
WEIGHTS_PATH = PROJECT_ROOT / "third_party" / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
SUBJECTS = list(range(1, 10))
BATCH_SIZE = 64
N_PATCHES = 4
PATCH_SIZE = 200


def load_model(device):
    model = CBraMod().to(device)
    model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=device))
    model.proj_out = nn.Identity()
    model.eval()
    return model


def extract_embeddings(apply_ea=True, force=False):
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(device)

    for subj in SUBJECTS:
        suffix = "ea" if apply_ea else "raw"
        avgpool_path = EMB_DIR / f"cbramod_{suffix}_{subj}_avgpool.npy"
        flatten_path = EMB_DIR / f"cbramod_{suffix}_{subj}_flatten.npy"
        labels_path = EMB_DIR / f"labels_{suffix}_{subj}.npy"

        if avgpool_path.exists() and flatten_path.exists() and not force:
            print(f"Subject {subj} embeddings already cached")
            continue

        data = np.load(ALIGNED_DIR / f"subject_{subj}_{suffix}.npz")
        X = data["X"]  # (n_trials, 22, 800)
        y = data["y"]

        X_patches = X.reshape(X.shape[0], 22, N_PATCHES, PATCH_SIZE)  # (n, 22, 4, 200)
        X_tensor = torch.from_numpy(X_patches).float()

        all_avgpool = []
        all_flatten = []

        n = X_tensor.shape[0]
        with torch.no_grad():
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch = X_tensor[start:end].to(device)
                out = model(batch)  # (batch, 22, 4, 200)
                avg = out.mean(dim=(1, 2))  # (batch, 200)
                flat = out.reshape(out.shape[0], -1)  # (batch, 17600)
                all_avgpool.append(avg.cpu().numpy())
                all_flatten.append(flat.cpu().numpy())

        emb_avgpool = np.concatenate(all_avgpool, axis=0)
        emb_flatten = np.concatenate(all_flatten, axis=0)

        np.save(avgpool_path, emb_avgpool)
        np.save(flatten_path, emb_flatten)
        np.save(labels_path, y)

        print(f"Subject {subj}: avgpool={emb_avgpool.shape}, flatten={emb_flatten.shape}, "
              f"nan_avg={np.any(np.isnan(emb_avgpool))}, nan_flat={np.any(np.isnan(emb_flatten))}")


if __name__ == "__main__":
    extract_embeddings(apply_ea=True, force=False)
    print("\n=== Verification ===")
    for subj in SUBJECTS:
        avg = np.load(EMB_DIR / f"cbramod_ea_{subj}_avgpool.npy")
        flat = np.load(EMB_DIR / f"cbramod_ea_{subj}_flatten.npy")
        lab = np.load(EMB_DIR / f"labels_ea_{subj}.npy")
        print(f"Subject {subj}: avgpool={avg.shape}, flatten={flat.shape}, labels={lab.shape}, "
              f"nan={np.any(np.isnan(avg)) or np.any(np.isnan(flat))}, "
              f"inf={np.any(np.isinf(avg)) or np.any(np.isinf(flat))}")
