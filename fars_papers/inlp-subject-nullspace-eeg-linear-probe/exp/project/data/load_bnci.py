# Load BNCI2014001 (BCI Competition IV-2a) motor imagery dataset.
# Downloads .mat files from TU Graz mirror, preprocesses (bandpass 0.3-50Hz,
# resample to 200Hz, epoch 2-6s MI window), applies Euclidean Alignment
# per subject, and caches aligned EEG arrays to disk.

import os
import numpy as np
import scipy.io
import scipy.signal
import scipy.linalg
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "project" / "outputs" / "aligned_eeg"
RAW_DIR = PROJECT_ROOT / "project" / "outputs" / "raw_mat"

BASE_URL = "https://lampx.tugraz.at/~bci/database/001-2014/"
SUBJECTS = list(range(1, 10))
SESSIONS = ["T", "E"]
ORIG_SFREQ = 250
TARGET_SFREQ = 200
N_EEG_CHANNELS = 22
MI_START_SEC = 2
MI_END_SEC = 6
MI_SAMPLES = (MI_END_SEC - MI_START_SEC) * TARGET_SFREQ  # 800
PATCH_SIZE = 200
N_PATCHES = MI_SAMPLES // PATCH_SIZE  # 4

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}


def download_mat_files(subjects=SUBJECTS, force=False):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for subj in subjects:
        for sess in SESSIONS:
            fname = f"A{subj:02d}{sess}.mat"
            fpath = RAW_DIR / fname
            if fpath.exists() and not force:
                continue
            url = BASE_URL + fname
            print(f"Downloading {url} ...")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            fpath.write_bytes(r.content)
            print(f"  Saved to {fpath} ({len(r.content)} bytes)")


def load_single_subject_raw(subject_id):
    trials_all = []
    labels_all = []
    for sess in SESSIONS:
        fname = f"A{subject_id:02d}{sess}.mat"
        fpath = RAW_DIR / fname
        mat = scipy.io.loadmat(str(fpath), squeeze_me=False, struct_as_record=True)
        data_struct = mat["data"]
        for run_idx in range(3, data_struct.shape[1]):
            run = data_struct[0, run_idx]
            raw = run["X"][0, 0][:, :N_EEG_CHANNELS].T  # (22, time)
            trial_pos = run["trial"][0, 0].flatten()
            labels = run["y"][0, 0].flatten()

            nyq = 0.5 * ORIG_SFREQ
            b, a = scipy.signal.butter(5, [0.3 / nyq, 50.0 / nyq], btype="band")
            raw = scipy.signal.filtfilt(b, a, raw, axis=-1)

            for pos, label_orig in zip(trial_pos, labels):
                label_orig = int(label_orig)
                if label_orig not in LABEL_MAP:
                    continue
                label = LABEL_MAP[label_orig]
                start = int(pos) - 1 + int(MI_START_SEC * ORIG_SFREQ)
                end = int(pos) - 1 + int(MI_END_SEC * ORIG_SFREQ)
                if end > raw.shape[1]:
                    continue
                segment = raw[:, start:end]  # (22, 1000) at 250Hz
                segment = segment - segment.mean(axis=1, keepdims=True)
                resampled = scipy.signal.resample(segment, MI_SAMPLES, axis=1)  # (22, 800)
                trials_all.append(resampled)
                labels_all.append(label)

    X = np.stack(trials_all, axis=0)  # (n_trials, 22, 800)
    y = np.array(labels_all)
    return X, y


def euclidean_alignment(X):
    n_trials, n_ch, n_time = X.shape
    R = np.zeros((n_ch, n_ch))
    for i in range(n_trials):
        R += X[i] @ X[i].T
    R /= n_trials
    R_inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(R))
    R_inv_sqrt = np.real(R_inv_sqrt)
    X_aligned = np.array([R_inv_sqrt @ X[i] for i in range(n_trials)])
    return X_aligned


def process_and_cache(subjects=SUBJECTS, apply_ea=True, force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_mat_files(subjects)

    for subj in subjects:
        suffix = "ea" if apply_ea else "raw"
        out_path = DATA_DIR / f"subject_{subj}_{suffix}.npz"
        if out_path.exists() and not force:
            print(f"Subject {subj} already cached at {out_path}")
            continue

        print(f"Processing subject {subj} ...")
        X, y = load_single_subject_raw(subj)
        print(f"  Loaded: X={X.shape}, y={y.shape}")

        if apply_ea:
            X = euclidean_alignment(X)
            print(f"  After EA: X={X.shape}")

        np.savez(out_path, X=X, y=y)
        print(f"  Saved to {out_path}")

        label_counts = {i: int(np.sum(y == i)) for i in range(4)}
        print(f"  Labels: {label_counts}, total={len(y)}")


def load_cached(subject_id, apply_ea=True):
    suffix = "ea" if apply_ea else "raw"
    fpath = DATA_DIR / f"subject_{subject_id}_{suffix}.npz"
    data = np.load(fpath)
    return data["X"], data["y"]


if __name__ == "__main__":
    process_and_cache(apply_ea=True, force=False)
    print("\n=== Verification ===")
    for subj in SUBJECTS:
        X, y = load_cached(subj, apply_ea=True)
        vals, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(vals.tolist(), counts.tolist()))
        print(f"Subject {subj}: X={X.shape}, y={y.shape}, labels={label_dist}")
