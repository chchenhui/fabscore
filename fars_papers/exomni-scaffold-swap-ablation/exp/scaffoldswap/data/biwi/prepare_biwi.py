"""Prepare BIWI dataset: load PCA coefficients, audio, precompute prosody, cache as .pt.

Precomputes prosody features (F0 + energy) at preparation time to avoid
numba/librosa import issues in DataLoader worker processes on shared filesystems.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

from scaffoldswap.data.preprocessing import extract_prosody_features


SUBJECT_TO_IDX = {"F2": 0, "F3": 1, "F4": 2, "M3": 3, "M4": 4, "M5": 5}


def prepare_split(data_dir, split_json_path, split_name, output_dir, sr=16000):
    with open(split_json_path) as f:
        meta = json.load(f)

    entries = meta["data"]
    samples = []

    for i, entry in enumerate(entries):
        annot_path = os.path.join(data_dir, entry["annot_path"])
        audio_path = os.path.join(data_dir, entry["audio_path"])

        pca_coeffs = np.load(annot_path)  # (T, 512)
        subject_name = os.path.basename(entry["annot_path"]).split("_")[0]
        subject_id = SUBJECT_TO_IDX[subject_name]

        audio, file_sr = sf.read(audio_path)
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        audio = audio.astype(np.float32)

        prosody = extract_prosody_features(audio, sr=sr)  # (T_50hz, 2)

        samples.append({
            "pca_coeffs": torch.from_numpy(pca_coeffs).float(),
            "audio": torch.from_numpy(audio).float(),
            "prosody": torch.from_numpy(prosody).float(),
            "subject_id": subject_id,
            "subject_name": subject_name,
            "seq_name": os.path.splitext(os.path.basename(entry["annot_path"]))[0],
            "fps": entry["fps"],
            "n_frames": pca_coeffs.shape[0],
        })

        if (i + 1) % 50 == 0:
            print(f"    {split_name}: processed {i+1}/{len(entries)}")

    out_path = os.path.join(output_dir, f"{split_name}.pt")
    torch.save(samples, out_path)
    print(f"  {split_name}: {len(samples)} sequences saved to {out_path}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="scaffoldswap/data/biwi/synthetic")
    parser.add_argument("--output_dir", default="scaffoldswap/data/biwi/processed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pca_model = np.load(
        os.path.join(args.data_dir, "pca_model.npy"), allow_pickle=True
    ).item()
    pca_save = {
        "components_to_data": torch.from_numpy(pca_model["components_to_data"]).float(),
        "original_data_mean": torch.from_numpy(pca_model["original_data_mean"]).float(),
    }
    torch.save(pca_save, os.path.join(args.output_dir, "pca_model.pt"))
    print("PCA model saved")

    id_template = np.load(os.path.join(args.data_dir, "id_template.npy"))
    torch.save(torch.from_numpy(id_template).float(), os.path.join(args.output_dir, "id_template.pt"))
    print(f"Identity templates saved: {id_template.shape}")

    lip_indices = _load_region("scaffoldswap/data/biwi/regions/lve.txt")
    upper_face_indices = _load_region("scaffoldswap/data/biwi/regions/fdd.txt")
    torch.save({
        "lip": torch.tensor(lip_indices, dtype=torch.long),
        "upper_face": torch.tensor(upper_face_indices, dtype=torch.long),
    }, os.path.join(args.output_dir, "region_indices.pt"))
    print(f"Region indices saved: lip={len(lip_indices)}, upper_face={len(upper_face_indices)}")

    for split in ["train", "val", "test"]:
        json_path = os.path.join(args.data_dir, f"{split}.json")
        prepare_split(args.data_dir, json_path, split, args.output_dir)

    print("Done.")


def _load_region(path):
    with open(path) as f:
        content = f.read().strip()
    return [int(x.strip()) for x in content.split(",") if x.strip()]


if __name__ == "__main__":
    main()
