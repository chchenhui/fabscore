"""Prepare VOCASET dataset: load PCA coefficients, audio, precompute prosody, cache as .pt.

VOCASET has 12 subjects, 5023 vertices, 30 fps (downsampled from 60).
Uses same prosody extraction as BIWI (F0 + energy at 50 Hz).
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

from scaffoldswap.data.preprocessing import extract_prosody_features


VOCASET_SUBJECTS = [
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170731_00024_TA",
    "FaceTalk_170809_00138_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170811_03275_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170908_03277_TA",
    "FaceTalk_170912_03278_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170915_00223_TA",
]

SUBJECT_TO_IDX = {s: i for i, s in enumerate(VOCASET_SUBJECTS)}


def extract_subject_name(annot_path):
    fn = os.path.basename(annot_path).replace(".npy", "")
    parts = fn.rsplit("_sentence", 1)
    return parts[0]


def prepare_split(data_dir, split_json_path, split_name, output_dir, sr=16000):
    with open(split_json_path) as f:
        meta = json.load(f)

    entries = meta["data"]
    samples = []

    for i, entry in enumerate(entries):
        annot_path = os.path.join(data_dir, entry["annot_path"])
        audio_path = os.path.join(data_dir, entry["audio_path"])

        pca_coeffs = np.load(annot_path)  # (T, 512)
        subject_name = extract_subject_name(entry["annot_path"])
        subject_id = SUBJECT_TO_IDX[subject_name]

        audio, file_sr = sf.read(audio_path)
        if file_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        audio = audio.astype(np.float32)

        prosody = extract_prosody_features(audio, sr=sr)

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


def _load_region(path):
    with open(path) as f:
        content = f.read().strip()
    return [int(x.strip()) for x in content.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="scaffoldswap/data/vocaset/synthetic")
    parser.add_argument("--output_dir", default="scaffoldswap/data/vocaset/processed")
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

    lip_indices = _load_region("scaffoldswap/data/vocaset/regions/lve.txt")
    upper_face_indices = _load_region("scaffoldswap/data/vocaset/regions/fdd.txt")
    torch.save({
        "lip": torch.tensor(lip_indices, dtype=torch.long),
        "upper_face": torch.tensor(upper_face_indices, dtype=torch.long),
    }, os.path.join(args.output_dir, "region_indices.pt"))
    print(f"Region indices saved: lip={len(lip_indices)}, upper_face={len(upper_face_indices)}")

    for split in ["train", "val", "test"]:
        json_path = os.path.join(args.data_dir, f"{split}.json")
        prepare_split(args.data_dir, json_path, split, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
