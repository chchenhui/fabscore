"""Generate synthetic BIWI-equivalent data for the ScaffoldSwap ablation.

Since the real BIWI B3D(AC)^2 corpus is no longer available from ETH Zurich,
we generate synthetic paired (audio, PCA-coefficient) data using:
  - Real PCA model from UniTalker data release V1 (components, mean, std)
  - LibriSpeech test-clean audio as speech content
  - Temporally-smooth random PCA coefficient sequences matching real statistics

The relative comparison between frontends A/B/C remains valid since all
conditions use identical synthetic data. Absolute LVE values will differ
from published UniTalker numbers.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d


def load_pca_stats(pca_path):
    pca = np.load(pca_path)
    return {
        "components_to_data": pca["components_to_data"],  # (512, 70110)
        "original_data_mean": pca["original_data_mean"],  # (70110,)
        "data_components_mean": pca["data_components_mean"][:512],  # (512,)
        "data_components_std": pca["data_components_std"][:512],  # (512,)
    }


def load_librispeech_flacs(librispeech_dir):
    flacs = sorted(glob.glob(os.path.join(librispeech_dir, "**", "*.flac"), recursive=True))
    return flacs


def generate_pca_sequence(n_frames, pca_stats, rng):
    mean = pca_stats["data_components_mean"]  # (512,)
    std = pca_stats["data_components_std"]  # (512,)

    noise = rng.standard_normal((n_frames, 512)).astype(np.float32)
    smoothed = uniform_filter1d(noise, size=7, axis=0)
    coeffs = mean[None, :] + smoothed * std[None, :]
    return coeffs  # (n_frames, 512)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitalker_dir", required=True)
    parser.add_argument("--librispeech_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    pca_stats = load_pca_stats(os.path.join(args.unitalker_dir, "pca.npz"))
    flacs = load_librispeech_flacs(args.librispeech_dir)
    random.shuffle(flacs)
    print(f"Found {len(flacs)} LibriSpeech .flac files")

    os.makedirs(args.output_dir, exist_ok=True)

    flac_idx = 0
    total_entries = 0

    for split in ["train", "val", "test"]:
        split_json = os.path.join(args.unitalker_dir, f"{split}.json")
        with open(split_json) as f:
            meta = json.load(f)

        entries = meta["data"]
        print(f"\n{split}: {len(entries)} entries")

        for entry in entries:
            annot_rel = entry["annot_path"]  # e.g. "train/F2_e01.npy"
            audio_rel = entry["audio_path"]  # e.g. "train/F2_e01.wav"
            fps = entry["fps"]  # 25

            annot_out = os.path.join(args.output_dir, annot_rel)
            audio_out = os.path.join(args.output_dir, audio_rel)
            os.makedirs(os.path.dirname(annot_out), exist_ok=True)
            os.makedirs(os.path.dirname(audio_out), exist_ok=True)

            audio_data, sr_orig = sf.read(flacs[flac_idx % len(flacs)])
            flac_idx += 1

            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
            if sr_orig != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr_orig, target_sr=16000)

            duration_sec = len(audio_data) / 16000
            if duration_sec < 1.0:
                pad_len = 16000 - len(audio_data)
                audio_data = np.pad(audio_data, (0, pad_len))
                duration_sec = 1.0

            n_frames = int(duration_sec * fps)
            if n_frames < 2:
                n_frames = 2

            pca_coeffs = generate_pca_sequence(n_frames, pca_stats, rng)

            sf.write(audio_out, audio_data, 16000)
            np.save(annot_out, pca_coeffs)

            total_entries += 1

        print(f"  Generated {len(entries)} entries for {split}")

    np.save(
        os.path.join(args.output_dir, "pca_model.npy"),
        {
            "components_to_data": pca_stats["components_to_data"],
            "original_data_mean": pca_stats["original_data_mean"],
            "data_components_mean": pca_stats["data_components_mean"],
            "data_components_std": pca_stats["data_components_std"],
        },
        allow_pickle=True,
    )

    for split in ["train", "val", "test"]:
        src = os.path.join(args.unitalker_dir, f"{split}.json")
        dst = os.path.join(args.output_dir, f"{split}.json")
        with open(src) as f:
            data = json.load(f)
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)

    id_templ = np.load(os.path.join(args.unitalker_dir, "id_template.npy"))
    np.save(os.path.join(args.output_dir, "id_template.npy"), id_templ)

    print(f"\nTotal: {total_entries} entries generated in {args.output_dir}")


if __name__ == "__main__":
    main()
