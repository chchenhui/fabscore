"""Generate synthetic VOCASET-equivalent data for ScaffoldSwap ablation on VOCASET.

Same approach as BIWI synthetic data: use real PCA model from UniTalker D1_vocaset
release with LibriSpeech audio. VOCASET has 5023 vertices x 3 = 15069 dims,
12 subjects, 473 sequences (377 train / 48 val / 48 test), at 30 fps
(downsampled from 60 fps following UniTalker convention).
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d


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

TARGET_FPS = 30


def load_pca_stats(pca_path):
    pca = np.load(pca_path)
    return {
        "components_to_data": pca["components_to_data"],  # (512, 15069)
        "original_data_mean": pca["original_data_mean"],  # (15069,)
        "data_components_mean": pca["data_components_mean"][:512],  # (512,)
        "data_components_std": pca["data_components_std"][:512],  # (512,)
    }


def load_librispeech_flacs(librispeech_dir):
    flacs = sorted(glob.glob(os.path.join(librispeech_dir, "**", "*.flac"), recursive=True))
    return flacs


def generate_pca_sequence(n_frames, pca_stats, rng):
    mean = pca_stats["data_components_mean"]
    std = pca_stats["data_components_std"]
    noise = rng.standard_normal((n_frames, 512)).astype(np.float32)
    smoothed = uniform_filter1d(noise, size=7, axis=0)
    coeffs = mean[None, :] + smoothed * std[None, :]
    return coeffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocaset_dir", required=True,
                        help="D1_vocaset dir with pca.npz, train/val/test.json")
    parser.add_argument("--librispeech_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    pca_stats = load_pca_stats(os.path.join(args.vocaset_dir, "pca.npz"))
    flacs = load_librispeech_flacs(args.librispeech_dir)
    random.shuffle(flacs)
    print(f"Found {len(flacs)} LibriSpeech .flac files")

    os.makedirs(args.output_dir, exist_ok=True)

    flac_idx = 0
    total_entries = 0

    for split in ["train", "val", "test"]:
        split_json = os.path.join(args.vocaset_dir, f"{split}.json")
        with open(split_json) as f:
            meta = json.load(f)

        entries = meta["data"]
        print(f"\n{split}: {len(entries)} entries")

        for entry in entries:
            annot_rel = entry["annot_path"]
            audio_rel = entry["audio_path"]
            orig_fps = entry["fps"]  # 60

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

            n_frames = int(duration_sec * TARGET_FPS)
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
        src = os.path.join(args.vocaset_dir, f"{split}.json")
        dst = os.path.join(args.output_dir, f"{split}.json")
        with open(src) as f:
            data = json.load(f)
        for entry in data["data"]:
            entry["fps"] = TARGET_FPS
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)

    id_templ = np.load(os.path.join(args.vocaset_dir, "id_template.npy"))
    np.save(os.path.join(args.output_dir, "id_template.npy"), id_templ)

    print(f"\nTotal: {total_entries} entries generated in {args.output_dir}")


if __name__ == "__main__":
    main()
