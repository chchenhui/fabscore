"""Prepare VOCASET Condition C data: convert MFA phone alignments to per-frame
phoneme_id, within-phone position, and phone duration features at 50Hz (prosody rate).
Adapted from scaffoldswap/data/biwi/prepare_biwi_condC.py for VOCASET paths.
"""
import argparse
import json
import os

import numpy as np
import torch

PHONEME_VOCAB = [
    "sil",
    "spn",
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
    "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY1",
    "P", "R", "S", "SH",
    "T", "TH",
    "UH1", "UH2", "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH",
]
PHONE2IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}


def alignment_to_frame_features(phones, n_frames, hop_sec=0.02):
    phoneme_ids = np.zeros(n_frames, dtype=np.int64)
    phoneme_pos = np.zeros(n_frames, dtype=np.float32)
    phoneme_dur = np.zeros(n_frames, dtype=np.float32)

    for frame_idx in range(n_frames):
        frame_center = frame_idx * hop_sec + hop_sec / 2.0
        found = False
        for phone_info in phones:
            if phone_info["start"] <= frame_center < phone_info["end"]:
                phone_label = phone_info["phone"]
                phone_start = phone_info["start"]
                phone_end = phone_info["end"]
                phone_duration = phone_end - phone_start

                pid = PHONE2IDX.get(phone_label, PHONE2IDX["spn"])
                pos = (frame_center - phone_start) / max(phone_duration, 1e-6)
                pos = min(max(pos, 0.0), 1.0)

                phoneme_ids[frame_idx] = pid
                phoneme_pos[frame_idx] = pos
                phoneme_dur[frame_idx] = phone_duration
                found = True
                break

        if not found:
            phoneme_ids[frame_idx] = PHONE2IDX["sil"]
            phoneme_pos[frame_idx] = 0.5
            phoneme_dur[frame_idx] = hop_sec

    return phoneme_ids, phoneme_pos, phoneme_dur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="scaffoldswap/data/vocaset/processed")
    parser.add_argument("--alignment_path", default="scaffoldswap/data/vocaset/mfa_alignments.json")
    args = parser.parse_args()

    with open(args.alignment_path) as f:
        alignments = json.load(f)
    print(f"Loaded {len(alignments)} alignments")
    print(f"Phoneme vocabulary size: {len(PHONEME_VOCAB)}")

    for split in ["train", "val", "test"]:
        in_path = os.path.join(args.processed_dir, f"{split}.pt")
        samples = torch.load(in_path, weights_only=False)
        print(f"\n{split}: {len(samples)} sequences")

        condC_samples = []
        missing = 0
        for sample in samples:
            seq_name = sample["seq_name"]
            if seq_name not in alignments:
                print(f"  WARNING: no alignment for {seq_name}")
                missing += 1
                continue

            phones = alignments[seq_name]
            n_prosody_frames = sample["prosody"].shape[0]

            phoneme_ids, phoneme_pos, phoneme_dur = alignment_to_frame_features(
                phones, n_prosody_frames, hop_sec=0.02
            )

            condC_samples.append({
                "phoneme_ids": torch.from_numpy(phoneme_ids).long(),
                "phoneme_pos": torch.from_numpy(phoneme_pos).float(),
                "phoneme_dur": torch.from_numpy(phoneme_dur).float(),
                "prosody": sample["prosody"],
                "pca_coeffs": sample["pca_coeffs"],
                "subject_id": sample["subject_id"],
                "subject_name": sample["subject_name"],
                "seq_name": sample["seq_name"],
                "fps": sample["fps"],
                "n_frames": sample["n_frames"],
            })

        out_path = os.path.join(args.processed_dir, f"{split}_condC.pt")
        torch.save(condC_samples, out_path)
        print(f"  Saved {len(condC_samples)} samples to {out_path} (missing: {missing})")

    vocab_path = os.path.join(args.processed_dir, "phoneme_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"vocab": PHONEME_VOCAB, "phone2idx": PHONE2IDX}, f, indent=2)
    print(f"\nPhoneme vocab saved to {vocab_path}")


if __name__ == "__main__":
    main()
