"""VOCASET dataset: loads precomputed .pt files with audio, prosody, and PCA coefficients.

Same interface as BIWI dataset. 12 subjects, 5023 vertices, 30 fps.
Condition C variant loads phoneme features instead of raw audio.
"""
import os

import torch
from torch.utils.data import Dataset


class VOCASETDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data = torch.load(os.path.join(data_dir, f"{split}.pt"), weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "audio": sample["audio"],
            "prosody": sample["prosody"],
            "pca_coeffs": sample["pca_coeffs"],
            "subject_id": sample["subject_id"],
            "n_frames": sample["n_frames"],
            "seq_name": sample["seq_name"],
        }


class VOCASETPhonemeDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data = torch.load(
            os.path.join(data_dir, f"{split}_condC.pt"), weights_only=False
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "phoneme_ids": sample["phoneme_ids"],
            "phoneme_pos": sample["phoneme_pos"],
            "phoneme_dur": sample["phoneme_dur"],
            "prosody": sample["prosody"],
            "pca_coeffs": sample["pca_coeffs"],
            "subject_id": sample["subject_id"],
            "n_frames": sample["n_frames"],
            "seq_name": sample["seq_name"],
        }


def vocaset_collate_fn(batch):
    max_audio_len = max(b["audio"].shape[0] for b in batch)
    max_prosody_len = max(b["prosody"].shape[0] for b in batch)
    max_target_len = max(b["n_frames"] for b in batch)

    audios = []
    audio_masks = []
    prosodies = []
    pca_targets = []
    target_masks = []
    subject_ids = []
    n_frames_list = []
    seq_names = []

    for b in batch:
        alen = b["audio"].shape[0]
        audio_pad = torch.zeros(max_audio_len)
        audio_pad[:alen] = b["audio"]
        audios.append(audio_pad)

        mask = torch.zeros(max_audio_len)
        mask[:alen] = 1.0
        audio_masks.append(mask)

        plen = b["prosody"].shape[0]
        pros_pad = torch.zeros(max_prosody_len, 2)
        pros_pad[:plen] = b["prosody"]
        prosodies.append(pros_pad)

        tlen = b["n_frames"]
        tgt_pad = torch.zeros(max_target_len, b["pca_coeffs"].shape[-1])
        tgt_pad[:tlen] = b["pca_coeffs"]
        pca_targets.append(tgt_pad)

        tmask = torch.zeros(max_target_len)
        tmask[:tlen] = 1.0
        target_masks.append(tmask)

        subject_ids.append(b["subject_id"])
        n_frames_list.append(b["n_frames"])
        seq_names.append(b["seq_name"])

    return {
        "audio": torch.stack(audios),
        "audio_mask": torch.stack(audio_masks),
        "prosody": torch.stack(prosodies),
        "pca_target": torch.stack(pca_targets),
        "target_mask": torch.stack(target_masks),
        "subject_id": torch.tensor(subject_ids, dtype=torch.long),
        "n_frames": n_frames_list,
        "seq_name": seq_names,
    }


def vocaset_phoneme_collate_fn(batch):
    max_feat_len = max(b["phoneme_ids"].shape[0] for b in batch)
    max_target_len = max(b["n_frames"] for b in batch)

    phoneme_ids_list = []
    phoneme_pos_list = []
    phoneme_dur_list = []
    prosodies = []
    feat_masks = []
    pca_targets = []
    target_masks = []
    subject_ids = []
    n_frames_list = []
    seq_names = []

    for b in batch:
        flen = b["phoneme_ids"].shape[0]
        pid_pad = torch.zeros(max_feat_len, dtype=torch.long)
        pid_pad[:flen] = b["phoneme_ids"]
        phoneme_ids_list.append(pid_pad)

        ppos_pad = torch.zeros(max_feat_len)
        ppos_pad[:flen] = b["phoneme_pos"]
        phoneme_pos_list.append(ppos_pad)

        pdur_pad = torch.zeros(max_feat_len)
        pdur_pad[:flen] = b["phoneme_dur"]
        phoneme_dur_list.append(pdur_pad)

        plen = b["prosody"].shape[0]
        pros_pad = torch.zeros(max_feat_len, 2)
        pros_pad[:plen] = b["prosody"]
        prosodies.append(pros_pad)

        fmask = torch.zeros(max_feat_len)
        fmask[:flen] = 1.0
        feat_masks.append(fmask)

        tlen = b["n_frames"]
        tgt_pad = torch.zeros(max_target_len, b["pca_coeffs"].shape[-1])
        tgt_pad[:tlen] = b["pca_coeffs"]
        pca_targets.append(tgt_pad)

        tmask = torch.zeros(max_target_len)
        tmask[:tlen] = 1.0
        target_masks.append(tmask)

        subject_ids.append(b["subject_id"])
        n_frames_list.append(b["n_frames"])
        seq_names.append(b["seq_name"])

    return {
        "phoneme_ids": torch.stack(phoneme_ids_list),
        "phoneme_pos": torch.stack(phoneme_pos_list),
        "phoneme_dur": torch.stack(phoneme_dur_list),
        "prosody": torch.stack(prosodies),
        "feat_mask": torch.stack(feat_masks),
        "pca_target": torch.stack(pca_targets),
        "target_mask": torch.stack(target_masks),
        "subject_id": torch.tensor(subject_ids, dtype=torch.long),
        "n_frames": n_frames_list,
        "seq_name": seq_names,
    }
