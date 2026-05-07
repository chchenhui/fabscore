"""Run MFA forced alignment on synthetic BIWI audio to get phoneme-level TextGrids.

Recreates the LibriSpeech->BIWI mapping from generate_synthetic_data.py (seed=0),
retrieves transcripts, creates an MFA corpus directory, runs alignment, and
parses the resulting TextGrids into a per-sequence JSON file.
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess

import tgt


PROJ_ROOT = "/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/exomni-scaffold-swap-ablation/exp"


def get_flac_to_biwi_mapping(unitalker_dir, librispeech_dir):
    flacs = sorted(glob.glob(os.path.join(librispeech_dir, "**", "*.flac"), recursive=True))
    random.seed(0)
    random.shuffle(flacs)

    flac_idx = 0
    mapping = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(unitalker_dir, f"{split}.json")) as f:
            meta = json.load(f)
        for entry in meta["data"]:
            audio_rel = entry["audio_path"]
            seq_name = os.path.splitext(os.path.basename(audio_rel))[0]
            mapping[seq_name] = {
                "flac": flacs[flac_idx % len(flacs)],
                "split": split,
            }
            flac_idx += 1
    return mapping


def get_transcript(flac_path):
    flac_dir = os.path.dirname(flac_path)
    flac_name = os.path.splitext(os.path.basename(flac_path))[0]
    trans_files = [f for f in os.listdir(flac_dir) if f.endswith(".trans.txt")]
    with open(os.path.join(flac_dir, trans_files[0])) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if parts[0] == flac_name:
                return parts[1] if len(parts) > 1 else ""
    return None


def prepare_mfa_corpus(mapping, synthetic_dir, corpus_dir):
    os.makedirs(corpus_dir, exist_ok=True)
    for seq_name, info in mapping.items():
        transcript = get_transcript(info["flac"])
        if transcript is None:
            print(f"WARNING: no transcript for {seq_name}")
            continue
        wav_src = os.path.join(synthetic_dir, info["split"], f"{seq_name}.wav")
        wav_dst = os.path.join(corpus_dir, f"{seq_name}.wav")
        txt_dst = os.path.join(corpus_dir, f"{seq_name}.txt")

        if not os.path.exists(wav_dst):
            shutil.copy2(wav_src, wav_dst)
        with open(txt_dst, "w") as f:
            f.write(transcript)

    n_wav = len([f for f in os.listdir(corpus_dir) if f.endswith(".wav")])
    n_txt = len([f for f in os.listdir(corpus_dir) if f.endswith(".txt")])
    print(f"MFA corpus: {n_wav} wav, {n_txt} txt files in {corpus_dir}")


def run_mfa_align(corpus_dir, output_dir):
    mamba_root = os.path.join(PROJ_ROOT, "micromamba_envs")
    micromamba = os.path.join(PROJ_ROOT, "bin", "micromamba")

    cmd = [
        micromamba, "run", "-n", "mfa",
        "mfa", "align",
        corpus_dir,
        "english_us_arpa",
        "english_us_arpa",
        output_dir,
        "--clean",
        "--single_speaker",
        "--num_jobs", "4",
    ]
    env = os.environ.copy()
    env["MAMBA_ROOT_PREFIX"] = mamba_root

    print(f"Running MFA: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(f"MFA stdout: {result.stdout[-2000:] if result.stdout else '(empty)'}")
    if result.returncode != 0:
        print(f"MFA stderr: {result.stderr[-3000:] if result.stderr else '(empty)'}")
        raise RuntimeError(f"MFA alignment failed with return code {result.returncode}")
    print("MFA alignment complete")


def parse_textgrids(output_dir, alignment_save_path):
    alignments = {}
    tg_files = sorted(glob.glob(os.path.join(output_dir, "*.TextGrid")))
    print(f"Found {len(tg_files)} TextGrid files")

    for tg_path in tg_files:
        seq_name = os.path.splitext(os.path.basename(tg_path))[0]
        tg = tgt.io.read_textgrid(tg_path)
        phones_tier = tg.get_tier_by_name("phones")

        phones = []
        for interval in phones_tier:
            phones.append({
                "phone": interval.text if interval.text else "sil",
                "start": float(interval.start_time),
                "end": float(interval.end_time),
            })
        alignments[seq_name] = phones

    with open(alignment_save_path, "w") as f:
        json.dump(alignments, f, indent=2)
    print(f"Saved {len(alignments)} alignments to {alignment_save_path}")
    return alignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unitalker_dir",
                        default="scaffoldswap/data/biwi/raw/unitalker_data_release_V1/D0_BIWI")
    parser.add_argument("--librispeech_dir",
                        default="scaffoldswap/data/biwi/raw/librispeech/LibriSpeech/test-clean")
    parser.add_argument("--synthetic_dir",
                        default="scaffoldswap/data/biwi/synthetic")
    parser.add_argument("--corpus_dir",
                        default="scaffoldswap/data/biwi/mfa_corpus")
    parser.add_argument("--mfa_output_dir",
                        default="scaffoldswap/data/biwi/mfa_output")
    parser.add_argument("--alignment_output",
                        default="scaffoldswap/data/biwi/mfa_alignments.json")
    args = parser.parse_args()

    mapping = get_flac_to_biwi_mapping(args.unitalker_dir, args.librispeech_dir)
    print(f"Mapped {len(mapping)} sequences")

    prepare_mfa_corpus(mapping, args.synthetic_dir, args.corpus_dir)
    run_mfa_align(args.corpus_dir, args.mfa_output_dir)
    parse_textgrids(args.mfa_output_dir, args.alignment_output)


if __name__ == "__main__":
    main()
