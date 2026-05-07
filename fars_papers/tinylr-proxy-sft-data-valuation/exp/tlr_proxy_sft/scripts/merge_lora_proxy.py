# Merge LoRA adapters for proxy model (Qwen2.5-1.5B) with base model using PEFT.
# Usage: python merge_lora_proxy.py --regime proxy_std [--dataset DATASET] [--seed SEED] [--dry-run]
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path("/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp")
BASE_MODEL = "Qwen/Qwen2.5-1.5B"

DATASETS = [
    "AM-Thinking-v1-Distilled-math", "DeepMath-309K", "Maths-College",
    "OpenR1-Math", "QwQ-LongCoT-130K-math", "R1-Distill-SFT-math",
    "hkust-nlp__dart-math-hard", "mathplus", "numinamath-cot",
    "numinamath1_5", "openmathinstruct-2", "Magpie-Reasoning-V2-250K-CoT-QwQ-math",
]
SEEDS = [42, 123, 456]


def merge_one(outputs_dir: Path, dataset: str, seed: int, dry_run: bool = False) -> bool:
    adapter_dir = outputs_dir / dataset / f"seed_{seed}"
    merged_dir = outputs_dir / dataset / f"seed_{seed}" / "merged"

    if not (adapter_dir / "adapter_config.json").exists():
        print(f"SKIP {dataset}/seed_{seed}: no adapter_config.json")
        return False

    has_weights = any(merged_dir.glob("*.safetensors")) if merged_dir.exists() else False
    if has_weights and (merged_dir / "config.json").exists():
        print(f"SKIP {dataset}/seed_{seed}: already merged")
        return True

    if dry_run:
        print(f"WOULD MERGE {dataset}/seed_{seed}")
        return True

    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    print(f"MERGING {dataset}/seed_{seed} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()

        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(str(merged_dir))
        print(f"DONE {dataset}/seed_{seed}")
        return True
    except Exception as e:
        print(f"FAILED {dataset}/seed_{seed}: {e}")
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, required=True, choices=["proxy_std", "proxy_tiny", "proxy_mid", "proxy_tiny_v2"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outputs_dir = PROJECT_ROOT / "tlr_proxy_sft" / "outputs" / args.regime

    pairs = []
    for d in DATASETS:
        for s in SEEDS:
            if args.dataset and d != args.dataset:
                continue
            if args.seed is not None and s != args.seed:
                continue
            pairs.append((d, s))

    print(f"Processing {len(pairs)} adapter(s) from {outputs_dir} ...")
    success, fail = 0, 0
    for d, s in pairs:
        ok = merge_one(outputs_dir, d, s, args.dry_run)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\nSummary: {success} merged, {fail} failed (out of {len(pairs)} total)")


if __name__ == "__main__":
    main()
