# Merge LoRA adapters with base model using PEFT directly (no LlamaFactory).
# Avoids omegaconf/antlr4 version conflict by not importing LlamaFactory.
# Usage: python merge_lora.py [--dataset DATASET] [--seed SEED] [--dry-run]
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path("/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp")
OUTPUTS_DIR = PROJECT_ROOT / "tlr_proxy_sft" / "outputs" / "target"
BASE_MODEL = "Qwen/Qwen2.5-7B"

DATASETS = [
    "AM-Thinking-v1-Distilled-math", "DeepMath-309K", "Maths-College",
    "OpenR1-Math", "QwQ-LongCoT-130K-math", "R1-Distill-SFT-math",
    "hkust-nlp__dart-math-hard", "mathplus", "numinamath-cot",
    "numinamath1_5", "openmathinstruct-2", "Magpie-Reasoning-V2-250K-CoT-QwQ-math",
]
SEEDS = [42, 123, 456]


def merge_one(dataset: str, seed: int, dry_run: bool = False) -> bool:
    adapter_dir = OUTPUTS_DIR / dataset / f"seed_{seed}"
    merged_dir = OUTPUTS_DIR / dataset / f"seed_{seed}" / "merged"

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
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pairs = []
    for d in DATASETS:
        for s in SEEDS:
            if args.dataset and d != args.dataset:
                continue
            if args.seed is not None and s != args.seed:
                continue
            pairs.append((d, s))

    print(f"Processing {len(pairs)} adapter(s) ...")
    success, fail = 0, 0
    for d, s in pairs:
        ok = merge_one(d, s, args.dry_run)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\nSummary: {success} merged, {fail} failed (out of {len(pairs)} total)")


if __name__ == "__main__":
    main()
