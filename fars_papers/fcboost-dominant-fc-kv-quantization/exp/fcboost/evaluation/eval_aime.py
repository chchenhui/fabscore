# Reusable evaluation wrapper for AIME benchmarks with Kitty KV cache quantization.
# Accepts a quantization method config, runs eval via Kitty's runner, logs to WandB.
# Usage: python -m fcboost.evaluation.eval_aime --method kivi_kv2star --task aime24 \
#        --num_repeats 3 --max_new_tokens 32768 --results_dir ./eval_results
#
# Seed protocol: Kitty's run_evaluation_repeats uses base_seed + repeat_idx.
#   base_random_seed=0, base_numpy/torch/fewshot_seed=1234.
#   For 3 repeats: random={0,1,2}, numpy/torch/fewshot={1234,1235,1236}.

import argparse
import json
import os
import sys

import torch
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MASK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "masks", "qwen3_8b_ca_masks.pt"
)

DEFAULT_MASK_V2_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "masks_v2", "qwen3_8b_ca_masks.pt"
)

METHOD_CONFIGS = {
    "kivi_kv2star": {
        "sink_length": 32,
        "buffer_length": 128,
        "group_size": 128,
        "kbits": 2,
        "vbits": 2,
        "promote_ratio": 0.0,
        "promote_bit": 4,
        "channel_selection": 0,
    },
    "kivi_kv2": {
        "sink_length": 0,
        "buffer_length": 128,
        "group_size": 128,
        "kbits": 2,
        "vbits": 2,
        "promote_ratio": 0.0,
        "promote_bit": 4,
        "channel_selection": 0,
    },
    "kitty": {
        "sink_length": 32,
        "buffer_length": 128,
        "group_size": 128,
        "kbits": 2,
        "vbits": 2,
        "promote_ratio": 0.125,
        "promote_bit": 4,
        "channel_selection": 1,
    },
    "fcboost": {
        "sink_length": 32,
        "buffer_length": 128,
        "group_size": 128,
        "kbits": 2,
        "vbits": 2,
        "promote_ratio": 0.125,
        "promote_bit": 4,
        "channel_selection": -1,
        "mask_path": DEFAULT_MASK_PATH,
        "boost_values": False,
    },
    "fcboost_v2": {
        "sink_length": 32,
        "buffer_length": 128,
        "group_size": 128,
        "kbits": 2,
        "vbits": 2,
        "promote_ratio": 0.125,
        "promote_bit": 4,
        "channel_selection": -1,
        "mask_path": DEFAULT_MASK_V2_PATH,
        "boost_values": True,
    },
    "fp16": None,
}


def build_kv_cache(config, method=None, mask_path=None):
    if config is None:
        return None

    from kitty_sim import KittyKVCacheConfig, KittyKVCache

    is_fcboost = method in ("fcboost", "fcboost_v2")
    effective_channel_selection = config["channel_selection"]
    if is_fcboost:
        effective_channel_selection = 0

    cache_config = KittyKVCacheConfig(
        sink_length=config["sink_length"],
        buffer_length=config["buffer_length"],
        group_size=config["group_size"],
        kbits=config["kbits"],
        vbits=config["vbits"],
        promote_ratio=config["promote_ratio"],
        promote_bit=config["promote_bit"],
        channel_selection=effective_channel_selection,
        VCache_BitDecoding=False,
        PostQuant=True,
    )

    if is_fcboost:
        from fcboost.quantization.fcboost_cache import FCBoostKVCache
        effective_mask_path = mask_path or config.get("mask_path", DEFAULT_MASK_PATH)
        boost_values = config.get("boost_values", False)
        return FCBoostKVCache(cache_config=cache_config, mask_path=effective_mask_path,
                              boost_values=boost_values)

    return KittyKVCache(cache_config=cache_config)


def build_file_name(method, config):
    if config is None:
        return f"{method}_fp16"
    if method in ("fcboost", "fcboost_v2"):
        suffix = "_vboost" if config.get("boost_values", False) else ""
        return "{}_g{}_b{}_s{}_k{}_v{}_pb{}_pr{}{}".format(
            method, config["group_size"], config["buffer_length"], config["sink_length"],
            config["kbits"], config["vbits"],
            config["promote_bit"], config["promote_ratio"], suffix,
        )
    return "kitty_g{}_b{}_s{}_sel{}_k{}_v{}_pb{}_pr{}".format(
        config["group_size"], config["buffer_length"], config["sink_length"],
        config["channel_selection"], config["kbits"], config["vbits"],
        config["promote_bit"], config["promote_ratio"],
    )


def log_to_wandb(method, task, results_dir, model_name):
    try:
        import wandb
    except ImportError:
        print("[WandB] wandb not installed, skipping logging")
        return

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    wandb_project = os.environ.get("WANDB_PROJECT", "fcboost-kv-quantization")
    wandb_mode = os.environ.get("WANDB_MODE", "offline")

    if not wandb_key:
        print("[WandB] No WANDB_API_KEY found, skipping logging")
        return

    summary_pattern = f"{results_dir}/{model_name}/{task}"
    summary_files = []
    for root, dirs, files in os.walk(summary_pattern):
        for f in files:
            if f.endswith("_summary.json"):
                summary_files.append(os.path.join(root, f))

    if not summary_files:
        print(f"[WandB] No summary files found in {summary_pattern}")
        return

    for sf in summary_files:
        with open(sf) as fh:
            summary = json.load(fh)

        run_name = f"{method}_{task}"
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            mode=wandb_mode,
            config={
                "method": method,
                "task": task,
                "model": model_name,
                "num_repeats": summary.get("num_repeats", 0),
                "model_configs": summary.get("model_configs", {}),
            },
            reinit=True,
        )

        stats = summary.get("statistics", {})
        for metric_name, metric_data in stats.items():
            if isinstance(metric_data, dict):
                wandb.log({
                    f"{task}/{metric_name}_mean": metric_data.get("mean", 0),
                    f"{task}/{metric_name}_std": metric_data.get("std", 0),
                    f"{task}/{metric_name}_min": metric_data.get("min", 0),
                    f"{task}/{metric_name}_max": metric_data.get("max", 0),
                })
                for i, v in enumerate(metric_data.get("values", [])):
                    wandb.log({f"{task}/{metric_name}_repeat_{i}": v})

        wandb.finish()
        print(f"[WandB] Logged {run_name} to project {wandb_project} (mode={wandb_mode})")


def main():
    parser = argparse.ArgumentParser(description="AIME evaluation with KV cache quantization")
    parser.add_argument("--method", type=str, required=True,
                        choices=list(METHOD_CONFIGS.keys()),
                        help="Quantization method name")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model path")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (aime24, aime25)")
    parser.add_argument("--num_repeats", type=int, default=3,
                        help="Number of evaluation repeats (seeds)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=32768,
                        help="Maximum new tokens to generate")
    parser.add_argument("--results_dir", type=str, default="./eval_results",
                        help="Directory to save results")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (limit=8 samples, 1 repeat)")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path to FCBoost mask file (overrides default)")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM
    from kitty_sim.eval.runner import eval_model_downstream, release_model_memory

    config = METHOD_CONFIGS[args.method]
    kv_cache = build_kv_cache(config, method=args.method, mask_path=args.mask_path)
    file_name = build_file_name(args.method, config)

    print("=" * 80)
    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Config: {config}")
    print(f"Num repeats: {args.num_repeats}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Batch size: {args.batch_size}")
    print(f"Results dir: {args.results_dir}")
    print(f"Debug: {args.debug}")
    print("=" * 80)

    model_name = args.model.split("/")[-1]

    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto"
    )
    print("Model loaded.")

    eval_model_downstream(
        model=model,
        task=args.task,
        ModelName=model_name,
        fileName=file_name,
        DEBUG=args.debug,
        kv_cache=kv_cache,
        num_repeats=args.num_repeats,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        results_dir=args.results_dir,
    )

    release_model_memory(model)

    print("\nLogging results to WandB...")
    log_to_wandb(args.method, args.task, args.results_dir, model_name)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
