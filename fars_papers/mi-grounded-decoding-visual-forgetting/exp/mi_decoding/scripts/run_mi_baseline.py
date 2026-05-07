# Run adaptive MI decoding on MMStar or HallusionBench with data-parallel sharding.
# Supports --num_shards N --shard_id K. Logs per-item metrics to WandB.
import argparse
import json
import os
import sys
import time

import torch
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import wandb
from mi_decoding.models.load_model import load_model, VLAA_THINKER_SYSTEM_PROMPT, VLAA_THINKER_IDS
from mi_decoding.decoding.mi_decoding import generate_mi_decoding
from mi_decoding.evaluation.extract_answer import extract_mc_answer, extract_yesno_answer


def load_benchmark(benchmark, bench_dir):
    if benchmark == "mmstar":
        from mi_decoding.data.mmstar import load_mmstar
        return load_mmstar()
    elif benchmark == "hallusionbench":
        from mi_decoding.data.hallusionbench import load_hallusionbench
        return load_hallusionbench(bench_dir)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B")
    parser.add_argument("--benchmark", choices=["mmstar", "hallusionbench"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--lam", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--t0", type=int, default=0)
    parser.add_argument("--max_weight", type=float, default=5.0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bench_dir", default=os.path.join(PROJECT_ROOT, "HallusionBench"))
    parser.add_argument("--subset_file", default=None)
    parser.add_argument("--t0_prompt_length", action="store_true", default=False)
    parser.add_argument("--fixed_gamma", type=float, default=None)
    args = parser.parse_args()

    model_name = args.model_id.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = os.path.join(
            PROJECT_ROOT, "mi_decoding", "outputs",
            f"adaptive_mi_{model_name}", args.benchmark,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_project = os.environ.get("WANDB_PROJECT", "mi-grounded-decoding-visual-forgetting")
    method_name = "fixed_gamma_mi_decoding" if args.fixed_gamma is not None else "adaptive_mi_decoding"
    run_name = f"{method_name}_{args.benchmark}_t{args.max_new_tokens}_shard{args.shard_id}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "method": method_name,
            "model_id": args.model_id,
            "benchmark": args.benchmark,
            "max_new_tokens": args.max_new_tokens,
            "lam": args.lam,
            "alpha": args.alpha,
            "t0": "prompt_length" if args.t0_prompt_length else args.t0,
            "max_weight": args.max_weight,
            "num_shards": args.num_shards,
            "shard_id": args.shard_id,
            "fixed_gamma": args.fixed_gamma,
        },
    )

    system_prompt = None
    if args.model_id in VLAA_THINKER_IDS:
        system_prompt = VLAA_THINKER_SYSTEM_PROMPT
        print(f"Using VLAA-Thinker system prompt (thinking mode)")

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)
    print(f"Model loaded on {model.device}")

    print(f"Loading benchmark: {args.benchmark}")
    items = load_benchmark(args.benchmark, args.bench_dir)
    total = len(items)
    print(f"Total items: {total}")

    if args.subset_file is not None:
        with open(args.subset_file) as f:
            subset_ids = set(json.load(f))
        items = [it for it in items if it["id"] in subset_ids]
        print(f"Filtered to {len(items)} items from subset file {args.subset_file}")

    if args.max_items is not None:
        items = items[:args.max_items]
        print(f"Truncated to {len(items)} items")

    shard_size = (len(items) + args.num_shards - 1) // args.num_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, len(items))
    items = items[start:end]
    print(f"Shard {args.shard_id}/{args.num_shards}: items [{start}, {end})")

    output_file = os.path.join(args.output_dir, f"shard_{args.shard_id}.jsonl")
    results = []
    t0 = time.time()
    correct_count = 0

    for i, item in enumerate(items):
        t_item = time.time()
        try:
            t0_val = "prompt_length" if args.t0_prompt_length else args.t0
            generated_text = generate_mi_decoding(
                model, processor, item["image"], item["question"],
                max_new_tokens=args.max_new_tokens,
                system_prompt=system_prompt,
                lam=args.lam, alpha=args.alpha,
                t0=t0_val, max_weight=args.max_weight,
                fixed_gamma=args.fixed_gamma,
            )
        except Exception as e:
            print(f"[{i}/{len(items)}] Error on item {item['id']}: {e}")
            import traceback
            traceback.print_exc()
            generated_text = ""

        item_time = time.time() - t_item

        if args.benchmark == "mmstar":
            extracted = extract_mc_answer(generated_text)
            gt = item["answer"]
        else:
            extracted = extract_yesno_answer(generated_text)
            gt = item["gt_answer"]

        correct = (extracted == gt)
        if correct:
            correct_count += 1

        record = {
            "id": item["id"],
            "question": item["question"],
            "gt_answer": gt,
            "generated_text": generated_text,
            "extracted_answer": extracted,
            "time_seconds": round(item_time, 2),
        }
        if args.benchmark == "hallusionbench":
            record["category"] = item.get("category", "")
            record["figure_id"] = item.get("figure_id", "")
            record["set_id"] = item.get("set_id", "")
            record["visual_input"] = item.get("visual_input", "")
            record["index"] = item.get("index", i)

        results.append(record)

        running_acc = correct_count / (i + 1)
        wandb.log({
            "item_correct": int(correct),
            "time_seconds": item_time,
            "running_accuracy": running_acc,
            "step": i,
        })

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"[{i+1}/{len(items)}] id={item['id']} pred={extracted} gt={gt} correct={correct} time={item_time:.1f}s ({rate:.2f} items/s) acc={running_acc:.3f}")

        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            with open(output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    final_acc = correct_count / len(results) if results else 0
    print(f"\nShard {args.shard_id} done. {len(results)} items in {elapsed:.1f}s ({len(results)/elapsed:.2f} items/s)")
    print(f"Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

    wandb.summary.update({
        "final_accuracy": final_acc,
        "total_items": len(results),
        "total_correct": correct_count,
        "total_time_seconds": elapsed,
        "items_per_second": len(results) / elapsed if elapsed > 0 else 0,
    })

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {output_file}")

    wandb.finish()


if __name__ == "__main__":
    main()
