# Run visual replay decoding on MMStar or HallusionBench.
# Two-pass: vanilla generation -> re-insert downsampled image at punctuation boundaries.
# Supports data-parallel sharding: --num_shards N --shard_id K.
import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from mi_decoding.models.load_model import load_model, VLAA_THINKER_SYSTEM_PROMPT, VLAA_THINKER_IDS
from mi_decoding.decoding.visual_replay import generate_visual_replay
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
    parser.add_argument("--num_insertions", type=int, default=4)
    parser.add_argument("--downsample_scale", type=float, default=0.5)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bench_dir", default=os.path.join(PROJECT_ROOT, "HallusionBench"))
    args = parser.parse_args()

    model_name = args.model_id.split("/")[-1]
    if args.output_dir is None:
        args.output_dir = os.path.join(
            PROJECT_ROOT, "mi_decoding", "outputs",
            f"visual_replay_{model_name}", args.benchmark,
        )
    os.makedirs(args.output_dir, exist_ok=True)

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

    for i, item in enumerate(items):
        t_item = time.time()
        try:
            generated_text = generate_visual_replay(
                model, processor, item["image"], item["question"],
                max_new_tokens=args.max_new_tokens,
                system_prompt=system_prompt,
                num_insertions=args.num_insertions,
                downsample_scale=args.downsample_scale,
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

        correct = (extracted == gt)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"[{i+1}/{len(items)}] id={item['id']} pred={extracted} gt={gt} correct={correct} time={item_time:.1f}s ({rate:.2f} items/s)")

        if (i + 1) % 10 == 0 or (i + 1) == len(items):
            with open(output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    print(f"\nShard {args.shard_id} done. {len(results)} items in {elapsed:.1f}s ({len(results)/elapsed:.2f} items/s)")

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
