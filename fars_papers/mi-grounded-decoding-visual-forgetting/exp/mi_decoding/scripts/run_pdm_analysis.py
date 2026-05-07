# Collect per-step conditioned and masked logits for PDM-H analysis.
# Runs dual forward passes (conditioned + image-masked) for both vanilla and
# adaptive MI decoding, saving logit pairs at every save_interval steps.
# Supports data-parallel sharding (--num_shards / --shard_id).
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import wandb
from mi_decoding.models.load_model import (
    load_model, prepare_inputs, VLAA_THINKER_SYSTEM_PROMPT, VLAA_THINKER_IDS,
)
from mi_decoding.decoding.mi_decoding import IMAGE_PAD_TOKEN_ID, EOS_TOKEN_IDS
from mi_decoding.evaluation.pdm_h import hellinger_squared


def _prepare_masked_inputs_embeds(model, input_ids):
    embeds = model.model.get_input_embeddings()(input_ids)
    mask = (input_ids == IMAGE_PAD_TOKEN_ID).unsqueeze(-1)
    embeds = embeds.masked_fill(mask, 0.0)
    return embeds


def generate_with_logit_tracking(
    model, processor, image, question,
    max_new_tokens=512, system_prompt=None,
    method="vanilla", save_interval=10,
    lam=0.005, alpha=0.8, t0=0, max_weight=5.0,
):
    """Generate tokens with dual forward passes, saving logit pairs at sparse steps.

    For 'vanilla': tokens selected from l_c only (no MI correction).
    For 'adaptive_mi': tokens selected from corrected logits.
    Both methods maintain dual KV caches and save (l_c, l_u) at every save_interval steps.

    Returns:
        generated_text: str
        logit_pairs: dict {step_int: (logits_c_np, logits_u_np)} at save points
    """
    inputs = prepare_inputs(processor, image, question,
                            model_device=model.device, system_prompt=system_prompt)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    prompt_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    cache_position = torch.arange(prompt_len, device=input_ids.device)
    position_ids, rope_deltas = model.model.get_rope_index(
        input_ids, image_grid_thw, None, attention_mask=attention_mask,
    )

    kv_c = DynamicCache()
    model.model.rope_deltas = None
    with torch.no_grad():
        out_c = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=None,
            past_key_values=kv_c,
            use_cache=True,
            cache_position=cache_position,
        )
    logits_c = out_c.logits[:, -1:, :]
    rope_c = model.model.rope_deltas

    masked_embeds = _prepare_masked_inputs_embeds(model, input_ids)
    kv_u = DynamicCache()
    model.model.rope_deltas = None
    with torch.no_grad():
        out_u = model(
            inputs_embeds=masked_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            past_key_values=kv_u,
            use_cache=True,
            cache_position=cache_position,
        )
    logits_u = out_u.logits[:, -1:, :]
    rope_u = model.model.rope_deltas
    if rope_u is None:
        rope_u = rope_c

    generated_ids = []
    logit_pairs = {}

    for t in range(max_new_tokens):
        l_c = logits_c[:, -1, :]
        l_u = logits_u[:, -1, :]

        step_num = t + 1
        if step_num % save_interval == 0:
            logit_pairs[step_num] = (
                l_c[0].float().cpu().numpy().astype(np.float16),
                l_u[0].float().cpu().numpy().astype(np.float16),
            )

        if method == "adaptive_mi":
            p_c = F.softmax(l_c, dim=-1)
            max_p = p_c.max(dim=-1).values.item()
            if max_p < alpha:
                gamma_t = math.exp(-lam * (t + t0))
                gamma_t = max(gamma_t, 1e-8)
                weight = min((1.0 - gamma_t) / gamma_t, max_weight)
                corrected = l_c + weight * (l_c - l_u)
            else:
                corrected = l_c
            next_token = corrected.argmax(dim=-1)
        else:
            next_token = l_c.argmax(dim=-1)

        token_id = next_token.item()
        generated_ids.append(token_id)

        if token_id in EOS_TOKEN_IDS:
            break

        next_input = next_token.unsqueeze(0)
        cache_pos = torch.tensor([prompt_len + t], device=input_ids.device)
        new_attn_mask = torch.ones(
            (batch_size, prompt_len + t + 1), dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        model.model.rope_deltas = rope_c
        with torch.no_grad():
            out_c = model(
                input_ids=next_input,
                attention_mask=new_attn_mask,
                past_key_values=kv_c,
                use_cache=True,
                cache_position=cache_pos,
            )
        logits_c = out_c.logits
        kv_c = out_c.past_key_values

        model.model.rope_deltas = rope_u
        with torch.no_grad():
            out_u = model(
                input_ids=next_input,
                attention_mask=new_attn_mask,
                past_key_values=kv_u,
                use_cache=True,
                cache_position=cache_pos,
            )
        logits_u = out_u.logits
        kv_u = out_u.past_key_values

    model.model.rope_deltas = rope_c
    generated_text = processor.decode(generated_ids, skip_special_tokens=True)
    return generated_text, logit_pairs


def load_benchmark(benchmark, bench_dir):
    if benchmark == "mmstar":
        from mi_decoding.data.mmstar import load_mmstar
        return load_mmstar()
    elif benchmark == "hallusionbench":
        from mi_decoding.data.hallusionbench import load_hallusionbench
        return load_hallusionbench(bench_dir)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def select_subset(items, n=50, seed=42):
    rng = random.Random(seed)
    if len(items) <= n:
        return items, [it["id"] for it in items]
    selected = rng.sample(items, n)
    return selected, [it["id"] for it in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B")
    parser.add_argument("--benchmark", choices=["mmstar", "hallusionbench"], required=True)
    parser.add_argument("--method", choices=["vanilla", "adaptive_mi"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--subset_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--t0", type=int, default=0)
    parser.add_argument("--max_weight", type=float, default=5.0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bench_dir", default=os.path.join(PROJECT_ROOT, "HallusionBench"))
    parser.add_argument("--max_items", type=int, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            PROJECT_ROOT, "mi_decoding", "outputs", "pdm_analysis",
            args.method, args.benchmark,
        )
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_project = os.environ.get("WANDB_PROJECT", "mi-grounded-decoding-visual-forgetting")
    run_name = f"pdm_{args.method}_{args.benchmark}_shard{args.shard_id}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        config=vars(args),
    )

    system_prompt = None
    if args.model_id in VLAA_THINKER_IDS:
        system_prompt = VLAA_THINKER_SYSTEM_PROMPT
        print(f"Using VLAA-Thinker system prompt")

    print(f"Loading model: {args.model_id}")
    model, processor = load_model(args.model_id)
    print(f"Model loaded on {model.device}")

    print(f"Loading benchmark: {args.benchmark}")
    all_items = load_benchmark(args.benchmark, args.bench_dir)
    print(f"Total benchmark items: {len(all_items)}")

    items, subset_ids = select_subset(all_items, n=args.subset_size, seed=args.seed)
    print(f"Selected {len(items)} items for PDM analysis")

    subset_ids_path = os.path.join(args.output_dir, "subset_ids.json")
    if args.shard_id == 0:
        with open(subset_ids_path, "w") as f:
            json.dump(subset_ids, f, indent=2)

    if args.max_items is not None:
        items = items[:args.max_items]
        print(f"Truncated to {len(items)} items (--max_items)")

    shard_size = (len(items) + args.num_shards - 1) // args.num_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, len(items))
    items = items[start:end]
    print(f"Shard {args.shard_id}/{args.num_shards}: items [{start}, {end})")

    all_logit_data = {}
    t0_total = time.time()
    pdm_h_step10_vals = []

    for i, item in enumerate(items):
        t_item = time.time()
        item_id = str(item["id"])
        print(f"[{i+1}/{len(items)}] Processing item {item_id} with {args.method}...")

        try:
            gen_text, logit_pairs = generate_with_logit_tracking(
                model, processor, item["image"], item["question"],
                max_new_tokens=args.max_new_tokens,
                system_prompt=system_prompt,
                method=args.method,
                save_interval=args.save_interval,
                lam=args.lam, alpha=args.alpha,
                t0=args.t0, max_weight=args.max_weight,
            )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            gen_text = ""
            logit_pairs = {}

        item_time = time.time() - t_item
        num_steps = len(gen_text.split()) if gen_text else 0
        num_saved_steps = len(logit_pairs)

        pdm_h_at_10 = None
        if 10 in logit_pairs:
            pdm_h_at_10 = hellinger_squared(logit_pairs[10][0], logit_pairs[10][1])
            pdm_h_step10_vals.append(pdm_h_at_10)

        all_logit_data[item_id] = {
            "logit_pairs": logit_pairs,
            "num_generated_tokens": len(gen_text) if gen_text else 0,
        }

        log_data = {
            "item_idx": i,
            "elapsed_time": item_time,
            "num_saved_steps": num_saved_steps,
            "step": i,
        }
        if pdm_h_at_10 is not None:
            log_data["pdm_h_step10"] = pdm_h_at_10
        if pdm_h_step10_vals:
            log_data["running_mean_pdm_h_step10"] = np.mean(pdm_h_step10_vals)
        wandb.log(log_data)

        total_elapsed = time.time() - t0_total
        rate = (i + 1) / total_elapsed if total_elapsed > 0 else 0
        pdm_str = f" pdm_h@10={pdm_h_at_10:.4f}" if pdm_h_at_10 is not None else ""
        print(f"  saved_steps={num_saved_steps} time={item_time:.1f}s rate={rate:.2f}it/s{pdm_str}")

    shard_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.npz")
    save_dict = {}
    for item_id, data in all_logit_data.items():
        for step, (lc, lu) in data["logit_pairs"].items():
            save_dict[f"{item_id}_step{step}_lc"] = lc
            save_dict[f"{item_id}_step{step}_lu"] = lu
    save_dict["_item_ids"] = np.array(list(all_logit_data.keys()), dtype=object)

    np.savez_compressed(shard_path, **save_dict)
    print(f"Saved logits to {shard_path}")

    total_time = time.time() - t0_total
    wandb.summary.update({
        "total_items": len(all_logit_data),
        "total_time_seconds": total_time,
        "mean_pdm_h_step10": np.mean(pdm_h_step10_vals) if pdm_h_step10_vals else None,
    })
    wandb.finish()
    print(f"Done. {len(all_logit_data)} items in {total_time:.1f}s")


if __name__ == "__main__":
    main()
