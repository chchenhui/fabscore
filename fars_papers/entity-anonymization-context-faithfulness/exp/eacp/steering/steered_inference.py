"""Steered inference on ConFiQA-MC using ContextFocus activation steering.
Applies steering vector at selected layer during generation with multiplier m=2.
Uses HF transformers model.generate() with forward hooks (not vLLM)."""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, PROJ_DIR)

from eacp.data.confiqa_loader import load_confiqa
from eacp.evaluation.confiqa_scorer import compute_metrics
from eacp.prompts.condition_a import build_chat_messages_a

VECTORS_DIR = os.path.join(SCRIPT_DIR, "vectors")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 8
MULTIPLIER = 2.0


class SteeringHook:
    """Adds steering vector to residual stream during inference.
    prefill_only=False (default): steer only generated tokens (KV-cache steps).
    prefill_only=True: steer ALL input positions during prefill, skip generation.
      This bakes the steering into the KV cache so generation sees steered context."""

    def __init__(self, steering_vector, multiplier, input_lengths, prefill_only=False):
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.input_lengths = input_lengths
        self.prefill_only = prefill_only

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if self.multiplier == 0:
            return output

        device = hidden.device
        sv = self.steering_vector.to(device=device, dtype=hidden.dtype)
        seq_len = hidden.size(1)

        if self.prefill_only:
            if seq_len > 1:
                hidden[:, :, :] = hidden[:, :, :] + self.multiplier * sv
        else:
            if seq_len == 1:
                hidden[:, :, :] = hidden[:, :, :] + self.multiplier * sv
            else:
                for b in range(hidden.size(0)):
                    if b < len(self.input_lengths):
                        start_pos = self.input_lengths[b]
                        if start_pos < seq_len:
                            hidden[b, start_pos:] = hidden[b, start_pos:] + self.multiplier * sv

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="MC")
    parser.add_argument("--subset_indices", default=os.path.join(PROJ_DIR, "eacp", "data", "confiqa_mc_1500_subset_indices.json"))
    parser.add_argument("--layer_config", default=os.path.join(VECTORS_DIR, "selected_layer.json"))
    parser.add_argument("--vectors_path", default=os.path.join(VECTORS_DIR, "llama31_8b_contextfocus.pt"))
    parser.add_argument("--output_dir", default=os.path.join(PROJ_DIR, "eacp", "outputs"))
    parser.add_argument("--multiplier", type=float, default=MULTIPLIER)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--prefill_only", action="store_true", help="Only steer during prefill, not KV-cache generation steps")
    args = parser.parse_args()

    print("Loading data...")
    data = load_confiqa(args.split)

    with open(args.subset_indices) as f:
        subset_indices = json.load(f)
    data = [data[i] for i in subset_indices]
    print(f"Using {len(data)} subset examples")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {len(data)} examples")

    steering_vectors = torch.load(args.vectors_path, weights_only=True)

    if args.layer is not None:
        selected_layer = args.layer
        print(f"Using manually specified layer {selected_layer}")
    else:
        with open(args.layer_config) as f:
            layer_config = json.load(f)
        selected_layer = layer_config["selected_layer"]
        print(f"Using selected layer {selected_layer} (Ps improvement: {layer_config['improvement']*100:.2f}pp)")

    sv = steering_vectors[selected_layer]
    print(f"Steering vector shape: {sv.shape}, L2 norm: {sv.float().norm().item():.4f}")

    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    print("Building prompts...")
    conversations = [build_chat_messages_a(inst) for inst in data]
    texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations]

    mode_str = "prefill_only" if args.prefill_only else "all_tokens"
    print(f"\nRunning steered inference (layer={selected_layer}, m={args.multiplier}, mode={mode_str})...")
    all_outputs = []
    t0 = time.time()

    for batch_start in range(0, len(texts), args.batch_size):
        batch_texts = texts[batch_start:batch_start + args.batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        hook_obj = SteeringHook(sv, args.multiplier, input_lengths, prefill_only=args.prefill_only)
        handle = model.model.layers[selected_layer].register_forward_hook(hook_obj)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        handle.remove()

        for i in range(len(batch_texts)):
            inp_len = input_lengths[i]
            gen_ids = output_ids[i][inp_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_outputs.append(text.strip())

        if (batch_start // args.batch_size) % 25 == 0:
            elapsed = time.time() - t0
            done = batch_start + len(batch_texts)
            print(f"  {done}/{len(texts)} examples, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"Inference complete: {len(all_outputs)} outputs in {elapsed:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    n_subset = len(subset_indices)
    suffix = f"_m{args.multiplier}"
    if args.prefill_only:
        suffix += "_prefill"
    out_filename = f"D_llama31_8b_confiqa_mc_{n_subset}{suffix}.jsonl"
    out_path = os.path.join(args.output_dir, out_filename)

    results = []
    for inst, raw_output in zip(data, all_outputs):
        record = {
            "instance_id": inst["instance_id"],
            "raw_output": raw_output,
            "parsed_answer": raw_output,
            "cf_answer": inst["cf_answer"],
            "orig_answer": inst["orig_answer"],
            "cf_alias": inst.get("cf_alias", []),
            "orig_alias": inst.get("orig_alias", []),
            "question": inst["question"],
        }
        results.append(record)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {out_path}")

    print("\n=== Evaluation ===")
    metrics = compute_metrics(results)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    metrics["steering_layer"] = selected_layer
    metrics["multiplier"] = args.multiplier
    metrics["prefill_only"] = args.prefill_only

    metrics_path = out_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
