"""Condition E: EACP + ContextFocus composition.
Combines EACP entity-anonymized prompts (condition C) with ContextFocus
activation steering (condition D) using HF model.generate() with forward hooks.
Outputs entity IDs which are parsed and mapped back to entity labels for scoring.

Fixes over original implementation:
  1. Robust answer extraction: HF generate tends to produce verbose output that
     re-states parts of the prompt before answering. We extract the answer from
     after the last 'A:' marker, falling back to last ENT mention.
  2. Configurable steering mode: prefill_only steers context KV cache only;
     generation steers decode tokens; both steers all positions."""

import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, PROJ_DIR)

from eacp.data.confiqa_loader import load_confiqa
from eacp.data.entity_inventory import build_id_to_entity_map
from eacp.evaluation.confiqa_scorer import compute_metrics
from eacp.evaluation.id_parser import map_id_to_entity
from eacp.prompts.condition_c import build_chat_messages_c
from eacp.steering.steered_inference import SteeringHook

VECTORS_DIR = os.path.join(PROJ_DIR, "eacp", "steering", "vectors")
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 8
DEFAULT_MULTIPLIER = 2.0
DEFAULT_LAYER = 13

_ID_PATTERN = re.compile(r"\b(ENT_\d+|UNKNOWN)\b")


def extract_answer_id(raw_output: str) -> str | None:
    """Extract entity ID from HF generate verbose output. The model often
    re-generates context text before the actual answer. We look for the answer
    after the last 'A:' marker, falling back to the last ENT mention."""
    a_marker = raw_output.rfind("A:")
    if a_marker >= 0:
        after_a = raw_output[a_marker:]
        match = _ID_PATTERN.search(after_a)
        if match:
            return match.group(1)

    all_matches = _ID_PATTERN.findall(raw_output)
    if all_matches:
        return all_matches[-1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="MC")
    parser.add_argument("--subset_indices", default=os.path.join(PROJ_DIR, "eacp", "data", "confiqa_mc_1500_subset_indices.json"))
    parser.add_argument("--vectors_path", default=os.path.join(VECTORS_DIR, "llama31_8b_contextfocus.pt"))
    parser.add_argument("--output_dir", default=os.path.join(PROJ_DIR, "eacp", "outputs"))
    parser.add_argument("--multiplier", type=float, default=DEFAULT_MULTIPLIER)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--steer_mode", choices=["prefill", "generation", "both"],
                        default="both",
                        help="Where to apply steering: prefill=context KV only, generation=decode tokens only, both=all positions")
    parser.add_argument("--output_suffix", default=None, help="Custom output filename suffix")
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
    selected_layer = args.layer
    sv = steering_vectors[selected_layer]
    print(f"Steering layer={selected_layer}, multiplier={args.multiplier}, steer_mode={args.steer_mode}")
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

    print("Building EACP prompts (condition C format)...")
    conversations = []
    per_instance_meta = []
    for inst in data:
        msgs, entity_map, _anon_result = build_chat_messages_c(inst)
        id2ent = build_id_to_entity_map(entity_map)
        conversations.append(msgs)
        per_instance_meta.append({"entity_map": entity_map, "id_to_entity_map": id2ent})

    texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations]

    prefill_only = args.steer_mode == "prefill"
    print(f"\nRunning steered inference...")
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

        if args.steer_mode == "both":
            hook_obj = SteeringHook(sv, args.multiplier, input_lengths, prefill_only=False)

            class BothHook:
                def __init__(self, sv, multiplier, input_lengths):
                    self.sv = sv
                    self.multiplier = multiplier
                    self.input_lengths = input_lengths

                def __call__(self, module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    if self.multiplier == 0:
                        return output
                    device = hidden.device
                    sv = self.sv.to(device=device, dtype=hidden.dtype)
                    hidden[:, :, :] = hidden[:, :, :] + self.multiplier * sv
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden

            hook_obj = BothHook(sv, args.multiplier, input_lengths)
        else:
            hook_obj = SteeringHook(sv, args.multiplier, input_lengths, prefill_only=prefill_only)

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
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            all_outputs.append(text)

        if (batch_start // args.batch_size) % 25 == 0:
            elapsed = time.time() - t0
            done = batch_start + len(batch_texts)
            print(f"  {done}/{len(texts)} examples, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"Inference complete: {len(all_outputs)} outputs in {elapsed:.1f}s")

    avg_len = sum(len(o) for o in all_outputs) / len(all_outputs) if all_outputs else 0
    print(f"Average output length: {avg_len:.1f} chars")

    os.makedirs(args.output_dir, exist_ok=True)
    n_subset = len(subset_indices)
    if args.output_suffix:
        suffix = args.output_suffix
    else:
        suffix = f"_m{args.multiplier}_{args.steer_mode}"
    out_filename = f"E_llama31_8b_confiqa_mc_{n_subset}{suffix}.jsonl"
    out_path = os.path.join(args.output_dir, out_filename)

    results = []
    id_parse_successes = 0

    for inst, raw_output, meta in zip(data, all_outputs, per_instance_meta):
        predicted_id = extract_answer_id(raw_output)
        id2ent = meta["id_to_entity_map"]
        parsed = map_id_to_entity(predicted_id, id2ent)

        record = {
            "instance_id": inst["instance_id"],
            "raw_output": raw_output,
            "predicted_id": predicted_id,
            "parsed_answer": parsed,
            "id_to_entity_map": id2ent,
            "id_parse_success": predicted_id is not None,
            "cf_answer": inst["cf_answer"],
            "orig_answer": inst["orig_answer"],
            "cf_alias": inst.get("cf_alias", []),
            "orig_alias": inst.get("orig_alias", []),
            "question": inst["question"],
        }
        results.append(record)
        if predicted_id is not None:
            id_parse_successes += 1

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {out_path}")

    print("\n=== Evaluation ===")
    metrics = compute_metrics(results)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    parse_rate = id_parse_successes * 100.0 / len(results) if results else 0.0
    print(f"  id_parse_success_rate: {round(parse_rate, 2)}")
    metrics["id_parse_success_rate"] = round(parse_rate, 2)
    metrics["steering_layer"] = selected_layer
    metrics["multiplier"] = args.multiplier
    metrics["steer_mode"] = args.steer_mode

    metrics_path = out_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
