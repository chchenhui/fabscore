"""Shared vLLM-based batched inference pipeline for all conditions.
Uses vLLM offline LLM class for high-throughput batch generation."""

import argparse
import json
import os
import sys

PROJ_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJ_DIR)

from vllm import LLM, SamplingParams

from eacp.data.confiqa_loader import load_confiqa
from eacp.data.entity_inventory import build_entity_inventory, build_id_to_entity_map
from eacp.evaluation.confiqa_scorer import compute_metrics
from eacp.evaluation.id_parser import map_id_to_entity, parse_id_output
from eacp.prompts.condition_a import build_chat_messages_a
from eacp.prompts.condition_b import build_chat_messages_b
from eacp.prompts.condition_c import build_chat_messages_c

MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
}

SIMPLE_PROMPT_CONDITIONS = {"A"}
ID_BASED_CONDITIONS = {"B", "C"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--split", default="MC")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances (for debugging)")
    parser.add_argument("--subset_indices", type=str, default=None, help="Path to JSON file with subset indices")
    parser.add_argument("--output_dir", default=os.path.join(PROJ_DIR, "eacp", "outputs"))
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ConFiQA-{args.split} data...")
    data = load_confiqa(args.split)
    subset_size = None
    if args.subset_indices:
        with open(args.subset_indices) as f:
            indices = json.load(f)
        data = [data[i] for i in indices]
        subset_size = len(indices)
        print(f"Using {subset_size}-example subset")
    if args.limit:
        data = data[:args.limit]
    print(f"Loaded {len(data)} instances.")

    print("Building prompts...")
    conversations = []
    per_instance_meta = []

    for inst in data:
        if args.condition == "A":
            conversations.append(build_chat_messages_a(inst))
            per_instance_meta.append({})
        elif args.condition == "B":
            entity_map = build_entity_inventory(inst)
            msgs, entity_map = build_chat_messages_b(inst, entity_map)
            id2ent = build_id_to_entity_map(entity_map)
            conversations.append(msgs)
            per_instance_meta.append({"entity_map": entity_map, "id_to_entity_map": id2ent})
        elif args.condition == "C":
            msgs, entity_map, _anon_result = build_chat_messages_c(inst)
            id2ent = build_id_to_entity_map(entity_map)
            conversations.append(msgs)
            per_instance_meta.append({"entity_map": entity_map, "id_to_entity_map": id2ent})
        else:
            raise ValueError(f"Condition {args.condition} not yet implemented")

    print(f"Loading model {args.model} with vLLM...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
    )

    print("Running inference...")
    outputs = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
    )

    model_short = MODEL_SHORT_NAMES.get(args.model, args.model.split("/")[-1])
    benchmark = f"confiqa_{args.split.lower()}"
    if subset_size:
        out_filename = f"{args.condition}_{model_short}_{benchmark}_{subset_size}.jsonl"
    else:
        out_filename = f"{args.condition}_{model_short}_{benchmark}.jsonl"
    out_path = os.path.join(args.output_dir, out_filename)

    is_id_based = args.condition in ID_BASED_CONDITIONS
    results = []
    id_parse_successes = 0

    for inst, output, meta in zip(data, outputs, per_instance_meta):
        raw_output = output.outputs[0].text
        record = {
            "instance_id": inst["instance_id"],
            "raw_output": raw_output,
            "cf_answer": inst["cf_answer"],
            "orig_answer": inst["orig_answer"],
            "cf_alias": inst.get("cf_alias", []),
            "orig_alias": inst.get("orig_alias", []),
            "question": inst["question"],
        }

        if is_id_based:
            predicted_id = parse_id_output(raw_output)
            id2ent = meta["id_to_entity_map"]
            parsed = map_id_to_entity(predicted_id, id2ent)
            record["predicted_id"] = predicted_id
            record["id_to_entity_map"] = id2ent
            record["id_parse_success"] = predicted_id is not None
            if predicted_id is not None:
                id_parse_successes += 1
        else:
            parsed = raw_output.strip()

        record["parsed_answer"] = parsed
        results.append(record)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {out_path}")

    print("\n=== Evaluation ===")
    metrics = compute_metrics(results)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if is_id_based:
        parse_rate = id_parse_successes * 100.0 / len(results) if results else 0.0
        print(f"  id_parse_success_rate: {round(parse_rate, 2)}")
        metrics["id_parse_success_rate"] = round(parse_rate, 2)

    metrics_path = out_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
