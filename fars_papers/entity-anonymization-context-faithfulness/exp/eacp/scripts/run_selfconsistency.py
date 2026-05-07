"""Self-consistency inference for condition C (EACP). Generates multiple
samples per instance and selects the majority-voted entity ID answer.
Falls back to greedy if all samples are UNKNOWN or unparseable."""

import argparse
import json
import os
import sys
from collections import Counter

PROJ_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJ_DIR)

from vllm import LLM, SamplingParams

from eacp.data.confiqa_loader import load_confiqa
from eacp.data.entity_inventory import build_id_to_entity_map
from eacp.evaluation.confiqa_scorer import compute_metrics
from eacp.evaluation.id_parser import map_id_to_entity, parse_id_output
from eacp.prompts.condition_c import build_chat_messages_c


def majority_vote(raw_outputs: list[str], id_to_entity_map: dict[str, str]) -> tuple[str, str, str]:
    parsed_ids = []
    for raw in raw_outputs:
        pid = parse_id_output(raw)
        if pid is not None:
            parsed_ids.append(pid)

    if not parsed_ids:
        return None, "", raw_outputs[0]

    ent_ids_no_unknown = [p for p in parsed_ids if p != "UNKNOWN"]
    if ent_ids_no_unknown:
        counter = Counter(ent_ids_no_unknown)
        winner_id = counter.most_common(1)[0][0]
    else:
        winner_id = "UNKNOWN"

    winner_entity = map_id_to_entity(winner_id, id_to_entity_map)
    winner_raw = next((r for r, p in zip(raw_outputs, [parse_id_output(r) for r in raw_outputs]) if p == winner_id), raw_outputs[0])
    return winner_id, winner_entity, winner_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--split", default="MC")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--subset_indices", type=str, default=None)
    parser.add_argument("--output_dir", default=os.path.join(PROJ_DIR, "eacp", "outputs"))
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_suffix", type=str, default="sc")
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
        data = data[: args.limit]
    print(f"Loaded {len(data)} instances.")

    print("Building prompts...")
    conversations = []
    per_instance_meta = []
    instance_indices = []

    for inst in data:
        msgs, entity_map, _anon = build_chat_messages_c(inst)
        id2ent = build_id_to_entity_map(entity_map)
        for _ in range(args.n_samples):
            conversations.append(msgs)
        per_instance_meta.append({"entity_map": entity_map, "id_to_entity_map": id2ent})
        instance_indices.append(inst["instance_id"])

    print(f"Total prompts to process: {len(conversations)} ({len(data)} instances x {args.n_samples} samples)")

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
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("Running inference...")
    outputs = llm.chat(messages=conversations, sampling_params=sampling_params)

    MODEL_SHORT_NAMES = {
        "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b",
        "Qwen/Qwen2.5-7B-Instruct": "qwen25_7b",
    }
    model_short = MODEL_SHORT_NAMES.get(args.model, args.model.split("/")[-1])
    benchmark = f"confiqa_{args.split.lower()}"
    size_str = f"_{subset_size}" if subset_size else ""
    out_filename = f"C_{model_short}_{benchmark}{size_str}_{args.output_suffix}.jsonl"
    out_path = os.path.join(args.output_dir, out_filename)

    results = []
    for i, (inst, meta) in enumerate(zip(data, per_instance_meta)):
        raw_outputs = [outputs[i * args.n_samples + j].outputs[0].text for j in range(args.n_samples)]
        id2ent = meta["id_to_entity_map"]

        winner_id, winner_entity, winner_raw = majority_vote(raw_outputs, id2ent)

        record = {
            "instance_id": inst["instance_id"],
            "raw_output": winner_raw,
            "raw_outputs_all": raw_outputs,
            "predicted_id": winner_id,
            "parsed_answer": winner_entity,
            "cf_answer": inst["cf_answer"],
            "orig_answer": inst["orig_answer"],
            "cf_alias": inst.get("cf_alias", []),
            "orig_alias": inst.get("orig_alias", []),
            "question": inst["question"],
            "id_to_entity_map": id2ent,
            "id_parse_success": winner_id is not None,
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

    id_parse_successes = sum(1 for r in results if r["id_parse_success"])
    parse_rate = id_parse_successes * 100.0 / len(results) if results else 0.0
    print(f"  id_parse_success_rate: {round(parse_rate, 2)}")
    metrics["id_parse_success_rate"] = round(parse_rate, 2)

    metrics_path = out_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
