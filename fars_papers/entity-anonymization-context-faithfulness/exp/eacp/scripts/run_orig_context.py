"""Orig-context no-harm control experiment for conditions A, B, C.
Swaps cf_context with orig_context in each instance so existing prompt builders
use the original (factual) context. Scores EM against orig_answer (not cf_answer)
with is_cf=False (no negation filter). This checks whether entity anonymization
harms basic QA accuracy when there is no knowledge conflict."""

import argparse
import copy
import json
import os
import sys

PROJ_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, PROJ_DIR)

from vllm import LLM, SamplingParams

from eacp.data.confiqa_loader import load_confiqa
from eacp.data.entity_inventory import build_entity_inventory, build_id_to_entity_map
from eacp.evaluation.confiqa_scorer import exact_match_score, normalize_answer, recall_score
from eacp.evaluation.id_parser import map_id_to_entity, parse_id_output
from eacp.prompts.condition_a import build_chat_messages_a
from eacp.prompts.condition_b import build_chat_messages_b
from eacp.prompts.condition_c import build_chat_messages_c

MODEL_SHORT_NAMES = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama31_8b",
}

ID_BASED_CONDITIONS = {"B", "C"}


def swap_to_orig_context(instance: dict) -> dict:
    inst = copy.deepcopy(instance)
    inst["cf_context"] = inst["orig_context"]
    return inst


def compute_orig_metrics(results: list[dict]) -> dict:
    """Custom scorer for orig-context experiment. EM and context-follow rate
    computed against orig_answer + orig_alias with is_cf=False."""
    n = len(results)
    if n == 0:
        return {"EM": 0.0, "context_follow_rate": 0.0}

    total_em = 0
    total_follow = 0

    for r in results:
        pred = r["parsed_answer"]
        orig_answer = r["orig_answer"]
        orig_alias = r.get("orig_alias", [])

        candidates = [orig_answer] if isinstance(orig_answer, str) else list(orig_answer)
        if orig_alias:
            candidates.extend(orig_alias)

        em = any(exact_match_score(pred, g, is_cf=False) for g in candidates)
        follow = any(recall_score(pred, g, is_cf=False) for g in candidates)

        total_em += int(em)
        total_follow += int(follow)

    return {
        "EM": round(total_em * 100.0 / n, 2),
        "context_follow_rate": round(total_follow * 100.0 / n, 2),
    }


def build_conversations(condition: str, data: list[dict], debug: bool = False):
    conversations = []
    per_instance_meta = []

    for i, inst in enumerate(data):
        swapped = swap_to_orig_context(inst)

        if condition == "A":
            msgs = build_chat_messages_a(swapped)
            meta = {}
        elif condition == "B":
            entity_map = build_entity_inventory(swapped)
            msgs, entity_map = build_chat_messages_b(swapped, entity_map)
            id2ent = build_id_to_entity_map(entity_map)
            meta = {"entity_map": entity_map, "id_to_entity_map": id2ent}
        elif condition == "C":
            msgs, entity_map, _anon_result = build_chat_messages_c(swapped)
            id2ent = build_id_to_entity_map(entity_map)
            meta = {"entity_map": entity_map, "id_to_entity_map": id2ent}
        else:
            raise ValueError(f"Unsupported condition: {condition}")

        if debug and i == 0:
            print(f"\n=== Sample prompt for condition {condition} (instance 0) ===")
            for msg in msgs:
                print(f"[{msg['role']}]: {msg['content'][:500]}")
            print(f"=== orig_answer: {inst['orig_answer']} ===\n")

        conversations.append(msgs)
        per_instance_meta.append(meta)

    return conversations, per_instance_meta


def run_condition(condition, data, llm, sampling_params, output_dir, model_short, debug=False):
    print(f"\n{'='*60}")
    print(f"Running condition {condition} ({len(data)} instances, orig-context)")
    print(f"{'='*60}")

    conversations, per_instance_meta = build_conversations(condition, data, debug=debug)

    print("Running inference...")
    outputs = llm.chat(messages=conversations, sampling_params=sampling_params)

    is_id_based = condition in ID_BASED_CONDITIONS
    results = []
    id_parse_successes = 0

    for inst, output, meta in zip(data, outputs, per_instance_meta):
        raw_output = output.outputs[0].text
        record = {
            "instance_id": inst["instance_id"],
            "raw_output": raw_output,
            "orig_answer": inst["orig_answer"],
            "orig_alias": inst.get("orig_alias", []),
            "cf_answer": inst["cf_answer"],
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

    out_filename = f"{condition}_llama31_8b_confiqa_mc_orig500.jsonl"
    out_path = os.path.join(output_dir, out_filename)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {out_path}")

    metrics = compute_orig_metrics(results)
    if is_id_based:
        parse_rate = id_parse_successes * 100.0 / len(results) if results else 0.0
        metrics["id_parse_success_rate"] = round(parse_rate, 2)

    print(f"\n=== Condition {condition} Metrics (orig-context) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    metrics_path = out_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--split", default="MC")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default=os.path.join(PROJ_DIR, "eacp", "outputs"))
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    indices_path = os.path.join(PROJ_DIR, "eacp", "data", "confiqa_mc_orig_500_subset_indices.json")
    print(f"Loading ConFiQA-{args.split} data...")
    data = load_confiqa(args.split)
    with open(indices_path) as f:
        indices = json.load(f)
    data = [data[i] for i in indices]
    print(f"Using {len(data)}-example orig-context subset")

    if args.limit:
        data = data[:args.limit]
        print(f"Limited to {len(data)} instances (debug mode)")

    debug = args.limit is not None and args.limit <= 10

    print(f"Loading model {args.model} with vLLM...")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    all_metrics = {}
    for condition in ["A", "B", "C"]:
        metrics = run_condition(
            condition, data, llm, sampling_params,
            args.output_dir, MODEL_SHORT_NAMES.get(args.model, args.model.split("/")[-1]),
            debug=debug,
        )
        all_metrics[condition] = metrics

    print("\n" + "=" * 60)
    print("SUMMARY: Orig-Context No-Harm Control")
    print("=" * 60)
    print(f"{'Condition':<12} {'EM':>8} {'CtxFollow':>12}")
    print("-" * 36)
    for cond in ["A", "B", "C"]:
        m = all_metrics[cond]
        print(f"{cond:<12} {m['EM']:>8.2f} {m['context_follow_rate']:>12.2f}")

    em_a = all_metrics["A"]["EM"]
    em_b = all_metrics["B"]["EM"]
    em_c = all_metrics["C"]["EM"]
    diff_ca = abs(em_c - em_a)
    diff_cb = abs(em_c - em_b)
    print(f"\nC vs A EM diff: {diff_ca:.2f}")
    print(f"C vs B EM diff: {diff_cb:.2f}")
    passed = diff_ca <= 2.0 and diff_cb <= 2.0
    print(f"No-harm criterion (C within 2 pts of A and B): {'PASS' if passed else 'FAIL'}")

    summary_path = os.path.join(args.output_dir, "orig_context_noharm_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"metrics": all_metrics, "no_harm_pass": passed,
                    "diff_C_vs_A": diff_ca, "diff_C_vs_B": diff_cb}, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
