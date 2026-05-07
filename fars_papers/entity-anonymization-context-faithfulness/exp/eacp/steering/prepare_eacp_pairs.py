"""Build positive/negative EACP prompt pairs from ConFiQA-MC data for computing
an EACP-native ContextFocus steering vector. Positive = full EACP prompt (anonymized
context + question + entity inventory). Negative = question + inventory only (no
context). Uses 1500 held-out examples (NOT the evaluation subset)."""

import json
import os
import sys

from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, PROJ_DIR)

from eacp.data.confiqa_loader import load_confiqa
from eacp.data.entity_anonymizer import anonymize_instance
from eacp.prompts.condition_c import build_inventory_block_C, build_prompt_c

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_MESSAGE = "Answer concisely with the answer only."
N_PAIRS = 1500
SEED = 42


def build_negative_prompt_eacp(anon_question: str, inventory_block: str) -> str:
    if anon_question.endswith("?"):
        anon_question = anon_question[:-1]
    prompt = (
        f"{inventory_block}\n\n"
        "Instruction: read the given information and answer the corresponding question. "
        "Answer with exactly one entity ID (e.g., ENT_1) or UNKNOWN. Do not explain.\n\n"
        f"Q: {anon_question}?\n"
        "A:"
    )
    return prompt


def main():
    data = load_confiqa("MC")
    print(f"Loaded {len(data)} ConFiQA-MC examples")

    eval_indices_path = os.path.join(PROJ_DIR, "eacp", "data", "confiqa_mc_1500_subset_indices.json")
    with open(eval_indices_path) as f:
        eval_indices = set(json.load(f))

    non_eval = [i for i in range(len(data)) if i not in eval_indices]
    print(f"Non-eval indices: {len(non_eval)}")

    import random
    rng = random.Random(SEED)
    rng.shuffle(non_eval)
    selected = non_eval[:N_PAIRS]
    print(f"Selected {len(selected)} examples for steering pair generation")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    pairs = []
    for idx in selected:
        inst = data[idx]
        anon_result = anonymize_instance(inst)
        entity_map = anon_result["entity_map"]
        inventory_block = build_inventory_block_C(
            entity_map,
            replacement_stats=anon_result["replacement_stats"],
        )

        pos_user = build_prompt_c(
            anon_result["anon_context"],
            anon_result["anon_question"],
            inventory_block,
        )
        pos_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": pos_user},
        ]
        pos_text = tokenizer.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=True)

        neg_user = build_negative_prompt_eacp(
            anon_result["anon_question"],
            inventory_block,
        )
        neg_messages = [
            {"role": "user", "content": neg_user},
        ]
        neg_text = tokenizer.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=True)

        pairs.append({
            "idx": idx,
            "positive_text": pos_text,
            "negative_text": neg_text,
            "cf_answer": inst["cf_answer"],
            "orig_answer": inst["orig_answer"],
        })

    out_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eacp_steering_pairs.json")
    with open(out_path, "w") as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"Saved {len(pairs)} EACP steering pairs to {out_path}")


if __name__ == "__main__":
    main()
