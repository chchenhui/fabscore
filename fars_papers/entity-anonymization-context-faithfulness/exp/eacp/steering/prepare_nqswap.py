"""Download NQ-SWAP and prepare positive/negative prompt pairs for ContextFocus
steering vector generation. Creates 1,501 estimation + 200 layer-selection splits.
Each example gets a positive prompt (system + context + question) and negative
prompt (question only), both formatted with Llama-3.1 chat template."""

import json
import os
import random

from datasets import load_dataset
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
VARIANTS_PATH = os.path.join(DATA_DIR, "system_instruction_variants.json")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
N_ESTIMATION = 1501
N_LAYERSELECT = 200


def build_positive_prompt(context: str, question: str) -> str:
    """Paper format: 'Context: ... Question: ...' (not O&I Bob format)."""
    prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
    return prompt


def build_negative_prompt(question: str) -> str:
    return f"Question: {question}"


def main():
    with open(VARIANTS_PATH) as f:
        system_variants = json.load(f)
    assert len(system_variants) == 20

    print("Loading NQ-SWAP dataset from HuggingFace...")
    ds = load_dataset("pminervini/NQ-Swap", split="dev")
    print(f"Loaded {len(ds)} examples")

    indices = list(range(len(ds)))
    rng = random.Random(SEED)
    rng.shuffle(indices)

    est_indices = indices[:N_ESTIMATION]
    ls_indices = indices[N_ESTIMATION:N_ESTIMATION + N_LAYERSELECT]

    print(f"Estimation split: {len(est_indices)} examples")
    print(f"Layer selection split: {len(ls_indices)} examples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def build_pairs(idx_list, rng_local):
        pairs = []
        for idx in idx_list:
            ex = ds[idx]
            sys_instr = rng_local.choice(system_variants)

            pos_user = build_positive_prompt(ex["sub_context"], ex["question"])
            pos_messages = [
                {"role": "system", "content": sys_instr},
                {"role": "user", "content": pos_user},
            ]
            pos_text = tokenizer.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=True)

            neg_user = build_negative_prompt(ex["question"])
            neg_messages = [
                {"role": "user", "content": neg_user},
            ]
            neg_text = tokenizer.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=True)

            pairs.append({
                "idx": idx,
                "question": ex["question"],
                "sub_answer": ex["sub_answer"],
                "org_answer": ex["org_answer"],
                "positive_text": pos_text,
                "negative_text": neg_text,
                "system_variant": sys_instr,
            })
        return pairs

    rng_est = random.Random(SEED + 1)
    rng_ls = random.Random(SEED + 2)

    est_pairs = build_pairs(est_indices, rng_est)
    ls_pairs = build_pairs(ls_indices, rng_ls)

    os.makedirs(DATA_DIR, exist_ok=True)

    est_path = os.path.join(DATA_DIR, "nqswap_steering_pairs.json")
    with open(est_path, "w") as f:
        json.dump(est_pairs, f, ensure_ascii=False)
    print(f"Saved {len(est_pairs)} estimation pairs to {est_path}")

    ls_path = os.path.join(DATA_DIR, "nqswap_layerselect.json")
    with open(ls_path, "w") as f:
        json.dump(ls_pairs, f, ensure_ascii=False)
    print(f"Saved {len(ls_pairs)} layer-selection pairs to {ls_path}")

    print("\nSample positive prompt (first):")
    print(est_pairs[0]["positive_text"][:500])
    print("\nSample negative prompt (first):")
    print(est_pairs[0]["negative_text"][:500])


if __name__ == "__main__":
    main()
