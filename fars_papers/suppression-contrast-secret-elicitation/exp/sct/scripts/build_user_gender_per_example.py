# Build per-example comparison JSON for User Gender/Direct across SCT, logit lens, DoLa.
# For each example, records top-20 tokens+scores from each method, and which
# gender-indicative tokens appear at what rank.

import json
import os
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
os.chdir(PROJECT_ROOT)

GENDER_TARGET_TOKENS = {
    "female": ["female", "woman", "she", "her", "girl"],
    "male": ["male", "man", "he", "him", "boy"],
}


def get_gender_token_ids(gender: str, tokenizer) -> List[int]:
    words = GENDER_TARGET_TOKENS.get(gender, [])
    result = set()
    for word in words:
        ids_bare = tokenizer.encode(word, add_special_tokens=False)
        ids_space = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids_bare) == 1:
            result.add(ids_bare[0])
        if len(ids_space) >= 1:
            result.add(ids_space[-1])
    return list(result)


def find_target_ranks(ranked_tokens: List[Dict], target_ids: Set[int]) -> List[Dict]:
    found = []
    for rank_idx, t in enumerate(ranked_tokens):
        if t["token_id"] in target_ids:
            found.append({
                "token_id": t["token_id"],
                "token": t.get("token", ""),
                "rank": rank_idx + 1,
                "score": t.get("score", 0.0),
            })
    return found


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

    genders = ["female", "male"]
    gender_token_ids = {}
    gender_token_id_sets = {}
    for g in genders:
        ids = get_gender_token_ids(g, tokenizer)
        gender_token_ids[g] = ids
        gender_token_id_sets[g] = set(ids)

    scored_files = {
        "female": {
            "sct": "sct/outputs/user_gender_female_sct_scored.json",
            "logit_lens": "sct/outputs/user_gender_female_scored.json",
            "dola_direction": "sct/outputs/user_gender_female_dola_scored.json",
        },
        "male": {
            "sct": "sct/outputs/user_gender_male_sct_scored.json",
            "logit_lens": "sct/outputs/user_gender_male_scored.json",
            "dola_direction": "sct/outputs/user_gender_male_dola_scored.json",
        },
    }

    all_data = {}
    for g in genders:
        all_data[g] = {}
        for method, path in scored_files[g].items():
            print(f"Loading {g}/{method}: {path}")
            with open(path) as f:
                all_data[g][method] = json.load(f)["results"]

    methods = ["sct", "logit_lens", "dola_direction"]
    examples = []

    for g in genders:
        target_set = gender_token_id_sets[g]
        n = len(all_data[g]["sct"])
        for i in range(n):
            sct_item = all_data[g]["sct"][i]
            ll_item = all_data[g]["logit_lens"][i]
            dola_item = all_data[g]["dola_direction"][i]

            if "error" in sct_item:
                continue

            example = {
                "model": f"user_gender_{g}",
                "secret_word": g,
                "example_index": i,
                "user_prompt": sct_item.get("user_prompt", ""),
                "model_response": sct_item.get("model_response", "")[:200],
            }

            for method, item in [("sct", sct_item), ("logit_lens", ll_item), ("dola_direction", dola_item)]:
                ranked = item.get("ranked_tokens", [])
                top20 = []
                for t in ranked[:20]:
                    entry = {
                        "token_id": t["token_id"],
                        "token": t.get("token", ""),
                        "score": t.get("score", 0.0),
                    }
                    if "position_count" in t:
                        entry["position_count"] = t["position_count"]
                    top20.append(entry)
                example[f"{method}_top20"] = top20
                example[f"{method}_gender_hits"] = find_target_ranks(ranked[:20], target_set)

            examples.append(example)

    token_id_strs = {}
    for g in genders:
        for tid in gender_token_ids[g]:
            token_id_strs[str(tid)] = tokenizer.decode([tid])

    output = {
        "description": "Per-example comparison of SCT, Constrained Logit Lens, and DoLa-Direction scoring on User Gender/Direct benchmark",
        "num_examples": len(examples),
        "gender_target_token_ids": {g: gender_token_ids[g] for g in genders},
        "gender_target_token_strs": {g: [tokenizer.decode([tid]) for tid in gender_token_ids[g]] for g in genders},
        "methods": methods,
        "examples": examples,
    }

    out_path = "sct/outputs/user_gender_direct_per_example_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved {len(examples)} examples to {out_path}")

    for g in genders:
        g_examples = [e for e in examples if e["secret_word"] == g]
        print(f"\n{g} ({len(g_examples)} examples):")
        for method in methods:
            hits = sum(1 for e in g_examples if e[f"{method}_gender_hits"])
            print(f"  {method}: {hits}/{len(g_examples)} examples with gender tokens in top-20 ({hits/len(g_examples)*100:.1f}%)")


if __name__ == "__main__":
    main()
