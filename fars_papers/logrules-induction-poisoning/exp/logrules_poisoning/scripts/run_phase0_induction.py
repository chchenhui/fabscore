"""Phase-0 induction: generate rules for BGL seed=0 under clean and 6 poisoned configs.
Runs gpt-4o-mini API calls (no GPU needed). Falls back to DeepSeek-V3.2 if content filtered.
Usage: python scripts/run_phase0_induction.py
"""

import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.evaluation.split import create_split, save_split
from logrules_poisoning.src.induction.rule_generation import induce_rules
from logrules_poisoning.src.induction.rule_ranking import rank_rules
from logrules_poisoning.src.poisoning.inject import inject_payload

DATASET = "BGL"
SEED = 0
MAX_RULES = 15
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "deepseek-ai/DeepSeek-V3.2"


def run_induction_with_fallback(client, examples, primary_model, fallback_model):
    model_used = primary_model
    try:
        result = induce_rules(client, examples, model=primary_model)
        if result["rules"]:
            return result, model_used
        print(f"  WARNING: {primary_model} returned empty rules, trying fallback...")
    except Exception as e:
        err_str = str(e).lower()
        if "content_filter" in err_str or "content filter" in err_str or "400" in str(e) or "refused" in err_str:
            print(f"  Content filter triggered on {primary_model}: {e}")
        else:
            print(f"  Error with {primary_model}: {e}")

    model_used = fallback_model
    print(f"  Falling back to {fallback_model}...")
    try:
        result = induce_rules(client, examples, model=fallback_model)
        return result, model_used
    except Exception as e:
        print(f"  Fallback also failed: {e}")
        traceback.print_exc()
        return {"rules": [], "num_rules": 0, "raw_response": f"ERROR: {e}"}, model_used


def main():
    load_dotenv(EXP_ROOT / ".env")
    api_key = os.environ.get("LEMMA_MAAS_API_KEY", "")
    base_url = f"http://{os.environ.get('LEMMA_MAAS_BASE_URL', '')}/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Generating BGL split for seed={SEED}...")
    split = create_split(DATASET, SEED)
    induction = split["induction"]
    canary = split["canary"]

    split_dir = PROJECT_ROOT / "data" / "splits" / DATASET / f"seed_{SEED}"
    for name, records in split.items():
        save_split(records, split_dir / f"{name}.jsonl")
    print(f"  Saved splits to {split_dir}")
    print(f"  Induction: {len(induction)}, Canary: {len(canary)}, Test: {len(split['test'])}")

    configs = [("clean", None, 0)]
    for payload in ["A", "B", "C"]:
        for k in [1, 3]:
            configs.append((f"{payload}_k{k}", payload, k))

    all_results = {}

    for config_name, payload_type, k in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")

        if payload_type is None:
            examples = induction
            injection_log = None
            condition_dir = "c0_clean"
        else:
            examples, injection_log = inject_payload(induction, payload_type, k, seed=SEED)
            condition_dir = f"phase0/{payload_type}_k{k}"
            inj_dir = PROJECT_ROOT / "outputs" / "phase0" / f"{payload_type}_k{k}"
            inj_dir.mkdir(parents=True, exist_ok=True)
            with open(inj_dir / "injection_log.json", "w") as f:
                json.dump(injection_log, f, indent=2, ensure_ascii=False)

            print(f"  Injected {len(injection_log['details'])} examples")
            for d in injection_log["details"]:
                changed = d["original_raw_log"] != d["modified_raw_log"]
                print(f"    idx={d['index']} changed={changed}")

        out_dir = PROJECT_ROOT / "outputs" / "rules" / condition_dir / DATASET / f"seed_{SEED}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Inducing rules...")
        gen_result, model_used = run_induction_with_fallback(
            client, examples, PRIMARY_MODEL, FALLBACK_MODEL
        )
        print(f"  Model: {model_used}, Generated {gen_result['num_rules']} rules")

        gen_result["model_used"] = model_used
        with open(out_dir / "raw_rules.json", "w") as f:
            json.dump(gen_result, f, indent=2, ensure_ascii=False)

        rules = gen_result["rules"]
        if not rules:
            print("  WARNING: No rules generated, skipping ranking")
            all_results[config_name] = {"rules": [], "model_used": model_used}
            continue

        print(f"  Ranking rules (max {MAX_RULES})...")
        rank_result = rank_rules(
            client, rules, examples, model=model_used, max_rules=MAX_RULES
        )
        rank_result["model_used"] = model_used
        print(f"  Selected {rank_result['num_selected']} / {rank_result['num_original']} rules")

        with open(out_dir / "ranked_rules.json", "w") as f:
            json.dump(rank_result, f, indent=2, ensure_ascii=False)

        print("  Top rules:")
        for i, rule in enumerate(rank_result["ranked_rules"][:5]):
            print(f"    {i+1}. {rule}")

        all_results[config_name] = {
            "rules": rank_result["ranked_rules"],
            "model_used": model_used,
            "num_rules": rank_result["num_selected"],
        }

    summary_path = PROJECT_ROOT / "outputs" / "phase0" / "induction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nInduction summary saved to {summary_path}")
    print("\nPhase-0 induction complete.")


if __name__ == "__main__":
    main()
