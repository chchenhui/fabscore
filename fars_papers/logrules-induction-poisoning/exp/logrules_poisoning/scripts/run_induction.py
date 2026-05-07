"""Run LogRules induction + ranking for all datasets and seeds.
Calls gpt-4o-mini API to induce rules from K=10 examples, ranks them, selects top N=15.
No GPU required -- API-only.
Usage: python scripts/run_induction.py [--condition c0_clean] [--datasets BGL Linux HDFS]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(EXP_ROOT))

from logrules_poisoning.src.induction.rule_generation import induce_rules
from logrules_poisoning.src.induction.rule_ranking import rank_rules

DATASETS = ["BGL", "Linux", "HDFS"]
SEEDS = [42, 123, 456]


def load_induction_set(dataset: str, seed: int) -> list:
    path = PROJECT_ROOT / "data" / "splits" / dataset / f"seed_{seed}" / "induction.jsonl"
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="c0_clean")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-rules", type=int, default=15)
    args = parser.parse_args()

    load_dotenv(EXP_ROOT / ".env")

    api_key = os.environ.get("LEMMA_MAAS_API_KEY", "")
    base_url = f"http://{os.environ.get('LEMMA_MAAS_BASE_URL', '')}/v1"

    client = OpenAI(api_key=api_key, base_url=base_url)

    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"Induction: {dataset} / seed_{seed} / {args.condition}")
            print(f"{'='*60}")

            examples = load_induction_set(dataset, seed)
            print(f"Loaded {len(examples)} induction examples")

            out_dir = PROJECT_ROOT / "outputs" / "rules" / args.condition / dataset / f"seed_{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"Inducing rules via {args.model}...")
            gen_result = induce_rules(client, examples, model=args.model)
            print(f"  Generated {gen_result['num_rules']} rules")

            raw_path = out_dir / "raw_rules.json"
            with open(raw_path, "w") as f:
                json.dump(gen_result, f, indent=2, ensure_ascii=False)
            print(f"  Saved raw rules to {raw_path}")

            rules = gen_result["rules"]
            if not rules:
                print("  WARNING: No rules generated, skipping ranking")
                continue

            print(f"Ranking rules (max {args.max_rules})...")
            rank_result = rank_rules(
                client, rules, examples,
                model=args.model, max_rules=args.max_rules,
            )
            print(f"  Selected {rank_result['num_selected']} / {rank_result['num_original']} rules")

            ranked_path = out_dir / "ranked_rules.json"
            with open(ranked_path, "w") as f:
                json.dump(rank_result, f, indent=2, ensure_ascii=False)
            print(f"  Saved ranked rules to {ranked_path}")

            print("  Top rules:")
            for i, rule in enumerate(rank_result["ranked_rules"][:5]):
                print(f"    {i+1}. {rule}")
            if rank_result["num_selected"] > 5:
                print(f"    ... ({rank_result['num_selected'] - 5} more)")

    print("\nInduction complete.")


if __name__ == "__main__":
    main()
