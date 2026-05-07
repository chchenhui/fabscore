"""Set up C1 poisoned evaluation by reorganizing Phase-0v2 artifacts.
Phase-0v2 used identical pipeline (gpt-4o-mini induction, N=15 ranking,
Qwen2.5-7B vLLM deduction). This script copies existing rules and predictions
into the C1 output structure, and prints qualifying payload analysis.
Usage: python scripts/setup_c1_from_phase0v2.py
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = PROJECT_ROOT.parent

DATASETS = ["BGL", "Linux", "HDFS"]
SEEDS = [42, 123, 456]
PAYLOADS = ["D", "E", "F"]
K_VALUES = [1, 3, 5, 7]


def analyze_qualifying_payloads():
    results_path = PROJECT_ROOT / "results" / "phase0v2_results.json"
    with open(results_path) as f:
        data = json.load(f)

    passing_by_payload = {p: [] for p in PAYLOADS}
    for key, val in data.items():
        if val.get("passes_criterion"):
            passing_by_payload[val["payload"]].append(key)

    print("=== Qualifying Payload Analysis ===")
    for p in PAYLOADS:
        configs = passing_by_payload[p]
        print(f"  Payload {p}: {len(configs)} passing configs")
        for c in configs:
            v = data[c]
            print(f"    {c}: PA_drop={v['PA_drop']:.4f}, test_PA_drop={v['test_PA_drop']:.4f}")

    mean_drop = {}
    for p in PAYLOADS:
        drops = [data[k]["test_PA_drop"] for k in data if data[k]["payload"] == p]
        mean_drop[p] = sum(drops) / len(drops) if drops else 0
    best = max(mean_drop, key=mean_drop.get)
    print(f"\n  Representative payload (highest mean test PA drop): {best} ({mean_drop[best]:.4f})")
    print(f"  All payloads qualify: {PAYLOADS}")
    print(f"  Config matrix: payloads={PAYLOADS}, k={K_VALUES}, datasets={DATASETS}, seeds={SEEDS}")
    print(f"  Total configs: {len(PAYLOADS) * len(K_VALUES) * len(DATASETS) * len(SEEDS)}")
    return best


def copy_artifacts():
    copied = 0
    errors = []

    for payload in PAYLOADS:
        for k in K_VALUES:
            for dataset in DATASETS:
                for seed in SEEDS:
                    src_rules = PROJECT_ROOT / "outputs" / "rules" / "phase0v2" / f"{payload}_k{k}" / dataset / f"seed_{seed}"
                    dst_rules = PROJECT_ROOT / "outputs" / "rules" / "c1_poisoned" / payload / dataset / f"k{k}" / f"seed_{seed}"

                    src_preds = PROJECT_ROOT / "outputs" / "predictions" / "phase0v2" / f"{payload}_k{k}" / dataset / f"seed_{seed}"
                    dst_preds = PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned" / payload / dataset / f"k{k}" / f"seed_{seed}"

                    for src, dst in [(src_rules, dst_rules), (src_preds, dst_preds)]:
                        if not src.exists():
                            errors.append(f"MISSING: {src}")
                            continue
                        dst.mkdir(parents=True, exist_ok=True)
                        for f in src.iterdir():
                            shutil.copy2(f, dst / f.name)
                            copied += 1

    print(f"\nCopied {copied} files")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    return len(errors) == 0


def verify_artifacts():
    ok = 0
    missing = []

    for payload in PAYLOADS:
        for k in K_VALUES:
            for dataset in DATASETS:
                for seed in SEEDS:
                    rules_path = PROJECT_ROOT / "outputs" / "rules" / "c1_poisoned" / payload / dataset / f"k{k}" / f"seed_{seed}" / "ranked_rules.json"
                    test_path = PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned" / payload / dataset / f"k{k}" / f"seed_{seed}" / "test_predictions.jsonl"
                    canary_path = PROJECT_ROOT / "outputs" / "predictions" / "c1_poisoned" / payload / dataset / f"k{k}" / f"seed_{seed}" / "canary_predictions.jsonl"

                    for p in [rules_path, test_path, canary_path]:
                        if p.exists() and p.stat().st_size > 0:
                            ok += 1
                        else:
                            missing.append(str(p))

    total = len(PAYLOADS) * len(K_VALUES) * len(DATASETS) * len(SEEDS) * 3
    print(f"\nVerification: {ok}/{total} files OK")
    if missing:
        print(f"Missing/empty ({len(missing)}):")
        for m in missing[:10]:
            print(f"  {m}")
    return len(missing) == 0


def main():
    best_payload = analyze_qualifying_payloads()
    print("\n=== Copying Phase-0v2 artifacts to C1 structure ===")
    copy_ok = copy_artifacts()
    verify_ok = verify_artifacts()

    config = {
        "qualifying_payloads": PAYLOADS,
        "representative_payload": best_payload,
        "k_values": K_VALUES,
        "datasets": DATASETS,
        "seeds": SEEDS,
        "total_configs": len(PAYLOADS) * len(K_VALUES) * len(DATASETS) * len(SEEDS),
    }
    config_path = PROJECT_ROOT / "results" / "c1_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    if copy_ok and verify_ok:
        print("\nC1 setup complete. All artifacts ready.")
    else:
        print("\nC1 setup completed with errors. Check above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
