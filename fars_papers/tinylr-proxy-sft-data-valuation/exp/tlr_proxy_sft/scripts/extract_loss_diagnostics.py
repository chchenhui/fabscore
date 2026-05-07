"""Extract per-run loss diagnostics from trainer_log.jsonl files.
Outputs initial loss (step 10), final loss (last step), and loss drop percentage.
Usage: python extract_loss_diagnostics.py --regime proxy_tiny"""
import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path("/mnt/bmcpfs-29000zjpjtl6xjmjiifyk/fars/fars-exp/live/exp/tinylr-proxy-sft-data-valuation/exp")

DATASETS = [
    "AM-Thinking-v1-Distilled-math", "DeepMath-309K", "Maths-College",
    "OpenR1-Math", "QwQ-LongCoT-130K-math", "R1-Distill-SFT-math",
    "hkust-nlp__dart-math-hard", "mathplus", "numinamath-cot",
    "numinamath1_5", "openmathinstruct-2", "Magpie-Reasoning-V2-250K-CoT-QwQ-math",
]
SEEDS = [42, 123, 456]


def extract_one(log_path: Path) -> tuple:
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "loss" in entry and "current_steps" in entry:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    if not entries:
        return None, None, None

    entries.sort(key=lambda e: e["current_steps"])
    initial_loss = entries[0]["loss"]
    final_loss = entries[-1]["loss"]
    loss_drop_pct = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0.0
    return initial_loss, final_loss, loss_drop_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, required=True, choices=["proxy_std", "proxy_tiny", "proxy_mid", "proxy_tiny_v2"])
    args = parser.parse_args()

    outputs_dir = PROJECT_ROOT / "tlr_proxy_sft" / "outputs" / args.regime
    results_dir = PROJECT_ROOT / "tlr_proxy_sft" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in DATASETS:
        for seed in SEEDS:
            log_path = outputs_dir / ds / f"seed_{seed}" / "trainer_log.jsonl"
            if not log_path.exists():
                print(f"MISSING: {ds}/seed_{seed}")
                rows.append({"dataset": ds, "seed": seed, "initial_loss": None, "final_loss": None, "loss_drop_pct": None})
                continue

            initial, final, drop = extract_one(log_path)
            rows.append({"dataset": ds, "seed": seed, "initial_loss": initial, "final_loss": final, "loss_drop_pct": drop})
            print(f"{ds}/seed_{seed}: initial={initial:.4f}, final={final:.4f}, drop={drop:.1f}%")

    csv_path = results_dir / f"{args.regime}_loss_diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "seed", "initial_loss", "final_loss", "loss_drop_pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {csv_path}")
    print(f"Total runs: {len(rows)}, with data: {sum(1 for r in rows if r['initial_loss'] is not None)}")

    valid = [r for r in rows if r["loss_drop_pct"] is not None]
    if valid:
        avg_drop = sum(r["loss_drop_pct"] for r in valid) / len(valid)
        min_drop = min(r["loss_drop_pct"] for r in valid)
        max_drop = max(r["loss_drop_pct"] for r in valid)
        print(f"Loss drop: avg={avg_drop:.1f}%, min={min_drop:.1f}%, max={max_drop:.1f}%")


if __name__ == "__main__":
    main()
