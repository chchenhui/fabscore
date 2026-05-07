# Log ablation evaluation results to WandB as summary metrics.
# Usage: python scripts/log_ablation_wandb.py

import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import wandb

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIGS = {
    "fastv_final": {"name": "FastV (L2, top-k)", "debiasing": "None"},
    "d2pruner_final": {"name": "D2Pruner (L2, offline)", "debiasing": "Offline prior"},
    "online_rawmis_km4": {"name": "A_mid+MIS (L4)", "debiasing": "None"},
    "opt_rawmis_km12": {"name": "A_mid+MIS (L12)", "debiasing": "None"},
    "online_ratio_ks3_km4": {"name": "Ratio Ks3 (L4)", "debiasing": "Online ratio"},
    "ablation_ratio_ks3_km12": {"name": "Ratio Ks3 (L12)", "debiasing": "Online ratio"},
    "ablation_ratio_ks2_km12": {"name": "Ratio Ks2 (L12)", "debiasing": "Online ratio"},
    "opt_wc_a05_ks2_km12": {"name": "WeightedCombo (L12)", "debiasing": "Weighted combo"},
    "nopruning_final": {"name": "No Pruning", "debiasing": "N/A"},
}

def load_results(results_dir):
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)

def main():
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "layer-ratio-attention-debias-vlm-pruning"),
        name="ablation-amid-only",
        tags=["ablation", "amid-only", "ratio-debiasing"],
        config={
            "experiment": "Ablation: A_mid-only vs ratio debiasing",
            "model": "InternVL2.5-8B",
            "keep_ratio": 0.1,
            "metric": "grounding_accuracy_iou05",
        },
    )

    results_base = BASE_DIR / "results"
    all_data = {}

    for config_key, config_info in CONFIGS.items():
        results_dir = results_base / config_key
        data = load_results(results_dir)
        if data is None:
            print(f"Skipping {config_key}: no summary.json found")
            continue

        print(f"\n{config_info['name']} ({config_key}):")
        splits = {}
        for k, v in sorted(data.items()):
            if isinstance(v, dict) and "accuracy" in v:
                acc = v["accuracy"]
                splits[k] = acc
                metric_key = f"{config_key}/{k}"
                wandb.log({metric_key: acc * 100})
                print(f"  {k}: {acc:.4f}")

        if "average" in data:
            avg = data["average"]["accuracy"]
        else:
            accs = [v for v in splits.values()]
            avg = sum(accs) / len(accs) if accs else 0

        wandb.log({f"{config_key}/average": avg * 100})
        run.summary[f"{config_key}_avg"] = avg * 100
        print(f"  average: {avg:.4f}")
        all_data[config_key] = {"info": config_info, "splits": splits, "average": avg}

    columns = ["Variant", "Debiasing", "RefCOCO avg", "RefCOCO+ avg", "RefCOCOg avg", "Overall avg"]
    table = wandb.Table(columns=columns)

    for config_key, d in all_data.items():
        refcoco_splits = [d["splits"].get(s, 0) for s in ["refcoco_val", "refcoco_testA", "refcoco_testB"]]
        refcoco_plus_splits = [d["splits"].get(s, 0) for s in ["refcoco+_val", "refcoco+_testA", "refcoco+_testB"]]
        refcocog_splits = [d["splits"].get(s, 0) for s in ["refcocog_val", "refcocog_test"]]

        refcoco_avg = sum(refcoco_splits) / len(refcoco_splits) * 100 if refcoco_splits else 0
        refcoco_plus_avg = sum(refcoco_plus_splits) / len(refcoco_plus_splits) * 100 if refcoco_plus_splits else 0
        refcocog_avg = sum(refcocog_splits) / len(refcocog_splits) * 100 if refcocog_splits else 0

        table.add_data(
            d["info"]["name"],
            d["info"]["debiasing"],
            f"{refcoco_avg:.2f}",
            f"{refcoco_plus_avg:.2f}",
            f"{refcocog_avg:.2f}",
            f"{d['average'] * 100:.2f}",
        )

    wandb.log({"ablation_table": table})
    run.finish()
    print("\nWandB logging complete.")

if __name__ == "__main__":
    main()
