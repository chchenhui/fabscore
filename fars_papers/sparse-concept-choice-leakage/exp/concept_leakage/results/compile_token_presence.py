"""Compile token-presence probe results into a summary CSV table.
Rows: Clean, Isotropic (B), Anisotropic (A), Smoothed (C).
Columns: Token-Presence AUC (mean+/-std), Token-Presence Accuracy (mean+/-std)."""

import json
import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "token_presence"

CONDITION_LABELS = {
    "clean": "Clean (no noise)",
    "isotropic": "Isotropic (B)",
    "anisotropic": "Anisotropic (A)",
    "smoothed": "Smoothed (C, λ=0.2)",
}

CONDITION_ORDER = ["clean", "isotropic", "anisotropic", "smoothed"]


def main():
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "token_presence_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition",
            "AUC_mean", "AUC_std", "AUC_formatted",
            "Accuracy_mean", "Accuracy_std", "Accuracy_formatted",
        ])

        for cond in CONDITION_ORDER:
            s = summary[cond]
            auc_fmt = f"{s['auc_mean']:.4f} ± {s['auc_std']:.4f}"
            acc_fmt = f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}"
            writer.writerow([
                CONDITION_LABELS[cond],
                f"{s['auc_mean']:.4f}",
                f"{s['auc_std']:.4f}",
                auc_fmt,
                f"{s['acc_mean']:.4f}",
                f"{s['acc_std']:.4f}",
                acc_fmt,
            ])

    print(f"Saved to {csv_path}")

    print(f"\n{'Condition':<25} {'Token-Presence AUC':>25} {'Token-Presence Accuracy':>25}")
    print("-" * 75)
    for cond in CONDITION_ORDER:
        s = summary[cond]
        print(f"{CONDITION_LABELS[cond]:<25} "
              f"{s['auc_mean']:.4f} ± {s['auc_std']:.4f}   "
              f"{'':>4}{s['acc_mean']:.4f} ± {s['acc_std']:.4f}")

    per_concept_csv = out_dir / "token_presence_per_concept.csv"
    with open(per_concept_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "Concept", "AUC_mean", "AUC_std", "Accuracy_mean", "Accuracy_std"])
        for cond in CONDITION_ORDER:
            for concept, vals in summary[cond]["per_concept"].items():
                writer.writerow([
                    CONDITION_LABELS[cond],
                    concept,
                    f"{vals['auc_mean']:.4f}",
                    f"{vals['auc_std']:.4f}",
                    f"{vals['acc_mean']:.4f}",
                    f"{vals['acc_std']:.4f}",
                ])
    print(f"\nPer-concept details saved to {per_concept_csv}")


if __name__ == "__main__":
    main()
