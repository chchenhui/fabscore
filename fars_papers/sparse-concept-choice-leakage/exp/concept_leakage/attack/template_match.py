"""Template-matching concept prediction from group fingerprints.
For isotropic noise, all K=5 templates are identical (uniform 1/d), so prediction
is effectively random. Handles variable group counts per concept."""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
K = len(CONCEPTS)
DIM = 768
SEEDS = [42, 123, 456]

BASE_DIR = Path(__file__).resolve().parent.parent
FP_DIR = BASE_DIR / "outputs" / "fingerprints"
RESULTS_DIR = BASE_DIR / "results"


def build_isotropic_templates() -> np.ndarray:
    templates = np.full((K, DIM), 1.0 / DIM, dtype=np.float32)
    return templates


def predict_concepts(
    fingerprints: np.ndarray,
    templates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = fingerprints.shape[0]
    predictions = np.zeros(n, dtype=np.int64)

    for i in range(n):
        dists = np.linalg.norm(fingerprints[i] - templates, axis=1)
        min_dist = dists.min()
        tied = np.where(np.abs(dists - min_dist) < 1e-10)[0]
        predictions[i] = rng.choice(tied)

    return predictions


def run_template_matching(noise_type: str = "isotropic"):
    templates = build_isotropic_templates()

    all_seed_results = []

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        all_true = []
        all_pred = []

        for k, concept in enumerate(CONCEPTS):
            fp_path = FP_DIR / noise_type / concept / f"fingerprints_seed{seed}.npy"
            if not fp_path.exists():
                print(f"WARNING: {fp_path} not found, skipping")
                continue

            fingerprints = np.load(fp_path)
            n_groups = fingerprints.shape[0]

            preds = predict_concepts(fingerprints, templates, rng)
            true_labels = np.full(n_groups, k, dtype=np.int64)

            all_true.append(true_labels)
            all_pred.append(preds)

        if not all_true:
            print(f"No fingerprints found for seed {seed}")
            continue

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print(f"Seed {seed}: N={len(y_true)}, accuracy={acc:.4f}, macro-F1={f1:.4f}")
        all_seed_results.append({"seed": seed, "n_samples": int(len(y_true)),
                                  "accuracy": acc, "macro_f1": f1})

    accs = [r["accuracy"] for r in all_seed_results]
    f1s = [r["macro_f1"] for r in all_seed_results]

    summary = {
        "noise_type": noise_type,
        "n_concepts": K,
        "seeds": SEEDS,
        "per_seed": all_seed_results,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "chance_level": 1.0 / K,
    }

    print(f"\n=== Summary ({noise_type}) ===")
    print(f"  Accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Macro-F1: {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    print(f"  Chance:   {summary['chance_level']:.4f}")

    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_template_matching(noise_type="isotropic")

    out_path = RESULTS_DIR / "isotropic_attack_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
