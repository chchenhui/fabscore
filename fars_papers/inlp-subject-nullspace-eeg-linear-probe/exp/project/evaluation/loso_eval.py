# Leave-One-Subject-Out (LOSO) evaluation pipeline for frozen CBraMod baseline.
# For each fold: split subjects into train/test, standardize, train logistic
# regression, evaluate balanced accuracy. Logs all metrics to WandB offline.

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import wandb

EMB_DIR = PROJECT_ROOT / "project" / "outputs" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "project" / "results"
SUBJECTS = list(range(1, 10))
SEEDS = [42, 123, 456]
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "inlp-subject-nullspace-eeg-linear-probe")


def load_embeddings(pooling="avgpool", suffix="ea"):
    embeddings = {}
    labels = {}
    for subj in SUBJECTS:
        emb = np.load(EMB_DIR / f"cbramod_{suffix}_{subj}_{pooling}.npy")
        lab = np.load(EMB_DIR / f"labels_{suffix}_{subj}.npy")
        embeddings[subj] = emb
        labels[subj] = lab
    return embeddings, labels


def run_loso_fold(embeddings, labels, test_subject, seed, C=1.0):
    train_X = np.concatenate([embeddings[s] for s in SUBJECTS if s != test_subject])
    train_y = np.concatenate([labels[s] for s in SUBJECTS if s != test_subject])
    test_X = embeddings[test_subject]
    test_y = labels[test_subject]

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    clf = LogisticRegression(
        solver="lbfgs", max_iter=1000,
        C=C, random_state=seed
    )
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    acc = balanced_accuracy_score(test_y, pred)
    return acc


def run_experiment(pooling="avgpool", suffix="ea"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, labels = load_embeddings(pooling=pooling, suffix=suffix)

    all_results = []

    for seed in SEEDS:
        run_name = f"baseline_{suffix}_linear_{pooling}_seed{seed}"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "method": "baseline_ea_linear",
                "pooling": pooling,
                "seed": seed,
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "ea": suffix == "ea",
                "n_subjects": len(SUBJECTS),
                "n_classes": 4,
                "embedding_dim": embeddings[1].shape[1],
            },
            tags=["baseline", "ea", pooling],
        )

        fold_accs = []
        for test_subj in SUBJECTS:
            acc = run_loso_fold(embeddings, labels, test_subj, seed)
            fold_accs.append(acc)
            wandb.log({f"fold_{test_subj}_acc": acc})
            all_results.append({
                "pooling": pooling,
                "seed": seed,
                "test_subject": test_subj,
                "balanced_accuracy": acc,
            })

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        wandb.log({
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "vanilla_baseline_reported": 0.4145,
            "delta_vs_vanilla": mean_acc - 0.4145,
        })
        wandb.finish()

        print(f"[{run_name}] Mean={mean_acc:.4f} +/- {std_acc:.4f}  "
              f"Per-fold: {[f'{a:.4f}' for a in fold_accs]}")

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"baseline_ea_linear_{pooling}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")
    return df


def run_all_poolings():
    results = {}
    for pooling in ["avgpool", "flatten"]:
        print(f"\n{'='*60}")
        print(f"Running LOSO with pooling={pooling}")
        print(f"{'='*60}")
        df = run_experiment(pooling=pooling, suffix="ea")

        mean_per_seed = df.groupby("seed")["balanced_accuracy"].mean()
        overall_mean = mean_per_seed.mean()
        overall_std = mean_per_seed.std()
        results[pooling] = {"mean": overall_mean, "std": overall_std, "df": df}
        print(f"\n{pooling}: Overall mean={overall_mean:.4f} +/- {overall_std:.4f}")

    print(f"\n{'='*60}")
    print("Summary (mean over seeds):")
    for pooling, r in results.items():
        print(f"  {pooling}: {r['mean']:.4f} +/- {r['std']:.4f}")

    best_pooling = max(results, key=lambda k: results[k]["mean"])
    print(f"\nBest pooling: {best_pooling} ({results[best_pooling]['mean']:.4f})")

    combined_df = pd.concat([r["df"] for r in results.values()])
    combined_csv = RESULTS_DIR / "baseline_ea_linear.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"Combined results saved to {combined_csv}")

    summary = {
        "best_pooling": best_pooling,
        "results": {
            p: {"mean": float(r["mean"]), "std": float(r["std"])}
            for p, r in results.items()
        },
        "vanilla_baseline_reported": 0.4145,
    }
    summary_path = RESULTS_DIR / "baseline_ea_linear_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    return results


if __name__ == "__main__":
    run_all_poolings()
