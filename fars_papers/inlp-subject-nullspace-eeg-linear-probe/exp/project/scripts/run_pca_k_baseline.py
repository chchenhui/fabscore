# PCA-k removal LOSO evaluation sweep.
# For each k in [1,2,3,5,7,10], removes top k PCA components from
# flatten CBraMod embeddings and evaluates 4-class MI via logistic regression.
# PCA is fit per-fold on training subjects only.

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "project"))
load_dotenv(PROJECT_ROOT / ".env")

import wandb
from methods.pca_removal import pca_k_removal

EMB_DIR = PROJECT_ROOT / "project" / "outputs" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "project" / "results"
SUBJECTS = list(range(1, 10))
SEEDS = [42, 123, 456]
K_VALUES = [1, 2, 3, 5, 7, 10]
POOLING = "flatten"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "inlp-subject-nullspace-eeg-linear-probe")


def load_embeddings():
    embeddings, labels = {}, {}
    for subj in SUBJECTS:
        embeddings[subj] = np.load(EMB_DIR / f"cbramod_ea_{subj}_{POOLING}.npy")
        labels[subj] = np.load(EMB_DIR / f"labels_ea_{subj}.npy")
    return embeddings, labels


def run_single_fold(embeddings, labels, test_subject, k, seed):
    train_X = np.concatenate([embeddings[s] for s in SUBJECTS if s != test_subject])
    train_y = np.concatenate([labels[s] for s in SUBJECTS if s != test_subject])
    test_X = embeddings[test_subject]
    test_y = labels[test_subject]

    train_X_proj, test_X_proj, _ = pca_k_removal(train_X, test_X, k)

    scaler = StandardScaler()
    train_X_proj = scaler.fit_transform(train_X_proj)
    test_X_proj = scaler.transform(test_X_proj)

    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=seed)
    clf.fit(train_X_proj, train_y)
    pred = clf.predict(test_X_proj)
    return balanced_accuracy_score(test_y, pred)


def run_verification():
    print("=" * 60)
    print("VERIFICATION DRY-RUN: k=1, seed=42, test_subject=1")
    print("=" * 60)

    embeddings, labels = load_embeddings()

    train_X = np.concatenate([embeddings[s] for s in SUBJECTS if s != 1])
    test_X = embeddings[1]

    train_proj, test_proj, V_k = pca_k_removal(train_X, test_X, k=1)

    assert train_proj.shape == train_X.shape, (
        f"Shape mismatch: {train_proj.shape} vs {train_X.shape}")
    assert test_proj.shape == test_X.shape, (
        f"Shape mismatch: {test_proj.shape} vs {test_X.shape}")
    print(f"  [PASS] Projected shapes: train={train_proj.shape}, test={test_proj.shape}")

    reproject = train_proj - (train_proj @ V_k.T) @ V_k
    assert np.allclose(reproject, train_proj, atol=1e-6), "Idempotency check failed"
    print("  [PASS] Projection is idempotent (re-projecting gives same result)")

    acc = run_single_fold(embeddings, labels, test_subject=1, k=1, seed=42)
    assert 0.0 <= acc <= 1.0, f"Invalid accuracy: {acc}"
    print(f"  [PASS] Balanced accuracy = {acc:.4f}")
    print("VERIFICATION PASSED\n")
    return True


def run_full_sweep():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, labels = load_embeddings()
    all_results = []
    t0 = time.time()

    for k in K_VALUES:
        for seed in SEEDS:
            run_name = f"pca_k{k}_seed{seed}"
            wandb.init(
                project=WANDB_PROJECT,
                name=run_name,
                mode="offline",
                config={
                    "method": "pca_k_removal",
                    "k": k,
                    "seed": seed,
                    "pooling": POOLING,
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "n_subjects": len(SUBJECTS),
                    "n_classes": 4,
                    "embedding_dim": embeddings[1].shape[1],
                },
                tags=["pca_k_removal", POOLING, f"k{k}"],
            )

            fold_accs = []
            for test_subj in SUBJECTS:
                acc = run_single_fold(embeddings, labels, test_subj, k, seed)
                fold_accs.append(acc)
                wandb.log({f"fold_{test_subj}_acc": acc})
                all_results.append({
                    "k": k,
                    "seed": seed,
                    "test_subject": test_subj,
                    "balanced_accuracy": acc,
                })

            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            wandb.log({"mean_acc": mean_acc, "std_acc": std_acc})
            wandb.finish()

            elapsed = time.time() - t0
            print(f"[{run_name}] Mean={mean_acc:.4f} +/- {std_acc:.4f}  "
                  f"({elapsed:.0f}s elapsed)")

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "baseline_pca_k.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    print("\n" + "=" * 60)
    print("SUMMARY (mean over 9 folds, per seed)")
    print("=" * 60)
    for k in K_VALUES:
        sub = df[df["k"] == k]
        per_seed_means = sub.groupby("seed")["balanced_accuracy"].mean()
        overall_mean = per_seed_means.mean()
        overall_std = per_seed_means.std()
        print(f"  k={k:>2d}: {overall_mean:.4f} +/- {overall_std:.4f}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s")
    return df


if __name__ == "__main__":
    if "--verify-only" in sys.argv:
        run_verification()
    else:
        run_verification()
        run_full_sweep()
