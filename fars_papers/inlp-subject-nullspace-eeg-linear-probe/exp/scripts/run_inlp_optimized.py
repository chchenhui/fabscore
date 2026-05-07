# Optimized INLP LOSO evaluation: sweeps INLP iteration count and task-head
# regularization C via inner cross-validation. For each outer LOSO fold, runs
# progressive INLP once, then evaluates all (n_iter, C) combos. Selects best
# config per fold using inner-LOSO on the 8 training subjects, then evaluates
# on held-out test subject with selected config.

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import wandb

sys.path.insert(0, str(PROJECT_ROOT))
from project.methods.inlp import run_inlp_progressive, apply_projection

EMB_DIR = PROJECT_ROOT / "project" / "outputs" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "project" / "results"
SUBJECTS = list(range(1, 10))
SEEDS = [42, 123, 456]
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "inlp-subject-nullspace-eeg-linear-probe")

ITER_GRID = [1, 2, 3, 4, 5, 7, 10]
C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_embeddings(pooling="flatten", suffix="ea"):
    embeddings, labels = {}, {}
    for subj in SUBJECTS:
        embeddings[subj] = np.load(EMB_DIR / f"cbramod_{suffix}_{subj}_{pooling}.npy")
        labels[subj] = np.load(EMB_DIR / f"labels_{suffix}_{subj}.npy")
    return embeddings, labels


def evaluate_config(train_X_proj, train_y, test_X_proj, test_y, C, seed):
    scaler = StandardScaler()
    tr = scaler.fit_transform(train_X_proj)
    te = scaler.transform(test_X_proj)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=C, random_state=seed)
    clf.fit(tr, train_y)
    pred = clf.predict(te)
    return balanced_accuracy_score(test_y, pred)


def inner_cv_select(embeddings, labels, train_subjects, V_per_iter, seed):
    best_score = -1
    best_config = (1, 1.0)

    config_scores = {}
    for n_iter in ITER_GRID:
        if n_iter not in V_per_iter:
            continue
        V = V_per_iter[n_iter]

        for C in C_GRID:
            fold_accs = []
            for val_subj in train_subjects:
                inner_train_subjs = [s for s in train_subjects if s != val_subj]
                inner_train_X = np.concatenate([embeddings[s] for s in inner_train_subjs])
                inner_train_y = np.concatenate([labels[s] for s in inner_train_subjs])
                val_X = embeddings[val_subj]
                val_y = labels[val_subj]

                inner_train_proj = apply_projection(inner_train_X, V)
                val_proj = apply_projection(val_X, V)

                acc = evaluate_config(inner_train_proj, inner_train_y, val_proj, val_y, C, seed)
                fold_accs.append(acc)

            mean_acc = np.mean(fold_accs)
            config_scores[(n_iter, C)] = mean_acc

            if mean_acc > best_score:
                best_score = mean_acc
                best_config = (n_iter, C)

    return best_config, best_score, config_scores


def run_single_fold(embeddings, labels, test_subject, seed):
    train_subjects = [s for s in SUBJECTS if s != test_subject]
    train_X = np.concatenate([embeddings[s] for s in train_subjects])
    train_y = np.concatenate([labels[s] for s in train_subjects])
    train_subj_ids = np.concatenate([
        np.full(len(labels[s]), s) for s in train_subjects
    ])
    test_X = embeddings[test_subject]
    test_y = labels[test_subject]

    t0 = time.time()
    inlp_result = run_inlp_progressive(
        train_X, train_subj_ids, max_iter=max(ITER_GRID), seed=seed
    )
    inlp_time = time.time() - t0

    V_per_iter = inlp_result["V_per_iter"]

    t1 = time.time()
    best_config, best_inner_score, config_scores = inner_cv_select(
        embeddings, labels, train_subjects, V_per_iter, seed
    )
    cv_time = time.time() - t1

    best_n_iter, best_C = best_config
    V_best = V_per_iter[best_n_iter]

    train_proj = apply_projection(train_X, V_best)
    test_proj = apply_projection(test_X, V_best)
    test_acc = evaluate_config(train_proj, train_y, test_proj, test_y, best_C, seed)

    all_configs_test = {}
    for n_iter in ITER_GRID:
        if n_iter not in V_per_iter:
            continue
        V = V_per_iter[n_iter]
        tr_p = apply_projection(train_X, V)
        te_p = apply_projection(test_X, V)
        for C in C_GRID:
            acc = evaluate_config(tr_p, train_y, te_p, test_y, C, seed)
            all_configs_test[(n_iter, C)] = acc

    return {
        "seed": seed,
        "test_subject": test_subject,
        "balanced_accuracy": float(test_acc),
        "best_n_iter": best_n_iter,
        "best_C": best_C,
        "best_inner_score": float(best_inner_score),
        "rank_removed": int(V_best.shape[1]),
        "pre_inlp_subject_acc": inlp_result["pre_inlp_subject_acc"],
        "inlp_time_s": float(inlp_time),
        "cv_time_s": float(cv_time),
        "iter_log": inlp_result["iter_log"],
        "config_scores_inner": {f"{k[0]}_{k[1]}": float(v) for k, v in config_scores.items()},
        "config_scores_test": {f"{k[0]}_{k[1]}": float(v) for k, v in all_configs_test.items()},
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, labels = load_embeddings(pooling="flatten", suffix="ea")
    all_results = []

    for seed in SEEDS:
        run_name = f"inlp_optimized_seed{seed}"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            mode="offline",
            config={
                "method": "inlp_optimized",
                "pooling": "flatten",
                "seed": seed,
                "max_iter": max(ITER_GRID),
                "iter_grid": ITER_GRID,
                "C_grid": C_GRID,
                "n_subjects": len(SUBJECTS),
                "n_classes": 4,
                "embedding_dim": embeddings[1].shape[1],
            },
            tags=["inlp", "optimized", "ea", "flatten"],
        )

        fold_accs = []
        for test_subj in SUBJECTS:
            print(f"[seed={seed}] Fold {test_subj}/9 ...", flush=True)
            result = run_single_fold(embeddings, labels, test_subj, seed)
            fold_accs.append(result["balanced_accuracy"])

            wandb.log({
                f"fold_{test_subj}_acc": result["balanced_accuracy"],
                f"fold_{test_subj}_best_iter": result["best_n_iter"],
                f"fold_{test_subj}_best_C": result["best_C"],
                f"fold_{test_subj}_inner_score": result["best_inner_score"],
                f"fold_{test_subj}_rank_removed": result["rank_removed"],
            })

            row = {k: v for k, v in result.items()
                   if k not in ("iter_log", "config_scores_inner", "config_scores_test")}
            all_results.append(row)

            print(f"  acc={result['balanced_accuracy']:.4f}, "
                  f"best_iter={result['best_n_iter']}, best_C={result['best_C']}, "
                  f"inner={result['best_inner_score']:.4f}, "
                  f"rank={result['rank_removed']}, "
                  f"inlp_time={result['inlp_time_s']:.0f}s, "
                  f"cv_time={result['cv_time_s']:.0f}s", flush=True)

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        wandb.log({
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "ea_baseline": 0.5627,
            "delta_vs_baseline": mean_acc - 0.5627,
        })
        wandb.finish()

        print(f"[seed={seed}] Mean={mean_acc:.4f} +/- {std_acc:.4f}\n", flush=True)

    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "inlp_optimized.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    mean_per_seed = df.groupby("seed")["balanced_accuracy"].mean()
    overall_mean = mean_per_seed.mean()
    overall_std = mean_per_seed.std()
    print(f"Overall: {overall_mean:.4f} +/- {overall_std:.4f}")
    print(f"EA baseline: 0.5627")
    print(f"Previous INLP: 0.5518")
    print(f"Delta vs baseline: {(overall_mean - 0.5627)*100:+.2f} pp")
    print(f"Delta vs prev INLP: {(overall_mean - 0.5518)*100:+.2f} pp")

    print(f"\nMost frequent selected configs per seed:")
    for seed in SEEDS:
        subset = df[df["seed"] == seed]
        counts = subset.groupby(["best_n_iter", "best_C"]).size().sort_values(ascending=False)
        print(f"  seed={seed}: {counts.to_dict()}")

    return df


if __name__ == "__main__":
    main()
