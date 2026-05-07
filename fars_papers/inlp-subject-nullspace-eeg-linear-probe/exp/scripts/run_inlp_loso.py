# Full LOSO evaluation with INLP subject-identity removal on frozen CBraMod embeddings.
# Runs 9 folds x 3 seeds = 27 evaluations. Each fold: INLP on train subjects,
# project train+test, standardize, train logistic regression, evaluate balanced accuracy.
# Outputs: project/results/main_inlp.csv + WandB offline logging.

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
from project.methods.inlp import run_inlp, apply_projection

EMB_DIR = PROJECT_ROOT / "project" / "outputs" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "project" / "results"
SUBJECTS = list(range(1, 10))
SEEDS = [42, 123, 456]
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "inlp-subject-nullspace-eeg-linear-probe")


def load_embeddings(pooling="flatten", suffix="ea"):
    embeddings, labels = {}, {}
    for subj in SUBJECTS:
        embeddings[subj] = np.load(EMB_DIR / f"cbramod_{suffix}_{subj}_{pooling}.npy")
        labels[subj] = np.load(EMB_DIR / f"labels_{suffix}_{subj}.npy")
    return embeddings, labels


def run_single_fold(embeddings, labels, test_subject, seed):
    train_X = np.concatenate([embeddings[s] for s in SUBJECTS if s != test_subject])
    train_y = np.concatenate([labels[s] for s in SUBJECTS if s != test_subject])
    train_subj = np.concatenate([
        np.full(len(labels[s]), s) for s in SUBJECTS if s != test_subject
    ])
    test_X = embeddings[test_subject]
    test_y = labels[test_subject]

    t0 = time.time()
    inlp_result = run_inlp(
        train_X, train_subj,
        max_iter=10, early_stop_threshold=1.25, seed=seed
    )
    inlp_time = time.time() - t0

    V = inlp_result["V"]
    train_proj = apply_projection(train_X, V)
    test_proj = apply_projection(test_X, V)

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_proj)
    test_s = scaler.transform(test_proj)

    clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
    clf.fit(train_s, train_y)
    pred = clf.predict(test_s)
    acc = balanced_accuracy_score(test_y, pred)

    return {
        "seed": seed,
        "test_subject": test_subject,
        "balanced_accuracy": float(acc),
        "inlp_iterations": inlp_result["num_iterations"],
        "rank_removed": inlp_result["rank_removed"],
        "subject_id_acc_pre": inlp_result["pre_inlp_subject_acc"],
        "subject_id_acc_post": inlp_result["post_inlp_subject_acc"],
        "chance_level": inlp_result["chance_level"],
        "inlp_time_s": float(inlp_time),
        "iter_log": inlp_result["iter_log"],
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings, labels = load_embeddings(pooling="flatten", suffix="ea")
    all_results = []

    for seed in SEEDS:
        run_name = f"inlp_ea_flatten_seed{seed}"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            mode="offline",
            config={
                "method": "inlp_ea_linear",
                "pooling": "flatten",
                "seed": seed,
                "max_iter": 10,
                "early_stop_threshold": 1.25,
                "solver_inlp": "sgd_log_loss",
                "solver_task": "lbfgs",
                "n_subjects": len(SUBJECTS),
                "n_classes": 4,
                "embedding_dim": embeddings[1].shape[1],
            },
            tags=["inlp", "ea", "flatten", "main"],
        )

        fold_accs = []
        for test_subj in SUBJECTS:
            print(f"[seed={seed}] Fold {test_subj}/9 ...", flush=True)
            result = run_single_fold(embeddings, labels, test_subj, seed)
            fold_accs.append(result["balanced_accuracy"])

            wandb.log({
                f"fold_{test_subj}_acc": result["balanced_accuracy"],
                f"fold_{test_subj}_inlp_iters": result["inlp_iterations"],
                f"fold_{test_subj}_rank_removed": result["rank_removed"],
                f"fold_{test_subj}_subj_acc_pre": result["subject_id_acc_pre"],
                f"fold_{test_subj}_subj_acc_post": result["subject_id_acc_post"],
            })

            row = {k: v for k, v in result.items() if k != "iter_log"}
            all_results.append(row)

            print(f"  acc={result['balanced_accuracy']:.4f}, "
                  f"INLP iters={result['inlp_iterations']}, "
                  f"rank={result['rank_removed']}, "
                  f"subj_acc: {result['subject_id_acc_pre']:.4f}->{result['subject_id_acc_post']:.4f}, "
                  f"time={result['inlp_time_s']:.0f}s", flush=True)

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
    csv_path = RESULTS_DIR / "main_inlp.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    mean_per_seed = df.groupby("seed")["balanced_accuracy"].mean()
    overall_mean = mean_per_seed.mean()
    overall_std = mean_per_seed.std()
    print(f"Overall: {overall_mean:.4f} +/- {overall_std:.4f}")
    print(f"EA baseline: 0.5627")
    print(f"Delta: {(overall_mean - 0.5627)*100:+.2f} pp")

    return df


if __name__ == "__main__":
    main()
