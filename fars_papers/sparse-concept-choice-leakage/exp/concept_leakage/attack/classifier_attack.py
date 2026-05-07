"""MLP classifier attack on concept-choice fingerprints.
Trains a lightweight MLP on validation-split fingerprints (Condition A)
and evaluates on test-split fingerprints (Conditions A and C) to test
whether an attacker without exact template knowledge can still identify concepts.
Generates G_train=30 and G_test=50 groups per concept with replacement sampling."""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concept_leakage.noise.mahalanobis import MahalanobisNoiseSampler
from concept_leakage.attack.fingerprint import compute_fingerprints

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
K = len(CONCEPTS)
DIM = 768
EPSILON = 10.0
MASK_SEED = 42
N_RELEASES = 10
G_TRAIN = 30
G_TEST = 50
M = 200
SEEDS = [42, 123, 456]
SMOOTH_LAM = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent
EMB_DIR = BASE_DIR / "outputs" / "embeddings"
CKPT_DIR = BASE_DIR / "checkpoints_opt"
FP_DIR = BASE_DIR / "outputs" / "fingerprints"
RESULTS_DIR = BASE_DIR / "results" / "classifier_attack"


def load_sigma(concept, ckpt_dir=CKPT_DIR, mask_seed=MASK_SEED):
    return np.load(ckpt_dir / concept / f"seed{mask_seed}" / "sigma.npy")


def load_sigma_smoothed(concept, ckpt_dir=CKPT_DIR, mask_seed=MASK_SEED, lam=SMOOTH_LAM):
    fname = f"sigma_smoothed_lam{lam:.2f}.npy"
    return np.load(ckpt_dir / concept / f"seed{mask_seed}" / fname)


def generate_fingerprints(emb_key, n_groups, sigma_loader, seed, fp_subdir):
    rng = np.random.default_rng(seed)
    all_fp = []
    all_labels = []

    for k, concept in enumerate(CONCEPTS):
        emb_path = EMB_DIR / concept / emb_key
        embeddings = np.load(emb_path)
        n_docs = embeddings.shape[0]

        sigma = sigma_loader(concept)
        sampler = MahalanobisNoiseSampler(
            sigma_diag=sigma, epsilon=EPSILON,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        fingerprints, indices = compute_fingerprints(
            embeddings, sampler,
            n_groups=n_groups, group_size=M,
            n_releases=N_RELEASES,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
        )

        out_dir = FP_DIR / fp_subdir / concept
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"fingerprints_seed{seed}.npy", fingerprints)

        print(f"  {concept}: {n_docs} docs -> {fingerprints.shape[0]} groups "
              f"(replacement={'yes' if n_groups * M > n_docs else 'no'})")

        all_fp.append(fingerprints)
        all_labels.append(np.full(fingerprints.shape[0], k, dtype=np.int64))

    X = np.concatenate(all_fp, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def verify_fingerprints(X, y, label=""):
    print(f"\n--- Verification: {label} ---")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    for k, concept in enumerate(CONCEPTS):
        mask = y == k
        print(f"  {concept}: {mask.sum()} fingerprints")

    means = np.zeros((K, DIM))
    for k in range(K):
        means[k] = X[y == k].mean(axis=0)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = means / norms
    cos_sim = normed @ normed.T
    print(f"  Pairwise cosine similarity of concept means:")
    for i in range(K):
        row = " ".join(f"{cos_sim[i, j]:.3f}" for j in range(K))
        print(f"    {CONCEPTS[i]:>10s}: {row}")


class MLP:
    def __init__(self, input_dim=768, hidden=[256, 64], output_dim=5, lr=1e-3):
        self.layers = []
        dims = [input_dim] + hidden + [output_dim]
        rng = np.random.default_rng(0)
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            scale = np.sqrt(2.0 / fan_in)
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float64) * scale
            b = np.zeros(fan_out, dtype=np.float64)
            self.layers.append((W, b))
        self.lr = lr
        self.n_hidden = len(hidden)

    def forward(self, X):
        self.activations = [X]
        h = X
        for i, (W, b) in enumerate(self.layers):
            z = h @ W + b
            if i < self.n_hidden:
                h = np.maximum(0, z)
            else:
                exp_z = np.exp(z - z.max(axis=1, keepdims=True))
                h = exp_z / exp_z.sum(axis=1, keepdims=True)
            self.activations.append(h)
        return h

    def backward(self, X, y_onehot):
        n = X.shape[0]
        probs = self.activations[-1]
        dz = (probs - y_onehot) / n

        grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            W, b = self.layers[i]
            h_prev = self.activations[i]
            dW = h_prev.T @ dz
            db = dz.sum(axis=0)
            grads.insert(0, (dW, db))
            if i > 0:
                dz = dz @ W.T
                dz = dz * (self.activations[i] > 0)

        for i, (dW, db) in enumerate(grads):
            W, b = self.layers[i]
            self.layers[i] = (W - self.lr * dW, b - self.lr * db)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def get_weights(self):
        return [(W.copy(), b.copy()) for W, b in self.layers]

    def set_weights(self, weights):
        self.layers = [(W.copy(), b.copy()) for W, b in weights]


def cross_entropy_loss(probs, y_onehot):
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


def train_mlp_cv(X_train, y_train, n_epochs=200, batch_size=32, lr=1e-3, n_folds=5):
    y_onehot = np.zeros((len(y_train), K), dtype=np.float64)
    y_onehot[np.arange(len(y_train)), y_train] = 1.0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    epoch_val_accs = np.zeros(n_epochs)

    print(f"\n=== 5-Fold Cross-Validation (n={len(y_train)}, epochs={n_epochs}) ===")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y_train[train_idx], y_train[val_idx]
        ytr_oh = y_onehot[train_idx]

        mlp = MLP(input_dim=DIM, hidden=[256, 64], output_dim=K, lr=lr)

        for epoch in range(n_epochs):
            perm = np.random.permutation(len(Xtr))
            for start in range(0, len(Xtr), batch_size):
                idx = perm[start:start + batch_size]
                mlp.forward(Xtr[idx])
                mlp.backward(Xtr[idx], ytr_oh[idx])

            val_preds = mlp.predict(Xval)
            val_acc = accuracy_score(yval, val_preds)
            epoch_val_accs[epoch] += val_acc / n_folds

            if (epoch + 1) % 20 == 0:
                train_probs = mlp.forward(Xtr)
                train_loss = cross_entropy_loss(train_probs, ytr_oh)
                print(f"  Fold {fold_idx+1}, Epoch {epoch+1:3d}: "
                      f"train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    best_epoch = int(np.argmax(epoch_val_accs))
    best_val_acc = epoch_val_accs[best_epoch]
    print(f"\n  Best epoch: {best_epoch+1} (mean val acc={best_val_acc:.4f})")

    print(f"\n=== Retraining on full data for {best_epoch+1} epochs ===")
    final_mlp = MLP(input_dim=DIM, hidden=[256, 64], output_dim=K, lr=lr)
    for epoch in range(best_epoch + 1):
        perm = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = perm[start:start + batch_size]
            final_mlp.forward(X_train[idx])
            final_mlp.backward(X_train[idx], y_onehot[idx])

        if (epoch + 1) % 20 == 0 or epoch == best_epoch:
            train_probs = final_mlp.forward(X_train)
            train_loss = cross_entropy_loss(train_probs, y_onehot)
            train_preds = np.argmax(train_probs, axis=1)
            train_acc = accuracy_score(y_train, train_preds)
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

    train_preds = final_mlp.predict(X_train)
    final_train_acc = accuracy_score(y_train, train_preds)
    print(f"\n  Final training accuracy: {final_train_acc:.4f}")
    assert final_train_acc > 0.3, f"Training accuracy {final_train_acc:.4f} too low, debugging needed"

    return final_mlp, best_epoch + 1, best_val_acc


def evaluate_classifier(mlp, condition_label, fp_subdir, scaler=None):
    print(f"\n=== Evaluating on {condition_label} test fingerprints ({fp_subdir}) ===")
    all_seed_results = []

    for seed in SEEDS:
        all_true = []
        all_pred = []

        for k, concept in enumerate(CONCEPTS):
            fp_path = FP_DIR / fp_subdir / concept / f"fingerprints_seed{seed}.npy"
            fingerprints = np.load(fp_path)
            if scaler is not None:
                fingerprints = scaler.transform(fingerprints.astype(np.float64))
            preds = mlp.predict(fingerprints)
            true_labels = np.full(fingerprints.shape[0], k, dtype=np.int64)
            all_true.append(true_labels)
            all_pred.append(preds)

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  Seed {seed}: N={len(y_true)}, accuracy={acc:.4f}, macro-F1={f1:.4f}")
        all_seed_results.append({"seed": seed, "n_samples": int(len(y_true)),
                                  "accuracy": float(acc), "macro_f1": float(f1)})

    accs = [r["accuracy"] for r in all_seed_results]
    f1s = [r["macro_f1"] for r in all_seed_results]
    summary = {
        "condition": condition_label,
        "per_seed": all_seed_results,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
    }
    print(f"  Mean accuracy: {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    print(f"  Mean macro-F1: {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}")
    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase A: Generate Training Fingerprints (val split, Condition A)")
    print("=" * 60)
    X_train, y_train = generate_fingerprints(
        emb_key="val_dplus.npy", n_groups=G_TRAIN,
        sigma_loader=load_sigma, seed=42,
        fp_subdir="classifier_train_A",
    )
    verify_fingerprints(X_train, y_train, label="Training (val, Condition A)")

    print("\n" + "=" * 60)
    print("Phase B: Generate Test Fingerprints (test split, Conditions A & C)")
    print("=" * 60)
    for seed in SEEDS:
        print(f"\n--- Seed {seed}, Condition A ---")
        generate_fingerprints(
            emb_key="embeddings.npy", n_groups=G_TEST,
            sigma_loader=load_sigma, seed=seed,
            fp_subdir="classifier_test_A",
        )
        print(f"\n--- Seed {seed}, Condition C (lambda={SMOOTH_LAM}) ---")
        generate_fingerprints(
            emb_key="embeddings.npy", n_groups=G_TEST,
            sigma_loader=load_sigma_smoothed, seed=seed,
            fp_subdir="classifier_test_C",
        )

    X_test_A, y_test_A = [], []
    for k, concept in enumerate(CONCEPTS):
        fp = np.load(FP_DIR / "classifier_test_A" / concept / f"fingerprints_seed42.npy")
        X_test_A.append(fp)
        y_test_A.append(np.full(fp.shape[0], k))
    X_test_A = np.concatenate(X_test_A)
    y_test_A = np.concatenate(y_test_A)
    verify_fingerprints(X_test_A, y_test_A, label="Test Condition A (seed=42)")

    X_test_C, y_test_C = [], []
    for k, concept in enumerate(CONCEPTS):
        fp = np.load(FP_DIR / "classifier_test_C" / concept / f"fingerprints_seed42.npy")
        X_test_C.append(fp)
        y_test_C.append(np.full(fp.shape[0], k))
    X_test_C = np.concatenate(X_test_C)
    y_test_C = np.concatenate(y_test_C)
    verify_fingerprints(X_test_C, y_test_C, label="Test Condition C (seed=42)")

    print("\n" + "=" * 60)
    print("Phase C: Train MLP Classifier with 5-Fold CV")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

    mlp, best_epoch, best_val_acc = train_mlp_cv(
        X_train_scaled, y_train, n_epochs=200, batch_size=32, lr=1e-3, n_folds=5,
    )

    print("\n" + "=" * 60)
    print("Phase D: Evaluate Classifier on Test Fingerprints")
    print("=" * 60)

    result_A = evaluate_classifier(mlp, "Condition A (anisotropic)", "classifier_test_A", scaler)
    result_C = evaluate_classifier(mlp, "Condition C (smoothed)", "classifier_test_C", scaler)

    template_A = {"accuracy_mean": 1.0, "accuracy_std": 0.0,
                  "macro_f1_mean": 1.0, "macro_f1_std": 0.0}
    template_C = {"accuracy_mean": 1.0, "accuracy_std": 0.0,
                  "macro_f1_mean": 1.0, "macro_f1_std": 0.0}

    print("\n" + "=" * 60)
    print("Comparison: Classifier vs Template Matching")
    print("=" * 60)
    print(f"  Condition A -- Template:   acc={template_A['accuracy_mean']:.3f}+/-{template_A['accuracy_std']:.3f}, "
          f"F1={template_A['macro_f1_mean']:.3f}+/-{template_A['macro_f1_std']:.3f}")
    print(f"  Condition A -- Classifier: acc={result_A['accuracy_mean']:.3f}+/-{result_A['accuracy_std']:.3f}, "
          f"F1={result_A['macro_f1_mean']:.3f}+/-{result_A['macro_f1_std']:.3f}")
    print(f"  Condition C -- Template:   acc={template_C['accuracy_mean']:.3f}+/-{template_C['accuracy_std']:.3f}, "
          f"F1={template_C['macro_f1_mean']:.3f}+/-{template_C['macro_f1_std']:.3f}")
    print(f"  Condition C -- Classifier: acc={result_C['accuracy_mean']:.3f}+/-{result_C['accuracy_std']:.3f}, "
          f"F1={result_C['macro_f1_mean']:.3f}+/-{result_C['macro_f1_std']:.3f}")

    rows = []
    rows.append({
        "condition": "A (anisotropic)",
        "attacker": "template_matching",
        "accuracy_mean": template_A["accuracy_mean"],
        "accuracy_std": template_A["accuracy_std"],
        "macro_f1_mean": template_A["macro_f1_mean"],
        "macro_f1_std": template_A["macro_f1_std"],
    })
    rows.append({
        "condition": "A (anisotropic)",
        "attacker": "mlp_classifier",
        "accuracy_mean": result_A["accuracy_mean"],
        "accuracy_std": result_A["accuracy_std"],
        "macro_f1_mean": result_A["macro_f1_mean"],
        "macro_f1_std": result_A["macro_f1_std"],
    })
    rows.append({
        "condition": "C (smoothed)",
        "attacker": "template_matching",
        "accuracy_mean": template_C["accuracy_mean"],
        "accuracy_std": template_C["accuracy_std"],
        "macro_f1_mean": template_C["macro_f1_mean"],
        "macro_f1_std": template_C["macro_f1_std"],
    })
    rows.append({
        "condition": "C (smoothed)",
        "attacker": "mlp_classifier",
        "accuracy_mean": result_C["accuracy_mean"],
        "accuracy_std": result_C["accuracy_std"],
        "macro_f1_mean": result_C["macro_f1_mean"],
        "macro_f1_std": result_C["macro_f1_std"],
    })

    csv_path = RESULTS_DIR / "classifier_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {csv_path}")

    full_results = {
        "best_epoch": best_epoch,
        "best_cv_val_acc": float(best_val_acc),
        "condition_A": result_A,
        "condition_C": result_C,
        "template_A": template_A,
        "template_C": template_C,
        "config": {
            "G_train": G_TRAIN, "G_test": G_TEST, "M": M,
            "n_releases": N_RELEASES, "smooth_lambda": SMOOTH_LAM,
            "mlp_hidden": [256, 64], "lr": 1e-3,
            "batch_size": 32, "n_epochs": 200, "n_folds": 5,
            "seeds": SEEDS,
        },
    }
    json_path = RESULTS_DIR / "classifier_results.json"
    with open(json_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Full results saved to {json_path}")


if __name__ == "__main__":
    main()
