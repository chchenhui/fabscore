"""Token-presence probe: binary MLP that predicts whether a concept token
is present in the original sentence from a single sanitized embedding.
Evaluates under Clean, Isotropic (B), Anisotropic (A), Smoothed (C) conditions."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from concept_leakage.noise.isotropic import IsotropicNoiseSampler
from concept_leakage.noise.mahalanobis import MahalanobisNoiseSampler

CONCEPTS = ["weekdays", "months", "countries", "gender", "cities"]
DIM = 768
EPSILON = 10.0
MASK_SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
EMB_DIR = BASE_DIR / "outputs" / "embeddings"
CKPT_DIR = BASE_DIR / "checkpoints_opt"
RESULTS_DIR = BASE_DIR / "results" / "token_presence"


class TokenPresenceProbe(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=(256, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def load_embeddings(concept, split):
    dplus = np.load(EMB_DIR / concept / f"{split}_dplus.npy")
    dminus = np.load(EMB_DIR / concept / f"{split}_dminus.npy")
    return dplus, dminus


def add_noise(embeddings, condition, concept, rng_seed):
    if condition == "clean":
        return embeddings.copy()

    rng = np.random.default_rng(rng_seed)

    if condition == "isotropic":
        sampler = IsotropicNoiseSampler(dim=DIM, epsilon=EPSILON, rng=rng)
    elif condition == "anisotropic":
        sigma = np.load(CKPT_DIR / concept / f"seed{MASK_SEED}" / "sigma.npy")
        sampler = MahalanobisNoiseSampler(sigma_diag=sigma, epsilon=EPSILON, rng=rng)
    elif condition == "smoothed":
        sigma = np.load(CKPT_DIR / concept / f"seed{MASK_SEED}" / "sigma_smoothed_lam0.20.npy")
        sampler = MahalanobisNoiseSampler(sigma_diag=sigma, epsilon=EPSILON, rng=rng)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    noise = sampler.sample(len(embeddings))
    return embeddings + noise


def prepare_data(concept, split, condition, rng_seed):
    dplus, dminus = load_embeddings(concept, split)
    dplus_noisy = add_noise(dplus, condition, concept, rng_seed)
    dminus_noisy = add_noise(dminus, condition, concept, rng_seed + 1000)

    X = np.concatenate([dplus_noisy, dminus_noisy], axis=0)
    y = np.concatenate([np.ones(len(dplus_noisy)), np.zeros(len(dminus_noisy))])
    return X, y


def train_probe(concept, condition, seed, n_epochs=50, batch_size=128, lr=1e-3,
                use_wandb=False, wandb_run=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train = prepare_data(concept, "train", condition, rng_seed=seed)
    X_val, y_val = prepare_data(concept, "val", condition, rng_seed=seed + 500)
    X_test, y_test = prepare_data(concept, "test", condition, rng_seed=seed + 999)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = TokenPresenceProbe(input_dim=DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_auc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        X_shuf = X_train_t[perm]
        y_shuf = y_train_t[perm]

        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_shuf), batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).numpy()
            val_auc = roc_auc_score(y_val, val_pred)
            val_acc = accuracy_score(y_val, (val_pred >= 0.5).astype(int))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if use_wandb and wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_auc": val_auc,
                "val_acc": val_acc,
                "best_val_auc": best_val_auc,
            })

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy()
        test_auc = roc_auc_score(y_test, test_pred)
        test_acc = accuracy_score(y_test, (test_pred >= 0.5).astype(int))

    print(f"  Best epoch: {best_epoch}, test_auc={test_auc:.4f}, test_acc={test_acc:.4f}")

    return {
        "concept": concept,
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_auc": float(test_auc),
        "test_acc": float(test_acc),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["clean", "isotropic", "anisotropic", "smoothed"])
    parser.add_argument("--concepts", nargs="+", default=CONCEPTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            import wandb
        except ImportError:
            print("WARNING: wandb or dotenv not available, disabling wandb")
            use_wandb = False

    all_results = {}

    for condition in args.conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {condition}")
        print(f"{'='*60}")

        condition_results = []

        for concept in args.concepts:
            for seed in args.seeds:
                print(f"\n--- {condition} / {concept} / seed={seed} ---")

                wandb_run = None
                if use_wandb:
                    import wandb
                    wandb_run = wandb.init(
                        project=os.environ.get("WANDB_PROJECT", "sparse-concept-choice-leakage"),
                        name=f"token_presence/{condition}/{concept}/seed{seed}",
                        config={
                            "condition": condition,
                            "concept": concept,
                            "seed": seed,
                            "n_epochs": args.n_epochs,
                            "batch_size": args.batch_size,
                            "lr": args.lr,
                            "epsilon": EPSILON,
                            "task": "token_presence",
                        },
                        reinit=True,
                    )

                result = train_probe(
                    concept=concept,
                    condition=condition,
                    seed=seed,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    use_wandb=use_wandb,
                    wandb_run=wandb_run,
                )
                condition_results.append(result)

                if wandb_run is not None:
                    wandb_run.summary["test_auc"] = result["test_auc"]
                    wandb_run.summary["test_acc"] = result["test_acc"]
                    wandb_run.summary["best_epoch"] = result["best_epoch"]
                    wandb_run.finish()

        all_results[condition] = condition_results

        aucs = [r["test_auc"] for r in condition_results]
        accs = [r["test_acc"] for r in condition_results]
        print(f"\n=== {condition} summary ===")
        print(f"  AUC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
        print(f"  Acc:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        cond_path = results_dir / f"{condition}_results.json"
        with open(cond_path, "w") as f:
            json.dump(condition_results, f, indent=2)
        print(f"  Saved to {cond_path}")

    summary = {}
    for condition, results in all_results.items():
        aucs = [r["test_auc"] for r in results]
        accs = [r["test_acc"] for r in results]

        per_concept = {}
        for concept in args.concepts:
            cr = [r for r in results if r["concept"] == concept]
            per_concept[concept] = {
                "auc_mean": float(np.mean([r["test_auc"] for r in cr])),
                "auc_std": float(np.std([r["test_auc"] for r in cr])),
                "acc_mean": float(np.mean([r["test_acc"] for r in cr])),
                "acc_std": float(np.std([r["test_acc"] for r in cr])),
            }

        summary[condition] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "per_concept": per_concept,
        }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'AUC':>20} {'Accuracy':>20}")
    print("-" * 55)
    for condition in args.conditions:
        s = summary[condition]
        print(f"{condition:<15} {s['auc_mean']:.4f} +/- {s['auc_std']:.4f}   "
              f"{s['acc_mean']:.4f} +/- {s['acc_std']:.4f}")

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
