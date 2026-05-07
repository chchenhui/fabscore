"""Shuffled-pair null control adapter training and evaluation.

Trains adapters on Wikipedia embeddings with randomly permuted targets,
breaking source-target correspondence. This verifies that public-anchor
adapter gains come from meaningful alignment rather than regularization.

Usage:
  python -m pada.scripts.run_shuffled_pair              # full run
  python -m pada.scripts.run_shuffled_pair --dry-run    # 1 seed, 1 dataset, 5 epochs
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pada.adapters.residual_mlp import ResidualMLPAdapter
from pada.data.beir_loader import DATASETS, load_dataset
from pada.evaluation.retrieval_eval import evaluate_retrieval
from pada.trainers.adapter_trainer import train_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_DIR = PROJECT_ROOT / "pada" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "pada" / "outputs" / "shuffled_pair"
RESULTS_DIR = PROJECT_ROOT / "pada" / "results"

N_P = 5000
SEEDS = [0, 1, 2]
SHUFFLE_SEED_OFFSET = 1000


def load_wikipedia_embeddings(seed: int):
    wiki_dir = EMBED_DIR / "wikipedia" / f"seed_{seed}"
    source_fnew = np.load(str(wiki_dir / "source_fnew.npy"))
    target_fold = np.load(str(wiki_dir / "target_fold.npy"))
    return source_fnew, target_fold


def load_dataset_embeddings(dataset_name: str):
    ds_dir = EMBED_DIR / dataset_name
    queries_fnew = np.load(str(ds_dir / "queries_fnew.npy"))
    corpus_fold = np.load(str(ds_dir / "corpus_fold.npy"))
    with open(ds_dir / "corpus_ids.json") as f:
        corpus_ids = json.load(f)
    with open(ds_dir / "query_ids.json") as f:
        query_ids = json.load(f)
    return queries_fnew, corpus_fold, corpus_ids, query_ids


def adapt_queries(model: ResidualMLPAdapter, queries_fnew: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        q_tensor = torch.tensor(queries_fnew, dtype=torch.float32).to(device)
        adapted = model(q_tensor)
    return adapted.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, 1 dataset, 5 epochs")
    args = parser.parse_args()

    seeds = [0] if args.dry_run else SEEDS
    datasets_to_eval = ["scifact"] if args.dry_run else DATASETS
    max_epochs = 5 if args.dry_run else 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL'}")
    logger.info(f"Seeds: {seeds}, Datasets: {datasets_to_eval}, max_epochs={max_epochs}")

    import wandb
    run_name = "shuffled-pair-adapter"
    if args.dry_run:
        run_name += "-dryrun"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "public-anchor-drift-adapter"),
        name=run_name,
        config={
            "task": "shuffled-pair-adapter",
            "n_p": N_P,
            "seeds": seeds,
            "datasets": datasets_to_eval,
            "max_epochs": max_epochs,
            "lr": 3e-4,
            "weight_decay": 0.01,
            "batch_size": 256,
            "patience": 5,
            "embed_dim": 768,
            "hidden_dim": 256,
            "dry_run": args.dry_run,
            "data_source": "wikipedia_shuffled",
            "shuffle_seed_offset": SHUFFLE_SEED_OFFSET,
        },
    )

    all_results = {}
    per_seed_all = {}

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*60}")

        source_fnew, target_fold = load_wikipedia_embeddings(seed)
        logger.info(f"Loaded Wikipedia embeddings: source={source_fnew.shape}, target={target_fold.shape}")

        shuffle_rng = np.random.RandomState(seed + SHUFFLE_SEED_OFFSET)
        perm = shuffle_rng.permutation(len(target_fold))
        target_fold_shuffled = target_fold[perm]
        logger.info(f"Shuffled target embeddings with permutation seed={seed + SHUFFLE_SEED_OFFSET}")

        save_dir = OUTPUT_DIR / f"seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(save_dir / "permutation.npy"), perm)

        logger.info(f"Training shuffled-pair adapter (seed={seed})...")
        t0 = time.time()
        result = train_adapter(
            source_embeds=source_fnew,
            target_embeds=target_fold_shuffled,
            save_dir=str(save_dir),
            embed_dim=768,
            hidden_dim=256,
            lr=3e-4,
            weight_decay=0.01,
            batch_size=256,
            max_epochs=max_epochs,
            patience=5,
            seed=seed,
            device=device,
        )
        train_time = time.time() - t0
        logger.info(f"Training done in {train_time:.1f}s, "
                    f"epochs={result['epochs_trained']}, "
                    f"best_val_loss={result['best_val_loss']:.6f}")

        with open(save_dir / "training_history.json", "w") as f:
            json.dump(result["history"], f, indent=2)

        for epoch_i, (tl, vl) in enumerate(
            zip(result["history"]["train_loss"], result["history"]["val_loss"])
        ):
            wandb.log({
                f"train/seed_{seed}/train_loss": tl,
                f"train/seed_{seed}/val_loss": vl,
                f"train/seed_{seed}/epoch": epoch_i + 1,
            })

        model = result["model"]
        model.to(device)
        per_seed_all[f"seed_{seed}"] = {}

        for ds_name in datasets_to_eval:
            queries_fnew, corpus_fold_ds, corpus_ids, query_ids = load_dataset_embeddings(ds_name)
            _, queries, qrels = load_dataset(ds_name)

            adapted_queries = adapt_queries(model, queries_fnew, device)

            metrics = evaluate_retrieval(
                adapted_queries, corpus_fold_ds, query_ids, corpus_ids, qrels
            )
            logger.info(f"  {ds_name}: nDCG@10={metrics['NDCG@10']:.5f}, "
                        f"Recall@10={metrics['Recall@10']:.5f}")

            wandb.log({
                f"eval/{ds_name}/seed_{seed}/nDCG@10": metrics["NDCG@10"],
                f"eval/{ds_name}/seed_{seed}/Recall@10": metrics["Recall@10"],
            })

            per_seed_all[f"seed_{seed}"][ds_name] = {
                "nDCG@10": round(metrics["NDCG@10"], 5),
                "Recall@10": round(metrics["Recall@10"], 5),
            }

        wandb.log({
            f"train/seed_{seed}/best_val_loss": result["best_val_loss"],
            f"train/seed_{seed}/epochs_trained": result["epochs_trained"],
            f"train/seed_{seed}/train_time_s": train_time,
            f"train/seed_{seed}/n_train_pairs": len(source_fnew),
        })

    for ds_name in datasets_to_eval:
        seed_ndcg = [per_seed_all[f"seed_{s}"][ds_name]["nDCG@10"] for s in seeds]
        seed_recall = [per_seed_all[f"seed_{s}"][ds_name]["Recall@10"] for s in seeds]
        per_seed_dict = {}
        for s in seeds:
            per_seed_dict[f"seed_{s}"] = per_seed_all[f"seed_{s}"][ds_name]
        all_results[ds_name] = {
            "nDCG@10_mean": round(float(np.mean(seed_ndcg)), 5),
            "nDCG@10_std": round(float(np.std(seed_ndcg)), 5),
            "Recall@10_mean": round(float(np.mean(seed_recall)), 5),
            "Recall@10_std": round(float(np.std(seed_recall)), 5),
            "per_seed": per_seed_dict,
        }
        wandb.log({
            f"summary/{ds_name}/nDCG@10_mean": all_results[ds_name]["nDCG@10_mean"],
            f"summary/{ds_name}/nDCG@10_std": all_results[ds_name]["nDCG@10_std"],
            f"summary/{ds_name}/Recall@10_mean": all_results[ds_name]["Recall@10_mean"],
            f"summary/{ds_name}/Recall@10_std": all_results[ds_name]["Recall@10_std"],
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_dryrun" if args.dry_run else ""
    out_file = RESULTS_DIR / f"shuffled_pair{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nShuffled-pair results saved to {out_file}")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 60)
    for ds_name in datasets_to_eval:
        r = all_results[ds_name]
        logger.info(f"  {ds_name}: nDCG@10={r['nDCG@10_mean']:.5f} +/- {r['nDCG@10_std']:.5f}, "
                    f"Recall@10={r['Recall@10_mean']:.5f} +/- {r['Recall@10_std']:.5f}")

    wandb.finish()
    logger.info("Done!")


if __name__ == "__main__":
    main()
