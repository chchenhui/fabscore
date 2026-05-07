"""In-domain drift adapter training and evaluation on 4 BEIR datasets x 3 seeds.

For each dataset, samples N_p=5000 corpus documents (or all if < 5000),
trains a ResidualMLPAdapter mapping f_new -> f_old with MSE loss, then
evaluates adapted query retrieval against the legacy corpus index.

Usage:
  python -m pada.scripts.run_in_domain              # full run
  python -m pada.scripts.run_in_domain --dry-run     # SciFact only, 100 docs, 5 epochs
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
OUTPUT_DIR = PROJECT_ROOT / "pada" / "outputs" / "in_domain"
RESULTS_DIR = PROJECT_ROOT / "pada" / "results"

N_P = 5000
SEEDS = [0, 1, 2]


def load_embeddings(dataset_name: str, suffix: str = ""):
    ds_dir = EMBED_DIR / dataset_name
    corpus_fnew = np.load(str(ds_dir / f"corpus_fnew{suffix}.npy"))
    corpus_fold = np.load(str(ds_dir / f"corpus_fold{suffix}.npy"))
    queries_fnew = np.load(str(ds_dir / f"queries_fnew{suffix}.npy"))
    with open(ds_dir / f"corpus_ids{suffix}.json") as f:
        corpus_ids = json.load(f)
    with open(ds_dir / f"query_ids{suffix}.json") as f:
        query_ids = json.load(f)
    return corpus_fnew, corpus_fold, queries_fnew, corpus_ids, query_ids


def sample_indices(n_total: int, n_sample: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if n_total <= n_sample:
        return np.arange(n_total)
    return rng.choice(n_total, size=n_sample, replace=False)


def adapt_queries(model: ResidualMLPAdapter, queries_fnew: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        q_tensor = torch.tensor(queries_fnew, dtype=torch.float32).to(device)
        adapted = model(q_tensor)
    return adapted.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="SciFact only, 100 docs, 5 epochs")
    args = parser.parse_args()

    datasets_to_run = ["scifact"] if args.dry_run else DATASETS
    max_corpus = 100 if args.dry_run else None
    max_epochs = 5 if args.dry_run else 50
    n_p = min(N_P, max_corpus) if max_corpus else N_P
    suffix = "_dryrun" if args.dry_run else ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL'}")
    logger.info(f"Datasets: {datasets_to_run}, N_p={n_p}, max_epochs={max_epochs}")

    import wandb
    run_name = "in-domain-adapter"
    if args.dry_run:
        run_name += "-dryrun"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "public-anchor-drift-adapter"),
        name=run_name,
        config={
            "task": "in-domain-adapter",
            "n_p": n_p,
            "seeds": SEEDS,
            "datasets": datasets_to_run,
            "max_epochs": max_epochs,
            "lr": 3e-4,
            "weight_decay": 0.01,
            "batch_size": 256,
            "patience": 5,
            "embed_dim": 768,
            "hidden_dim": 256,
            "dry_run": args.dry_run,
        },
    )

    all_results = {}
    per_seed_results = {}

    for ds_name in datasets_to_run:
        logger.info(f"{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        corpus_fnew, corpus_fold, queries_fnew, corpus_ids, query_ids = load_embeddings(
            ds_name, suffix=suffix
        )

        if max_corpus and len(corpus_ids) > max_corpus:
            corpus_fnew = corpus_fnew[:max_corpus]
            corpus_fold = corpus_fold[:max_corpus]
            corpus_ids = corpus_ids[:max_corpus]

        logger.info(f"Corpus: {len(corpus_ids)} docs, Queries: {len(query_ids)}")

        _, queries, qrels = load_dataset(ds_name)

        seed_metrics = []
        per_seed_results[ds_name] = {}

        for seed in SEEDS:
            logger.info(f"\n--- {ds_name} / seed={seed} ---")

            sampled_idx = sample_indices(len(corpus_ids), n_p, seed)
            logger.info(f"Sampled {len(sampled_idx)} documents (seed={seed})")

            save_dir = OUTPUT_DIR / ds_name / f"seed_{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)

            sampled_doc_ids = [corpus_ids[i] for i in sampled_idx]
            with open(save_dir / "sampled_doc_ids.json", "w") as f:
                json.dump(sampled_doc_ids, f)

            source = corpus_fnew[sampled_idx]
            target = corpus_fold[sampled_idx]

            t0 = time.time()
            result = train_adapter(
                source_embeds=source,
                target_embeds=target,
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
                    f"train/{ds_name}/seed_{seed}/train_loss": tl,
                    f"train/{ds_name}/seed_{seed}/val_loss": vl,
                    f"train/{ds_name}/seed_{seed}/epoch": epoch_i + 1,
                })

            model = result["model"]
            model.to(device)
            adapted_queries = adapt_queries(model, queries_fnew, device)

            metrics = evaluate_retrieval(
                adapted_queries, corpus_fold, query_ids, corpus_ids, qrels
            )
            logger.info(f"Retrieval: nDCG@10={metrics['NDCG@10']:.5f}, "
                        f"Recall@10={metrics['Recall@10']:.5f}")

            wandb.log({
                f"eval/{ds_name}/seed_{seed}/nDCG@10": metrics["NDCG@10"],
                f"eval/{ds_name}/seed_{seed}/Recall@10": metrics["Recall@10"],
                f"eval/{ds_name}/seed_{seed}/best_val_loss": result["best_val_loss"],
                f"eval/{ds_name}/seed_{seed}/epochs_trained": result["epochs_trained"],
                f"eval/{ds_name}/seed_{seed}/train_time_s": train_time,
            })

            per_seed_results[ds_name][f"seed_{seed}"] = {
                "nDCG@10": round(metrics["NDCG@10"], 5),
                "Recall@10": round(metrics["Recall@10"], 5),
                "best_val_loss": round(result["best_val_loss"], 6),
                "epochs_trained": result["epochs_trained"],
                "train_time_s": round(train_time, 1),
                "n_train_pairs": len(sampled_idx),
            }
            seed_metrics.append(metrics)

        ndcg_values = [m["NDCG@10"] for m in seed_metrics]
        recall_values = [m["Recall@10"] for m in seed_metrics]
        all_results[ds_name] = {
            "nDCG@10_mean": round(float(np.mean(ndcg_values)), 5),
            "nDCG@10_std": round(float(np.std(ndcg_values)), 5),
            "Recall@10_mean": round(float(np.mean(recall_values)), 5),
            "Recall@10_std": round(float(np.std(recall_values)), 5),
            "per_seed": per_seed_results[ds_name],
        }

        wandb.log({
            f"summary/{ds_name}/nDCG@10_mean": all_results[ds_name]["nDCG@10_mean"],
            f"summary/{ds_name}/nDCG@10_std": all_results[ds_name]["nDCG@10_std"],
            f"summary/{ds_name}/Recall@10_mean": all_results[ds_name]["Recall@10_mean"],
            f"summary/{ds_name}/Recall@10_std": all_results[ds_name]["Recall@10_std"],
        })

        logger.info(f"\n{ds_name} summary: "
                     f"nDCG@10={all_results[ds_name]['nDCG@10_mean']:.5f} "
                     f"+/- {all_results[ds_name]['nDCG@10_std']:.5f}, "
                     f"Recall@10={all_results[ds_name]['Recall@10_mean']:.5f} "
                     f"+/- {all_results[ds_name]['Recall@10_std']:.5f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"in_domain{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_file}")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for ds_name in datasets_to_run:
        r = all_results[ds_name]
        logger.info(f"  {ds_name}: nDCG@10={r['nDCG@10_mean']:.5f} +/- {r['nDCG@10_std']:.5f}, "
                     f"Recall@10={r['Recall@10_mean']:.5f} +/- {r['Recall@10_std']:.5f}")

    wandb.finish()
    logger.info("Done!")


if __name__ == "__main__":
    main()
