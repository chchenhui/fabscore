"""Public-anchor drift adapter training and evaluation.

Trains 3 adapters (one per seed) on Wikipedia paragraph embeddings, then
evaluates each adapter on all 4 BEIR datasets. Computes recovery ratios
against in-domain adapter baselines.

Usage:
  python -m pada.scripts.run_public_anchor              # full run
  python -m pada.scripts.run_public_anchor --dry-run    # 1 seed, 1 dataset, 100 pairs, 5 epochs
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
from sentence_transformers import SentenceTransformer

load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pada.adapters.residual_mlp import ResidualMLPAdapter
from pada.data.beir_loader import DATASETS, load_dataset
from pada.data.wikipedia_anchors import load_wikipedia_paragraphs, sample_anchors
from pada.evaluation.retrieval_eval import evaluate_retrieval
from pada.trainers.adapter_trainer import train_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_DIR = PROJECT_ROOT / "pada" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "pada" / "outputs" / "public_anchor"
RESULTS_DIR = PROJECT_ROOT / "pada" / "results"
WIKI_CACHE_DIR = PROJECT_ROOT / "data" / "wikipedia_cache"

N_P = 5000
SEEDS = [0, 1, 2]


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


def compute_recovery_ratios(public_anchor_results: dict, results_dir: pathlib.Path):
    with open(results_dir / "oracle.json") as f:
        oracle = json.load(f)
    with open(results_dir / "misaligned.json") as f:
        misaligned = json.load(f)
    with open(results_dir / "in_domain.json") as f:
        in_domain = json.load(f)

    recovery = {}
    for ds_name in DATASETS:
        m_oracle_ndcg = oracle[ds_name]["nDCG@10"]
        m_mis_ndcg = misaligned[ds_name]["nDCG@10"]
        m_indomain_ndcg = in_domain[ds_name]["nDCG@10_mean"]
        m_pub_ndcg = public_anchor_results[ds_name]["nDCG@10_mean"]

        m_oracle_recall = oracle[ds_name]["Recall@10"]
        m_mis_recall = misaligned[ds_name]["Recall@10"]
        m_indomain_recall = in_domain[ds_name]["Recall@10_mean"]
        m_pub_recall = public_anchor_results[ds_name]["Recall@10_mean"]

        indomain_headroom_ndcg = m_indomain_ndcg - m_mis_ndcg
        indomain_headroom_recall = m_indomain_recall - m_mis_recall

        rho_ndcg = (m_pub_ndcg - m_mis_ndcg) / indomain_headroom_ndcg if indomain_headroom_ndcg > 1e-9 else None
        rho_recall = (m_pub_recall - m_mis_recall) / indomain_headroom_recall if indomain_headroom_recall > 1e-9 else None

        recovery[ds_name] = {
            "oracle_nDCG@10": round(m_oracle_ndcg, 5),
            "misaligned_nDCG@10": round(m_mis_ndcg, 5),
            "in_domain_nDCG@10_mean": round(m_indomain_ndcg, 5),
            "public_anchor_nDCG@10_mean": round(m_pub_ndcg, 5),
            "rho_nDCG@10": round(rho_ndcg, 5) if rho_ndcg is not None else None,
            "absolute_gap_nDCG@10": round(m_indomain_ndcg - m_pub_ndcg, 5),
            "oracle_Recall@10": round(m_oracle_recall, 5),
            "misaligned_Recall@10": round(m_mis_recall, 5),
            "in_domain_Recall@10_mean": round(m_indomain_recall, 5),
            "public_anchor_Recall@10_mean": round(m_pub_recall, 5),
            "rho_Recall@10": round(rho_recall, 5) if rho_recall is not None else None,
            "absolute_gap_Recall@10": round(m_indomain_recall - m_pub_recall, 5),
            "low_signal": indomain_headroom_ndcg < 0.05,
        }

    return recovery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, 1 dataset, 100 pairs, 5 epochs")
    args = parser.parse_args()

    seeds = [0] if args.dry_run else SEEDS
    datasets_to_eval = ["scifact"] if args.dry_run else DATASETS
    n_p = 100 if args.dry_run else N_P
    max_epochs = 5 if args.dry_run else 50
    max_paragraphs = 200 if args.dry_run else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL'}")
    logger.info(f"Seeds: {seeds}, Datasets: {datasets_to_eval}, N_p={n_p}, max_epochs={max_epochs}")

    import wandb
    run_name = "public-anchor-adapter"
    if args.dry_run:
        run_name += "-dryrun"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "public-anchor-drift-adapter"),
        name=run_name,
        config={
            "task": "public-anchor-adapter",
            "n_p": n_p,
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
            "data_source": "wikipedia",
        },
    )

    logger.info("Loading Wikipedia paragraphs (streaming from HuggingFace)...")
    paragraphs = load_wikipedia_paragraphs(max_paragraphs=max_paragraphs)
    logger.info(f"Loaded {len(paragraphs)} Wikipedia paragraphs")

    logger.info("Loading sentence-transformer models for encoding...")
    model_old = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device=device)
    model_new = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

    all_results = {}
    per_seed_all = {}

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*60}")

        sampled = sample_anchors(n=n_p, seed=seed, paragraphs=paragraphs)
        logger.info(f"Sampled {len(sampled)} paragraphs for seed={seed}")

        wiki_embed_dir = EMBED_DIR / "wikipedia" / f"seed_{seed}"
        wiki_embed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Encoding {len(sampled)} paragraphs with f_old (all-distilroberta-v1)...")
        embeds_fold = model_old.encode(sampled, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
        logger.info(f"Encoding {len(sampled)} paragraphs with f_new (all-mpnet-base-v2)...")
        embeds_fnew = model_new.encode(sampled, normalize_embeddings=True, show_progress_bar=True, batch_size=128)

        np.save(str(wiki_embed_dir / "source_fnew.npy"), embeds_fnew)
        np.save(str(wiki_embed_dir / "target_fold.npy"), embeds_fold)
        with open(wiki_embed_dir / "paragraphs.json", "w") as f:
            json.dump(sampled, f)
        logger.info(f"Saved Wikipedia embeddings to {wiki_embed_dir}")

        save_dir = OUTPUT_DIR / f"seed_{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training public-anchor adapter (seed={seed})...")
        t0 = time.time()
        result = train_adapter(
            source_embeds=embeds_fnew,
            target_embeds=embeds_fold,
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

        adapter_model = result["model"]
        adapter_model.to(device)

        per_seed_all[f"seed_{seed}"] = {}

        for ds_name in datasets_to_eval:
            logger.info(f"\n--- Evaluating seed={seed} on {ds_name} ---")

            queries_fnew, corpus_fold, corpus_ids, query_ids = load_dataset_embeddings(ds_name)
            _, queries, qrels = load_dataset(ds_name)

            adapted_queries = adapt_queries(adapter_model, queries_fnew, device)

            metrics = evaluate_retrieval(
                adapted_queries, corpus_fold, query_ids, corpus_ids, qrels
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
            f"train/seed_{seed}/n_train_pairs": len(sampled),
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
    out_file = RESULTS_DIR / f"public_anchor{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nPublic-anchor results saved to {out_file}")

    if not args.dry_run:
        recovery = compute_recovery_ratios(all_results, RESULTS_DIR)
        recovery_file = RESULTS_DIR / "recovery_ratios.json"
        with open(recovery_file, "w") as f:
            json.dump(recovery, f, indent=2)
        logger.info(f"Recovery ratios saved to {recovery_file}")

        for ds_name, r in recovery.items():
            wandb.log({
                f"recovery/{ds_name}/rho_nDCG@10": r["rho_nDCG@10"],
                f"recovery/{ds_name}/absolute_gap_nDCG@10": r["absolute_gap_nDCG@10"],
            })
            if r["rho_Recall@10"] is not None:
                wandb.log({f"recovery/{ds_name}/rho_Recall@10": r["rho_Recall@10"]})

        logger.info("\n" + "=" * 60)
        logger.info("RECOVERY RATIO SUMMARY")
        logger.info("=" * 60)
        logger.info(f"{'Dataset':<15} {'Oracle':>10} {'Misaligned':>12} {'In-domain':>12} {'Pub-anchor':>12} {'rho':>8} {'Gap':>8} {'Low-sig':>8}")
        logger.info("-" * 85)
        for ds_name in DATASETS:
            r = recovery[ds_name]
            rho_str = f"{r['rho_nDCG@10']:.4f}" if r['rho_nDCG@10'] is not None else "N/A"
            low_sig = "YES" if r["low_signal"] else "no"
            logger.info(
                f"{ds_name:<15} "
                f"{r['oracle_nDCG@10']:>10.5f} "
                f"{r['misaligned_nDCG@10']:>12.5f} "
                f"{r['in_domain_nDCG@10_mean']:>12.5f} "
                f"{r['public_anchor_nDCG@10_mean']:>12.5f} "
                f"{rho_str:>8} "
                f"{r['absolute_gap_nDCG@10']:>8.5f} "
                f"{low_sig:>8}"
            )

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
