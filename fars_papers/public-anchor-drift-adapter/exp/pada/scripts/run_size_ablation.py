"""Size ablation: how N_p (number of public anchor pairs) affects retrieval recovery.

Encodes 10000 Wikipedia paragraphs with both f_old and f_new, then trains
adapters on nested subsets {500, 1000, 2000, 5000, 10000} and evaluates
on all 4 BEIR datasets. Saves results to pada/results/size_ablation.json.

Usage:
  python -m pada.scripts.run_size_ablation              # full run
  python -m pada.scripts.run_size_ablation --dry-run    # Np=500 only, 5 epochs, 1 dataset
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
from pada.data.wikipedia_anchors import load_wikipedia_paragraphs, sample_anchors
from pada.evaluation.retrieval_eval import evaluate_retrieval
from pada.trainers.adapter_trainer import train_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_DIR = PROJECT_ROOT / "pada" / "embeddings"
OUTPUT_DIR = PROJECT_ROOT / "pada" / "outputs" / "size_ablation"
RESULTS_DIR = PROJECT_ROOT / "pada" / "results"

DATASETS = ["scifact", "trec-covid", "fiqa", "arguana"]
N_SIZES = [500, 1000, 2000, 5000, 10000]
N_MAX = 10000
SEED = 0


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


def load_beir_qrels(dataset_name: str):
    from pada.data.beir_loader import load_dataset as load_beir
    _, _, qrels = load_beir(dataset_name)
    return qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Np=500 only, 5 epochs, 1 dataset")
    args = parser.parse_args()

    sizes = [500] if args.dry_run else N_SIZES
    datasets_to_eval = ["scifact"] if args.dry_run else DATASETS
    max_epochs = 5 if args.dry_run else 50
    n_max = 500 if args.dry_run else N_MAX

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL'}")
    logger.info(f"Sizes: {sizes}, Datasets: {datasets_to_eval}, max_epochs={max_epochs}")

    import wandb
    run_name = "size-ablation"
    if args.dry_run:
        run_name += "-dryrun"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "public-anchor-drift-adapter"),
        name=run_name,
        config={
            "task": "size-ablation",
            "sizes": sizes,
            "seed": SEED,
            "datasets": datasets_to_eval,
            "max_epochs": max_epochs,
            "lr": 3e-4,
            "weight_decay": 0.01,
            "patience": 5,
            "embed_dim": 768,
            "hidden_dim": 256,
            "dry_run": args.dry_run,
            "data_source": "wikipedia",
        },
    )

    logger.info("Loading Wikipedia paragraphs (streaming from HuggingFace)...")
    paragraphs = load_wikipedia_paragraphs(max_paragraphs=max(n_max * 2, 50000))
    logger.info(f"Loaded {len(paragraphs)} Wikipedia paragraphs from pool")

    sampled = sample_anchors(n=n_max, seed=SEED, paragraphs=paragraphs)
    logger.info(f"Sampled {len(sampled)} paragraphs (seed={SEED})")

    logger.info("Loading sentence-transformer models for encoding...")
    model_old = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device=device)
    model_new = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

    logger.info(f"Encoding {len(sampled)} paragraphs with f_old...")
    embeds_fold = model_old.encode(sampled, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
    logger.info(f"Encoding {len(sampled)} paragraphs with f_new...")
    embeds_fnew = model_new.encode(sampled, normalize_embeddings=True, show_progress_bar=True, batch_size=128)

    save_embed_dir = EMBED_DIR / "wikipedia" / "size_ablation"
    save_embed_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(save_embed_dir / "source_fnew.npy"), embeds_fnew)
    np.save(str(save_embed_dir / "target_fold.npy"), embeds_fold)
    with open(save_embed_dir / "paragraphs.json", "w") as f:
        json.dump(sampled, f)
    logger.info(f"Saved all {len(sampled)} Wikipedia embeddings to {save_embed_dir}")

    del model_old, model_new
    torch.cuda.empty_cache()

    logger.info("Loading BEIR qrels...")
    qrels_map = {}
    for ds_name in datasets_to_eval:
        qrels_map[ds_name] = load_beir_qrels(ds_name)

    logger.info("Loading BEIR embeddings...")
    beir_data = {}
    for ds_name in datasets_to_eval:
        beir_data[ds_name] = load_dataset_embeddings(ds_name)
        logger.info(f"  {ds_name}: queries={beir_data[ds_name][0].shape}, corpus={beir_data[ds_name][1].shape}")

    all_results = {}

    for n_p in sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING ADAPTER: N_p = {n_p}")
        logger.info(f"{'='*60}")

        sub_fnew = embeds_fnew[:n_p]
        sub_fold = embeds_fold[:n_p]

        save_dir = OUTPUT_DIR / f"Np_{n_p}"
        save_dir.mkdir(parents=True, exist_ok=True)

        n_train = int(n_p * 0.8)
        bs = min(256, n_train)
        logger.info(f"  N_p={n_p}, N_train={n_train}, batch_size={bs}")

        wandb_run_name = f"size-ablation-Np-{n_p}"
        if args.dry_run:
            wandb_run_name += "-dryrun"

        t0 = time.time()
        result = train_adapter(
            source_embeds=sub_fnew,
            target_embeds=sub_fold,
            save_dir=str(save_dir),
            embed_dim=768,
            hidden_dim=256,
            lr=3e-4,
            weight_decay=0.01,
            batch_size=bs,
            max_epochs=max_epochs,
            patience=5,
            seed=SEED,
            device=device,
        )
        train_time = time.time() - t0

        model = result["model"]
        best_val_loss = result["best_val_loss"]
        epochs_trained = result["epochs_trained"]

        logger.info(f"  Trained in {train_time:.1f}s, {epochs_trained} epochs, best_val_loss={best_val_loss:.6f}")

        wandb.log({
            f"train/Np_{n_p}/best_val_loss": best_val_loss,
            f"train/Np_{n_p}/epochs_trained": epochs_trained,
            f"train/Np_{n_p}/train_time_s": train_time,
        })
        for i, (tl, vl) in enumerate(zip(result["history"]["train_loss"], result["history"]["val_loss"])):
            wandb.log({
                f"train/Np_{n_p}/epoch": i + 1,
                f"train/Np_{n_p}/train_loss": tl,
                f"train/Np_{n_p}/val_loss": vl,
                f"train/Np_{n_p}/lr": result["history"]["lr"][i],
            })

        np_results = {"n_p": n_p, "best_val_loss": best_val_loss, "epochs_trained": epochs_trained, "train_time_s": round(train_time, 1)}

        for ds_name in datasets_to_eval:
            queries_fnew_ds, corpus_fold_ds, corpus_ids, query_ids = beir_data[ds_name]
            qrels = qrels_map[ds_name]

            adapted_queries = adapt_queries(model, queries_fnew_ds, device)
            metrics = evaluate_retrieval(adapted_queries, corpus_fold_ds, query_ids, corpus_ids, qrels)

            np_results[ds_name] = {
                "nDCG@10": round(metrics["NDCG@10"], 5),
                "Recall@10": round(metrics["Recall@10"], 5),
            }
            logger.info(f"  {ds_name}: nDCG@10={metrics['NDCG@10']:.5f}, Recall@10={metrics['Recall@10']:.5f}")

            wandb.log({
                f"eval/Np_{n_p}/{ds_name}/nDCG@10": metrics["NDCG@10"],
                f"eval/Np_{n_p}/{ds_name}/Recall@10": metrics["Recall@10"],
            })

        all_results[str(n_p)] = np_results

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_dryrun" if args.dry_run else ""
    out_file = RESULTS_DIR / f"size_ablation{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSize ablation results saved to {out_file}")

    logger.info("\n" + "=" * 60)
    logger.info("SIZE ABLATION SUMMARY")
    logger.info("=" * 60)
    header = f"{'N_p':>8}"
    for ds_name in datasets_to_eval:
        header += f"  {ds_name:>15}"
    logger.info(header)
    logger.info("-" * (8 + 17 * len(datasets_to_eval)))

    for n_p in sizes:
        row = f"{n_p:>8}"
        for ds_name in datasets_to_eval:
            ndcg = all_results[str(n_p)][ds_name]["nDCG@10"]
            row += f"  {ndcg:>15.5f}"
        logger.info(row)

    wandb.finish()
    logger.info("Done!")


if __name__ == "__main__":
    main()
