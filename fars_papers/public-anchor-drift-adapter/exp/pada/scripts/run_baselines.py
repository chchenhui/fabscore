"""Oracle and Misaligned baseline evaluation on 4 BEIR datasets.

Encodes corpus/queries with f_old (all-distilroberta-v1) and f_new (all-mpnet-base-v2),
saves embeddings, then evaluates Oracle (fnew-fnew) and Misaligned (fold-fnew) conditions.
Logs results to WandB and saves JSON files.

Usage:
  python -m pada.scripts.run_baselines              # full run
  python -m pada.scripts.run_baselines --dry-run     # SciFact only, 100 corpus docs
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pada.data.beir_loader import DATASETS, load_dataset
from pada.evaluation.retrieval_eval import evaluate_retrieval

EMBED_DIR = PROJECT_ROOT / "pada" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "pada" / "results"

MODEL_OLD = "sentence-transformers/all-distilroberta-v1"
MODEL_NEW = "sentence-transformers/all-mpnet-base-v2"


def prepare_texts(corpus: dict, queries: dict):
    corpus_ids = sorted(corpus.keys())
    corpus_texts = []
    for cid in corpus_ids:
        doc = corpus[cid]
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        corpus_texts.append(f"{title} {text}".strip() if title else text)

    query_ids = sorted(queries.keys())
    query_texts = [queries[qid].strip() for qid in query_ids]
    return corpus_ids, corpus_texts, query_ids, query_texts


def encode_and_save(
    model: SentenceTransformer,
    texts: list[str],
    save_path: pathlib.Path,
    batch_size: int = 128,
):
    if save_path.exists():
        print(f"  Loading cached embeddings from {save_path}")
        return np.load(str(save_path))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    np.save(str(save_path), embeddings)
    print(f"  Saved embeddings {embeddings.shape} to {save_path}")
    return embeddings


def save_ids(ids: list[str], save_path: pathlib.Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(ids, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on SciFact only with 100 corpus docs")
    args = parser.parse_args()

    datasets_to_run = ["scifact"] if args.dry_run else DATASETS
    max_corpus = 100 if args.dry_run else None

    print(f"Mode: {'DRY-RUN' if args.dry_run else 'FULL'}")
    print(f"Datasets: {datasets_to_run}")

    import wandb
    run_name = "baseline-oracle-misaligned"
    if args.dry_run:
        run_name += "-dryrun"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "public-anchor-drift-adapter"),
        name=run_name,
        config={
            "task": "baseline-oracle-misaligned",
            "model_old": MODEL_OLD,
            "model_new": MODEL_NEW,
            "datasets": datasets_to_run,
            "dry_run": args.dry_run,
        },
    )

    print("Loading models...")
    model_old = SentenceTransformer(MODEL_OLD)
    model_new = SentenceTransformer(MODEL_NEW)

    oracle_results = {}
    misaligned_results = {}
    dataset_stats = {}

    for ds_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {ds_name}")
        print(f"{'='*60}")

        corpus, queries, qrels = load_dataset(ds_name)

        corpus_ids, corpus_texts, query_ids, query_texts = prepare_texts(corpus, queries)

        if max_corpus and len(corpus_ids) > max_corpus:
            print(f"  DRY-RUN: Truncating corpus from {len(corpus_ids)} to {max_corpus}")
            corpus_ids = corpus_ids[:max_corpus]
            corpus_texts = corpus_texts[:max_corpus]

        dataset_stats[ds_name] = {
            "corpus_size": len(corpus_ids),
            "query_count": len(query_ids),
        }
        print(f"  Corpus: {len(corpus_ids)} docs, Queries: {len(query_ids)}")

        ds_embed_dir = EMBED_DIR / ds_name
        suffix = "_dryrun" if args.dry_run else ""

        print(f"\n  Encoding corpus with f_old ({MODEL_OLD})...")
        t0 = time.time()
        corpus_fold = encode_and_save(
            model_old, corpus_texts, ds_embed_dir / f"corpus_fold{suffix}.npy"
        )
        print(f"  Done in {time.time()-t0:.1f}s")

        print(f"\n  Encoding corpus with f_new ({MODEL_NEW})...")
        t0 = time.time()
        corpus_fnew = encode_and_save(
            model_new, corpus_texts, ds_embed_dir / f"corpus_fnew{suffix}.npy"
        )
        print(f"  Done in {time.time()-t0:.1f}s")

        print(f"\n  Encoding queries with f_old ({MODEL_OLD})...")
        t0 = time.time()
        queries_fold = encode_and_save(
            model_old, query_texts, ds_embed_dir / f"queries_fold{suffix}.npy"
        )
        print(f"  Done in {time.time()-t0:.1f}s")

        print(f"\n  Encoding queries with f_new ({MODEL_NEW})...")
        t0 = time.time()
        queries_fnew = encode_and_save(
            model_new, query_texts, ds_embed_dir / f"queries_fnew{suffix}.npy"
        )
        print(f"  Done in {time.time()-t0:.1f}s")

        save_ids(corpus_ids, ds_embed_dir / f"corpus_ids{suffix}.json")
        save_ids(query_ids, ds_embed_dir / f"query_ids{suffix}.json")

        print(f"\n  Evaluating ORACLE (fnew corpus + fnew queries)...")
        oracle_metrics = evaluate_retrieval(
            queries_fnew, corpus_fnew, query_ids, corpus_ids, qrels
        )
        oracle_results[ds_name] = {
            "nDCG@10": round(oracle_metrics["NDCG@10"], 5),
            "Recall@10": round(oracle_metrics["Recall@10"], 5),
        }
        print(f"  Oracle: nDCG@10={oracle_metrics['NDCG@10']:.5f}, Recall@10={oracle_metrics['Recall@10']:.5f}")

        print(f"\n  Evaluating MISALIGNED (fold corpus + fnew queries)...")
        mis_metrics = evaluate_retrieval(
            queries_fnew, corpus_fold, query_ids, corpus_ids, qrels
        )
        misaligned_results[ds_name] = {
            "nDCG@10": round(mis_metrics["NDCG@10"], 5),
            "Recall@10": round(mis_metrics["Recall@10"], 5),
        }
        print(f"  Misaligned: nDCG@10={mis_metrics['NDCG@10']:.5f}, Recall@10={mis_metrics['Recall@10']:.5f}")

        wandb.log({
            f"oracle/{ds_name}/nDCG@10": oracle_metrics["NDCG@10"],
            f"oracle/{ds_name}/Recall@10": oracle_metrics["Recall@10"],
            f"misaligned/{ds_name}/nDCG@10": mis_metrics["NDCG@10"],
            f"misaligned/{ds_name}/Recall@10": mis_metrics["Recall@10"],
        })

    headroom = {}
    print(f"\n{'='*60}")
    print("HEADROOM SUMMARY")
    print(f"{'='*60}")
    for ds_name in datasets_to_run:
        gap_ndcg = oracle_results[ds_name]["nDCG@10"] - misaligned_results[ds_name]["nDCG@10"]
        gap_recall = oracle_results[ds_name]["Recall@10"] - misaligned_results[ds_name]["Recall@10"]
        headroom[ds_name] = {
            "oracle_nDCG@10": oracle_results[ds_name]["nDCG@10"],
            "misaligned_nDCG@10": misaligned_results[ds_name]["nDCG@10"],
            "gap_nDCG@10": round(gap_ndcg, 5),
            "oracle_Recall@10": oracle_results[ds_name]["Recall@10"],
            "misaligned_Recall@10": misaligned_results[ds_name]["Recall@10"],
            "gap_Recall@10": round(gap_recall, 5),
            "sufficient_headroom": gap_ndcg >= 0.05,
        }
        print(f"  {ds_name}: gap_nDCG@10={gap_ndcg:.5f} ({'SUFFICIENT' if gap_ndcg >= 0.05 else 'INSUFFICIENT'})")
        wandb.log({
            f"headroom/{ds_name}/gap_nDCG@10": gap_ndcg,
            f"headroom/{ds_name}/gap_Recall@10": gap_recall,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_dryrun" if args.dry_run else ""

    with open(RESULTS_DIR / f"oracle{suffix}.json", "w") as f:
        json.dump(oracle_results, f, indent=2)
    with open(RESULTS_DIR / f"misaligned{suffix}.json", "w") as f:
        json.dump(misaligned_results, f, indent=2)
    with open(RESULTS_DIR / f"headroom_summary{suffix}.json", "w") as f:
        json.dump(headroom, f, indent=2)
    with open(RESULTS_DIR / f"dataset_stats{suffix}.json", "w") as f:
        json.dump(dataset_stats, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Done!")
    wandb.finish()


if __name__ == "__main__":
    main()
