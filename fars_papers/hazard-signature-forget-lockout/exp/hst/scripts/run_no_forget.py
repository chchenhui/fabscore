# Condition: No-Forget reference.
# Runs retrieval on the full 110-seed store (100 benign + 10 poisoned)
# without any forget operation, establishing baseline PRP and Benign Recall.
#
# Outputs: hst/outputs/no_forget_results.json

import json
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from hst.data.download_seeds import load_seeds, load_eval_queries
from hst.retrieval.embedder import Embedder
from hst.retrieval.faiss_index import FaissRetriever
from hst.evaluation.metrics import prp_at_k, benign_recall_at_k

OUTPUT_DIR = PROJECT_ROOT / "outputs"
K = 3


def main():
    print("=== No-Forget Reference Experiment ===")

    print("Loading seeds...")
    benign, poisoned = load_seeds()
    queries = load_eval_queries()
    all_records = benign + poisoned
    record_ids = [r["id"] for r in all_records]
    poisoned_id_set = {r["id"] for r in poisoned}
    benign_ids = [r["id"] for r in benign]

    print(f"  Benign: {len(benign)}, Poisoned: {len(poisoned)}, Total: {len(all_records)}")
    print(f"  Eval queries: {len(queries)}")

    print("Computing record embeddings...")
    embedder = Embedder(model_name="BAAI/bge-m3")
    all_embeddings = embedder.encode(all_records, cache_prefix="all_110")
    print(f"  Embedding shape: {all_embeddings.shape}")

    print("Building FAISS index (all 110 records)...")
    retriever = FaissRetriever(dim=all_embeddings.shape[1])
    retriever.build_index(all_embeddings)
    print(f"  Index size: {retriever.ntotal}")

    print(f"Encoding {len(queries)} evaluation queries...")
    query_embeddings = embedder.encode_queries(queries, cache_prefix="eval_queries")

    print(f"Retrieving top-{K} for each eval query...")
    distances, indices = retriever.search(query_embeddings, k=K)

    eval_retrieved_ids = []
    eval_details = []
    for qi, q in enumerate(queries):
        top_ids = [record_ids[int(idx)] for idx in indices[qi] if idx >= 0]
        top_scores = [float(distances[qi][j]) for j in range(len(top_ids))]
        eval_retrieved_ids.append(top_ids)
        eval_details.append({
            "query": q,
            "retrieved_ids": top_ids,
            "scores": top_scores,
            "has_poisoned": any(rid in poisoned_id_set for rid in top_ids),
        })

    prp3 = prp_at_k(eval_retrieved_ids, poisoned_id_set, K)
    print(f"\n  PRP@{K} = {prp3:.4f}")

    print(f"\nComputing Benign Recall@{K} (self-query on full 110-item index)...")
    benign_query_embeddings = embedder.encode_queries(
        [r["req"] for r in benign], cache_prefix="benign_req_queries"
    )
    _, benign_indices = retriever.search(benign_query_embeddings, k=K)

    benign_retrieved_ids = []
    for bi in range(len(benign)):
        top_ids = [record_ids[int(idx)] for idx in benign_indices[bi] if idx >= 0]
        benign_retrieved_ids.append(top_ids)

    br3 = benign_recall_at_k(benign_retrieved_ids, benign_ids, K)
    print(f"  Benign Recall@{K} (full index) = {br3:.4f}")

    print(f"\nSanity check: benign-only index (100 items)...")
    benign_embeddings = all_embeddings[:len(benign)]
    benign_retriever = FaissRetriever(dim=benign_embeddings.shape[1])
    benign_retriever.build_index(benign_embeddings)

    _, benign_only_indices = benign_retriever.search(benign_query_embeddings, k=K)
    benign_only_retrieved_ids = []
    for bi in range(len(benign)):
        top_ids = [benign_ids[int(idx)] for idx in benign_only_indices[bi] if idx >= 0]
        benign_only_retrieved_ids.append(top_ids)

    br3_clean = benign_recall_at_k(benign_only_retrieved_ids, benign_ids, K)
    print(f"  Benign Recall@{K} (benign-only) = {br3_clean:.4f}")
    sanity_ok = br3_clean >= 0.95
    print(f"  Sanity check {'PASSED' if sanity_ok else 'FAILED'} (threshold >= 0.95)")

    results = {
        "condition": "no_forget",
        "description": "Full 110-seed store (100 benign + 10 poisoned), no deletion",
        "k": K,
        "n_benign": len(benign),
        "n_poisoned": len(poisoned),
        "n_eval_queries": len(queries),
        "prp_at_3": round(prp3, 4),
        "benign_recall_at_3": round(br3, 4),
        "benign_recall_at_3_clean_index": round(br3_clean, 4),
        "sanity_check_passed": sanity_ok,
        "eval_query_details": eval_details,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "no_forget_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n=== Summary ===")
    print(f"  PRP@{K}:                        {prp3:.4f}")
    print(f"  Benign Recall@{K} (full index):  {br3:.4f}")
    print(f"  Benign Recall@{K} (clean index): {br3_clean:.4f}")
    print(f"  Sanity check:                    {'PASS' if sanity_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
