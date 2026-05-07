# Condition A: ID-delete only.
# Deletes 10 poisoned records by ID, then allows all paraphrased re-injections.
# Demonstrates the weakness of ID-based deletion against paraphrase attacks.
#
# Flow:
# 1. Load 110 records (100 benign + 10 poisoned), reuse cached embeddings
# 2. Build FAISS index, remove 10 poisoned IDs -> 100 benign remain
# 3. Embed 50 paraphrased poisoned records, add all to index -> 150 total
# 4. Retrieve top-3 for 12 eval queries, compute PRP@3
# 5. Compute Benign Recall@3 on 150-item index
# 6. WriteBlockRate=0%, Benign FalseBlock=0%
#
# Outputs: hst/outputs/condition_a_results.json

import json
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from hst.data.download_seeds import load_seeds, load_eval_queries
from hst.data.paraphrase import load_paraphrases
from hst.retrieval.embedder import Embedder
from hst.retrieval.faiss_index import FaissRetriever
from hst.evaluation.metrics import prp_at_k, benign_recall_at_k

OUTPUT_DIR = PROJECT_ROOT / "outputs"
K = 3


def main():
    print("=== Condition A: ID-Delete Only ===")

    print("Loading seeds and paraphrases...")
    benign, poisoned = load_seeds()
    paraphrases = load_paraphrases()
    queries = load_eval_queries()

    all_original = benign + poisoned
    original_ids = [r["id"] for r in all_original]
    poisoned_id_set = {r["id"] for r in poisoned}
    benign_ids = [r["id"] for r in benign]
    para_ids = [p["id"] for p in paraphrases]
    para_poisoned_id_set = set(para_ids)

    print(f"  Benign: {len(benign)}, Poisoned: {len(poisoned)}")
    print(f"  Paraphrases: {len(paraphrases)}")
    print(f"  Eval queries: {len(queries)}")

    print("Loading cached embeddings for 110 original records...")
    embedder = Embedder(model_name="BAAI/bge-m3")
    all_embeddings = embedder.encode(all_original, cache_prefix="all_110")
    print(f"  Embedding shape: {all_embeddings.shape}")

    print("Building FAISS index with all 110 records (integer ID mapped)...")
    retriever = FaissRetriever(dim=all_embeddings.shape[1])
    int_ids = np.arange(len(all_original), dtype=np.int64)
    retriever.build_index(all_embeddings, ids=int_ids)
    print(f"  Index size before deletion: {retriever.ntotal}")

    print("Deleting 10 poisoned record IDs from index...")
    poisoned_int_ids = [i for i, r in enumerate(all_original) if r["id"] in poisoned_id_set]
    retriever.remove(poisoned_int_ids)
    print(f"  Index size after deletion: {retriever.ntotal} (expected 100)")

    print("Computing embeddings for 50 paraphrased records...")
    para_embeddings = embedder.encode(paraphrases, cache_prefix="paraphrases_50")
    print(f"  Paraphrase embedding shape: {para_embeddings.shape}")

    print("Adding 50 paraphrased records to index (no write-time filtering)...")
    para_int_ids = np.arange(len(all_original), len(all_original) + len(paraphrases), dtype=np.int64)
    retriever.add(para_embeddings, ids=para_int_ids)
    print(f"  Index size after adding paraphrases: {retriever.ntotal} (expected 150)")

    combined_id_list = original_ids + para_ids

    all_poisoned_ids = poisoned_id_set | para_poisoned_id_set

    print(f"\nEncoding {len(queries)} evaluation queries...")
    query_embeddings = embedder.encode_queries(queries, cache_prefix="eval_queries")

    print(f"Retrieving top-{K} for each eval query...")
    distances, indices = retriever.search(query_embeddings, k=K)

    eval_retrieved_ids = []
    eval_details = []
    for qi, q in enumerate(queries):
        top_ids = [combined_id_list[int(idx)] for idx in indices[qi] if idx >= 0]
        top_scores = [float(distances[qi][j]) for j in range(len(top_ids))]
        eval_retrieved_ids.append(top_ids)
        eval_details.append({
            "query": q,
            "retrieved_ids": top_ids,
            "scores": top_scores,
            "has_poisoned": any(rid in all_poisoned_ids for rid in top_ids),
        })

    prp3 = prp_at_k(eval_retrieved_ids, all_poisoned_ids, K)
    print(f"\n  PRP@{K} = {prp3:.4f}")

    print(f"\nComputing Benign Recall@{K} on 150-item index...")
    benign_query_embeddings = embedder.encode_queries(
        [r["req"] for r in benign], cache_prefix="benign_req_queries"
    )
    _, benign_indices = retriever.search(benign_query_embeddings, k=K)

    benign_retrieved_ids = []
    for bi in range(len(benign)):
        top_ids = [combined_id_list[int(idx)] for idx in benign_indices[bi] if idx >= 0]
        benign_retrieved_ids.append(top_ids)

    br3 = benign_recall_at_k(benign_retrieved_ids, benign_ids, K)
    print(f"  Benign Recall@{K} (150-item index) = {br3:.4f}")

    n_queries_with_poison = sum(1 for d in eval_details if d["has_poisoned"])

    results = {
        "condition": "condition_a_id_delete_only",
        "description": "Delete 10 poisoned IDs, allow all paraphrased re-injections (50 paraphrases)",
        "k": K,
        "n_benign": len(benign),
        "n_poisoned_original": len(poisoned),
        "n_paraphrases": len(paraphrases),
        "n_total_in_index": retriever.ntotal,
        "n_eval_queries": len(queries),
        "prp_at_3": round(prp3, 4),
        "benign_recall_at_3": round(br3, 4),
        "write_block_rate": 0.0,
        "benign_false_block": 0.0,
        "n_queries_with_poisoned_in_top3": n_queries_with_poison,
        "eval_query_details": eval_details,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "condition_a_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n=== Condition A Summary ===")
    print(f"  PRP@{K}:                  {prp3:.4f}")
    print(f"  Benign Recall@{K}:         {br3:.4f}")
    print(f"  WriteBlockRate:            0.00%")
    print(f"  Benign FalseBlock:         0.00%")
    print(f"  Queries with poison:       {n_queries_with_poison}/{len(queries)}")
    print(f"  Index size:                {retriever.ntotal}")


if __name__ == "__main__":
    main()
