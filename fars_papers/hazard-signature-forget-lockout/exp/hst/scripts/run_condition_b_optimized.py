# Condition B Optimized: Retrieval-time HS denylist with fuzzy matching.
# Uses hazard-set containment instead of exact HS string match for retrieval-time
# filtering, providing a fair comparison baseline for Condition C Optimized.
#
# Outputs: hst/outputs/condition_b_optimized_results.json

import json
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from hst.data.download_seeds import load_seeds, load_eval_queries
from hst.data.paraphrase import load_paraphrases
from hst.data.hazard_signature import load_signatures
from hst.retrieval.embedder import Embedder
from hst.retrieval.faiss_index import FaissRetriever
from hst.evaluation.metrics import prp_at_k, benign_recall_at_k
from hst.memory.policies import RetrievalTimeDenylistFuzzy

OUTPUT_DIR = PROJECT_ROOT / "outputs"
K = 3
DELTA = 6


def main():
    print("=== Condition B Optimized: Fuzzy Retrieval-Time HS Denylist + Backfill ===")

    print("Loading seeds, paraphrases, and hazard signatures...")
    benign, poisoned = load_seeds()
    paraphrases = load_paraphrases()
    queries = load_eval_queries()
    signatures = load_signatures()

    all_original = benign + poisoned
    original_ids = [r["id"] for r in all_original]
    poisoned_id_set = {r["id"] for r in poisoned}
    benign_ids = [r["id"] for r in benign]
    para_ids = [p["id"] for p in paraphrases]
    para_poisoned_id_set = set(para_ids)
    all_poisoned_ids = poisoned_id_set | para_poisoned_id_set

    print(f"  Benign: {len(benign)}, Poisoned: {len(poisoned)}")
    print(f"  Paraphrases: {len(paraphrases)}")
    print(f"  Eval queries: {len(queries)}")
    print(f"  Signatures loaded: {len(signatures)}")

    print("\nBuilding structured tombstone set from 10 original poisoned records...")
    tombstone_sigs = []
    tombstone_hs_set = set()
    for r in poisoned:
        sig = signatures[r["id"]]
        tombstone_sigs.append(sig)
        tombstone_hs_set.add(sig["hs"])
        print(f"  {r['id']}: hazards={sorted(sig.get('hazards', []))} (HS: {sig['hs']})")
    print(f"  Unique tombstone HS values: {len(tombstone_hs_set)}")

    print("\nLoading cached embeddings for 110 original records...")
    embedder = Embedder(model_name="BAAI/bge-m3")
    all_embeddings = embedder.encode(all_original, cache_prefix="all_110")
    print(f"  Embedding shape: {all_embeddings.shape}")

    print("Building FAISS index with all 110 records...")
    retriever = FaissRetriever(dim=all_embeddings.shape[1])
    int_ids = np.arange(len(all_original), dtype=np.int64)
    retriever.build_index(all_embeddings, ids=int_ids)

    print("Deleting 10 poisoned record IDs from index...")
    poisoned_int_ids = [i for i, r in enumerate(all_original) if r["id"] in poisoned_id_set]
    retriever.remove(poisoned_int_ids)
    print(f"  Index size after deletion: {retriever.ntotal} (expected 100)")

    print("Loading cached paraphrase embeddings and adding to index...")
    para_embeddings = embedder.encode(paraphrases, cache_prefix="paraphrases_50")
    para_int_ids = np.arange(len(all_original), len(all_original) + len(paraphrases), dtype=np.int64)
    retriever.add(para_embeddings, ids=para_int_ids)
    print(f"  Index size: {retriever.ntotal} (expected 150)")

    combined_id_list = original_ids + para_ids
    int_to_record_id = {i: combined_id_list[i] for i in range(len(combined_id_list))}

    hs_map = {}
    for rid in combined_id_list:
        if rid in signatures:
            hs_map[rid] = signatures[rid]
        else:
            hs_map[rid] = {}

    denylist = RetrievalTimeDenylistFuzzy(tombstone_sigs=tombstone_sigs, delta=DELTA)

    print(f"\nEncoding {len(queries)} evaluation queries...")
    query_embeddings = embedder.encode_queries(queries, cache_prefix="eval_queries")

    print(f"Retrieving top-{K+DELTA} and fuzzy-filtering to top-{K} for each eval query...")
    eval_retrieved_ids = []
    eval_details = []
    for qi, q in enumerate(queries):
        passing_ids, passing_scores, n_filtered = denylist.filter_results(
            query_embeddings[qi], retriever, int_to_record_id, hs_map, K
        )
        eval_retrieved_ids.append(passing_ids)
        eval_details.append({
            "query": q,
            "retrieved_ids": passing_ids,
            "scores": passing_scores,
            "n_candidates_fetched": K + DELTA,
            "n_filtered_by_denylist": n_filtered,
            "n_final_results": len(passing_ids),
            "has_poisoned": any(rid in all_poisoned_ids for rid in passing_ids),
        })

    prp3 = prp_at_k(eval_retrieved_ids, all_poisoned_ids, K)
    print(f"\n  PRP@{K} = {prp3:.4f}")

    n_queries_with_poison = sum(1 for d in eval_details if d["has_poisoned"])
    total_filtered = sum(d["n_filtered_by_denylist"] for d in eval_details)
    total_final = sum(d["n_final_results"] for d in eval_details)
    print(f"  Total candidates filtered across all queries: {total_filtered}")
    print(f"  Total final results returned: {total_final} / {K * len(queries)} possible")

    print(f"\nComputing Benign Recall@{K} with fuzzy retrieval-time filtering...")
    benign_query_embeddings = embedder.encode_queries(
        [r["req"] for r in benign], cache_prefix="benign_req_queries"
    )

    benign_retrieved_ids = []
    benign_filter_stats = {"total_filtered": 0, "total_queries": len(benign)}
    for bi in range(len(benign)):
        passing_ids, _, n_filtered = denylist.filter_results(
            benign_query_embeddings[bi], retriever, int_to_record_id, hs_map, K
        )
        benign_retrieved_ids.append(passing_ids)
        benign_filter_stats["total_filtered"] += n_filtered

    br3 = benign_recall_at_k(benign_retrieved_ids, benign_ids, K)
    print(f"  Benign Recall@{K} = {br3:.4f}")
    print(f"  Total benign-query candidates filtered: {benign_filter_stats['total_filtered']}")

    results = {
        "condition": "condition_b_retrieval_time_denylist_fuzzy",
        "matching_strategy": "hazard_set_containment",
        "description": (
            f"Retrieval-time HS denylist + backfill with fuzzy matching. "
            f"Retrieve top-{K+DELTA}, filter by hazard set containment against tombstones, "
            f"return top-{K}."
        ),
        "k": K,
        "delta": DELTA,
        "n_benign": len(benign),
        "n_poisoned_original": len(poisoned),
        "n_paraphrases": len(paraphrases),
        "n_total_in_index": retriever.ntotal,
        "n_eval_queries": len(queries),
        "n_unique_tombstone_hs": len(tombstone_hs_set),
        "tombstone_hs_values": sorted(tombstone_hs_set),
        "prp_at_3": round(prp3, 4),
        "benign_recall_at_3": round(br3, 4),
        "write_block_rate": 0.0,
        "benign_false_block": 0.0,
        "n_queries_with_poisoned_in_top3": n_queries_with_poison,
        "total_candidates_filtered_eval": total_filtered,
        "total_final_results_eval": total_final,
        "benign_query_filter_stats": benign_filter_stats,
        "eval_query_details": eval_details,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "condition_b_optimized_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n=== Condition B Optimized Summary ===")
    print(f"  Matching strategy:         hazard_set_containment")
    print(f"  PRP@{K}:                  {prp3:.4f}")
    print(f"  Benign Recall@{K}:         {br3:.4f}")
    print(f"  WriteBlockRate:            0.00%")
    print(f"  Benign FalseBlock:         0.00%")
    print(f"  Queries with poison:       {n_queries_with_poison}/{len(queries)}")
    print(f"  Total filtered (eval):     {total_filtered}")
    print(f"  Index size:                {retriever.ntotal}")
    print(f"  Tombstone set size:        {len(tombstone_hs_set)}")


if __name__ == "__main__":
    main()
