# Condition C: Commit-time HS tombstone lockout.
# After deleting poisoned records, tombstones their hazard signatures.
# Paraphrase re-injection is blocked at commit time if HS matches a tombstone.
# Final index: 100 benign + only accepted paraphrases (those whose HS != tombstone).
#
# Outputs: hst/outputs/condition_c_results.json

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
from hst.memory.policies import CommitTimeTombstone
from hst.memory.store import MemoryStore
from hst.evaluation.metrics import prp_at_k, benign_recall_at_k

OUTPUT_DIR = PROJECT_ROOT / "outputs"
K = 3


def main():
    print("=== Condition C: Commit-Time HS Tombstone Lockout ===")

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

    print("\nBuilding tombstone set from 10 original poisoned records...")
    tombstone_set = set()
    for r in poisoned:
        hs = signatures[r["id"]]["hs"]
        tombstone_set.add(hs)
        print(f"  {r['id']}: {hs}")
    print(f"  Unique tombstone HS values: {len(tombstone_set)}")

    print("\nLoading cached embeddings for 110 original records...")
    embedder = Embedder(model_name="BAAI/bge-m3")
    all_embeddings = embedder.encode(all_original, cache_prefix="all_110")
    print(f"  Embedding shape: {all_embeddings.shape}")

    print("Building FAISS index with all 110 records...")
    retriever = FaissRetriever(dim=all_embeddings.shape[1])
    int_ids = np.arange(len(all_original), dtype=np.int64)
    retriever.build_index(all_embeddings, ids=int_ids)

    int_to_record_id = {i: original_ids[i] for i in range(len(original_ids))}
    policy = CommitTimeTombstone(tombstone_set=tombstone_set)
    store = MemoryStore(retriever, policy, int_to_record_id, signatures)

    print("Deleting 10 poisoned records by ID...")
    poisoned_record_ids = [r["id"] for r in poisoned]
    deleted_tombstone = store.delete_records(poisoned_record_ids)
    print(f"  Index size after deletion: {store.retriever.ntotal} (expected 100)")
    print(f"  Tombstone HS from deletion: {deleted_tombstone}")
    assert deleted_tombstone == tombstone_set

    print(f"\nAttempting to add {len(paraphrases)} paraphrased records with commit-time check...")
    para_embeddings = embedder.encode(paraphrases, cache_prefix="paraphrases_50")
    for i, para in enumerate(paraphrases):
        accepted = store.add_record(para["id"], para_embeddings[i])
        status = "ACCEPTED" if accepted else "BLOCKED"
        para_hs = signatures.get(para["id"], {}).get("hs", "N/A")
        print(f"  [{i+1:2d}/50] {para['id']}: {status} (HS: {para_hs})")

    n_blocked = len(store.blocked_writes)
    n_accepted = len(store.accepted_writes)
    write_block_rate = n_blocked / len(paraphrases) if paraphrases else 0.0

    print(f"\n  Blocked: {n_blocked}/50 ({write_block_rate:.1%})")
    print(f"  Accepted: {n_accepted}/50")
    print(f"  Final index size: {store.retriever.ntotal} (expected {100 + n_accepted})")

    print(f"\nEncoding {len(queries)} evaluation queries...")
    query_embeddings = embedder.encode_queries(queries, cache_prefix="eval_queries")

    print(f"Retrieving top-{K} for each eval query from final index...")
    eval_retrieved_ids = []
    eval_details = []
    for qi, q in enumerate(queries):
        top_ids, top_scores = store.query(query_embeddings[qi], k=K)
        eval_retrieved_ids.append(top_ids)
        eval_details.append({
            "query": q,
            "retrieved_ids": top_ids,
            "scores": top_scores,
            "has_poisoned": any(rid in all_poisoned_ids for rid in top_ids),
        })

    prp3 = prp_at_k(eval_retrieved_ids, all_poisoned_ids, K)
    n_queries_with_poison = sum(1 for d in eval_details if d["has_poisoned"])
    print(f"\n  PRP@{K} = {prp3:.4f}")
    print(f"  Queries with poison in top-{K}: {n_queries_with_poison}/{len(queries)}")

    print(f"\nComputing Benign Recall@{K}...")
    benign_query_embeddings = embedder.encode_queries(
        [r["req"] for r in benign], cache_prefix="benign_req_queries"
    )

    benign_retrieved_ids = []
    for bi in range(len(benign)):
        top_ids, _ = store.query(benign_query_embeddings[bi], k=K)
        benign_retrieved_ids.append(top_ids)

    br3 = benign_recall_at_k(benign_retrieved_ids, benign_ids, K)
    print(f"  Benign Recall@{K} = {br3:.4f}")

    print(f"\nComputing Benign FalseBlock...")
    benign_false_blocked = []
    for r in benign:
        rid = r["id"]
        benign_hs = signatures.get(rid, {}).get("hs", "")
        if benign_hs in tombstone_set:
            benign_false_blocked.append(rid)
    benign_false_block_rate = len(benign_false_blocked) / len(benign) if benign else 0.0
    print(f"  Benign FalseBlock = {benign_false_block_rate:.4f} ({len(benign_false_blocked)}/{len(benign)})")
    if benign_false_blocked:
        print(f"  Falsely blocked benign IDs: {benign_false_blocked}")

    results = {
        "condition": "condition_c_commit_time_tombstone",
        "description": (
            "Commit-time HS tombstone lockout. Delete 10 poisoned IDs, "
            "tombstone their HS. Block paraphrase re-injection at commit time "
            "if HS matches tombstone."
        ),
        "k": K,
        "n_benign": len(benign),
        "n_poisoned_original": len(poisoned),
        "n_paraphrases": len(paraphrases),
        "n_total_in_index": store.retriever.ntotal,
        "n_eval_queries": len(queries),
        "n_unique_tombstone_hs": len(tombstone_set),
        "tombstone_hs_values": sorted(tombstone_set),
        "write_block_rate": round(write_block_rate, 4),
        "n_blocked": n_blocked,
        "n_accepted": n_accepted,
        "blocked_paraphrase_ids": store.blocked_writes,
        "accepted_paraphrase_ids": store.accepted_writes,
        "prp_at_3": round(prp3, 4),
        "benign_recall_at_3": round(br3, 4),
        "benign_false_block": round(benign_false_block_rate, 4),
        "benign_false_blocked_ids": benign_false_blocked,
        "n_queries_with_poisoned_in_top3": n_queries_with_poison,
        "eval_query_details": eval_details,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "condition_c_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\n=== Condition C Summary ===")
    print(f"  PRP@{K}:                  {prp3:.4f}")
    print(f"  Benign Recall@{K}:         {br3:.4f}")
    print(f"  WriteBlockRate:            {write_block_rate:.2%}")
    print(f"  Benign FalseBlock:         {benign_false_block_rate:.2%}")
    print(f"  Blocked writes:            {n_blocked}/50")
    print(f"  Accepted writes:           {n_accepted}/50")
    print(f"  Queries with poison:       {n_queries_with_poison}/{len(queries)}")
    print(f"  Index size:                {store.retriever.ntotal}")
    print(f"  Tombstone set size:        {len(tombstone_set)}")


if __name__ == "__main__":
    main()
