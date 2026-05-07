# Alternative embedding ablation: re-run Conditions A, B (fuzzy), C (fuzzy)
# with Qwen/Qwen3-Embedding-4B to verify that the relative ranking
# C > B > A is embedding-backend-agnostic.
# Outputs: hst/outputs/ablation_alt_embedding_results.json

import argparse
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
from hst.memory.policies import (
    RetrievalTimeDenylistFuzzy,
    CommitTimeTombstoneFuzzy,
    _parse_tombstone_hazards,
    _fuzzy_hazard_match,
)
from hst.memory.store import MemoryStore

OUTPUT_DIR = PROJECT_ROOT / "outputs"
K = 3
DELTA = 6
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"


def run_condition_a(embedder, benign, poisoned, paraphrases, queries,
                    all_original, original_ids, poisoned_id_set,
                    benign_ids, para_ids, all_poisoned_ids):
    print("\n========== Condition A: ID-Delete Only (Qwen3) ==========")
    all_embeddings = embedder.encode(all_original, cache_prefix="qwen3_all_110")
    dim = all_embeddings.shape[1]
    print(f"  Embedding dim: {dim}")

    retriever = FaissRetriever(dim=dim)
    int_ids = np.arange(len(all_original), dtype=np.int64)
    retriever.build_index(all_embeddings, ids=int_ids)

    poisoned_int_ids = [i for i, r in enumerate(all_original) if r["id"] in poisoned_id_set]
    retriever.remove(poisoned_int_ids)
    print(f"  Index after deletion: {retriever.ntotal}")

    para_embeddings = embedder.encode(paraphrases, cache_prefix="qwen3_paraphrases_50")
    para_int_ids = np.arange(len(all_original), len(all_original) + len(paraphrases), dtype=np.int64)
    retriever.add(para_embeddings, ids=para_int_ids)
    print(f"  Index after adding paraphrases: {retriever.ntotal}")

    combined_id_list = original_ids + para_ids

    query_embeddings = embedder.encode_queries(queries, cache_prefix="qwen3_eval_queries")
    distances, indices = retriever.search(query_embeddings, k=K)

    eval_retrieved_ids = []
    for qi in range(len(queries)):
        top_ids = [combined_id_list[int(idx)] for idx in indices[qi] if idx >= 0]
        eval_retrieved_ids.append(top_ids)

    prp3 = prp_at_k(eval_retrieved_ids, all_poisoned_ids, K)

    benign_query_embeddings = embedder.encode_queries(
        [r["req"] for r in benign], cache_prefix="qwen3_benign_req_queries"
    )
    _, benign_indices = retriever.search(benign_query_embeddings, k=K)
    benign_retrieved_ids = []
    for bi in range(len(benign)):
        top_ids = [combined_id_list[int(idx)] for idx in benign_indices[bi] if idx >= 0]
        benign_retrieved_ids.append(top_ids)
    br3 = benign_recall_at_k(benign_retrieved_ids, benign_ids, K)

    print(f"  PRP@{K} = {prp3:.4f}")
    print(f"  Benign Recall@{K} = {br3:.4f}")
    return {"prp_at_3": round(prp3, 4), "benign_recall_at_3": round(br3, 4),
            "write_block_rate": 0.0, "benign_false_block": 0.0}


def run_condition_b(embedder, benign, poisoned, paraphrases, queries, signatures,
                    all_original, original_ids, poisoned_id_set,
                    benign_ids, para_ids, all_poisoned_ids, tombstone_sigs):
    print("\n========== Condition B: Retrieval-Time Fuzzy Denylist (Qwen3) ==========")
    all_embeddings = embedder.encode(all_original, cache_prefix="qwen3_all_110")
    dim = all_embeddings.shape[1]

    retriever = FaissRetriever(dim=dim)
    int_ids = np.arange(len(all_original), dtype=np.int64)
    retriever.build_index(all_embeddings, ids=int_ids)

    poisoned_int_ids = [i for i, r in enumerate(all_original) if r["id"] in poisoned_id_set]
    retriever.remove(poisoned_int_ids)

    para_embeddings = embedder.encode(paraphrases, cache_prefix="qwen3_paraphrases_50")
    para_int_ids = np.arange(len(all_original), len(all_original) + len(paraphrases), dtype=np.int64)
    retriever.add(para_embeddings, ids=para_int_ids)
    print(f"  Index size: {retriever.ntotal}")

    combined_id_list = original_ids + para_ids
    int_to_record_id = {i: combined_id_list[i] for i in range(len(combined_id_list))}

    hs_map = {}
    for rid in combined_id_list:
        hs_map[rid] = signatures.get(rid, {})

    denylist = RetrievalTimeDenylistFuzzy(tombstone_sigs=tombstone_sigs, delta=DELTA)

    query_embeddings = embedder.encode_queries(queries, cache_prefix="qwen3_eval_queries")
    eval_retrieved_ids = []
    for qi in range(len(queries)):
        passing_ids, _, _ = denylist.filter_results(
            query_embeddings[qi], retriever, int_to_record_id, hs_map, K
        )
        eval_retrieved_ids.append(passing_ids)

    prp3 = prp_at_k(eval_retrieved_ids, all_poisoned_ids, K)

    benign_query_embeddings = embedder.encode_queries(
        [r["req"] for r in benign], cache_prefix="qwen3_benign_req_queries"
    )
    benign_retrieved_ids = []
    for bi in range(len(benign)):
        passing_ids, _, _ = denylist.filter_results(
            benign_query_embeddings[bi], retriever, int_to_record_id, hs_map, K
        )
        benign_retrieved_ids.append(passing_ids)
    br3 = benign_recall_at_k(benign_retrieved_ids, benign_ids, K)

    print(f"  PRP@{K} = {prp3:.4f}")
    print(f"  Benign Recall@{K} = {br3:.4f}")
    return {"prp_at_3": round(prp3, 4), "benign_recall_at_3": round(br3, 4),
            "write_block_rate": 0.0, "benign_false_block": 0.0}


def run_condition_c(embedder, benign, poisoned, paraphrases, queries, signatures,
                    all_original, original_ids, poisoned_id_set,
                    benign_ids, para_ids, all_poisoned_ids, tombstone_sigs):
    print("\n========== Condition C: Commit-Time Fuzzy Tombstone (Qwen3) ==========")
    all_embeddings = embedder.encode(all_original, cache_prefix="qwen3_all_110")
    dim = all_embeddings.shape[1]

    retriever = FaissRetriever(dim=dim)
    int_ids = np.arange(len(all_original), dtype=np.int64)
    retriever.build_index(all_embeddings, ids=int_ids)

    int_to_record_id = {i: original_ids[i] for i in range(len(original_ids))}
    policy = CommitTimeTombstoneFuzzy(tombstone_sigs=tombstone_sigs)
    store = MemoryStore(retriever, policy, int_to_record_id, signatures)

    poisoned_record_ids = [r["id"] for r in poisoned]
    store.delete_records_structured(poisoned_record_ids)
    print(f"  Index after deletion: {store.retriever.ntotal}")

    para_embeddings = embedder.encode(paraphrases, cache_prefix="qwen3_paraphrases_50")
    for i, para in enumerate(paraphrases):
        accepted = store.add_record(para["id"], para_embeddings[i])
        status = "ACCEPTED" if accepted else "BLOCKED"
        print(f"  [{i+1:2d}/{len(paraphrases)}] {para['id']}: {status}")

    n_blocked = len(store.blocked_writes)
    n_accepted = len(store.accepted_writes)
    wbr = n_blocked / len(paraphrases) if paraphrases else 0.0
    print(f"  Blocked: {n_blocked}/{len(paraphrases)}, Accepted: {n_accepted}")
    print(f"  Final index size: {store.retriever.ntotal}")

    query_embeddings = embedder.encode_queries(queries, cache_prefix="qwen3_eval_queries")
    eval_retrieved_ids = []
    for qi in range(len(queries)):
        top_ids, _ = store.query(query_embeddings[qi], k=K)
        eval_retrieved_ids.append(top_ids)

    prp3 = prp_at_k(eval_retrieved_ids, all_poisoned_ids, K)

    benign_query_embeddings = embedder.encode_queries(
        [r["req"] for r in benign], cache_prefix="qwen3_benign_req_queries"
    )
    benign_retrieved_ids = []
    for bi in range(len(benign)):
        top_ids, _ = store.query(benign_query_embeddings[bi], k=K)
        benign_retrieved_ids.append(top_ids)
    br3 = benign_recall_at_k(benign_retrieved_ids, benign_ids, K)

    tomb_hz_sets = _parse_tombstone_hazards(tombstone_sigs)
    benign_false_blocked = []
    for r in benign:
        benign_sig = signatures.get(r["id"], {})
        benign_hz = frozenset(benign_sig.get("hazards", []))
        if _fuzzy_hazard_match(benign_hz, tomb_hz_sets):
            benign_false_blocked.append(r["id"])
    bfb = len(benign_false_blocked) / len(benign) if benign else 0.0

    print(f"  PRP@{K} = {prp3:.4f}")
    print(f"  Benign Recall@{K} = {br3:.4f}")
    print(f"  WriteBlockRate = {wbr:.2%}")
    print(f"  Benign FalseBlock = {bfb:.2%}")
    return {"prp_at_3": round(prp3, 4), "benign_recall_at_3": round(br3, 4),
            "write_block_rate": round(wbr, 4), "benign_false_block": round(bfb, 4),
            "n_blocked": n_blocked, "n_accepted": n_accepted,
            "benign_false_blocked_ids": benign_false_blocked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with tiny subset (3 records) for verification")
    args = parser.parse_args()

    print("=== Alternative Embedding Ablation: Qwen/Qwen3-Embedding-4B ===")
    print(f"  Model: {MODEL_NAME}")
    if args.dry_run:
        print("  *** DRY-RUN MODE: tiny subset ***")

    benign, poisoned = load_seeds()
    paraphrases = load_paraphrases()
    queries = load_eval_queries()
    signatures = load_signatures()

    if args.dry_run:
        benign = benign[:2]
        poisoned = poisoned[:1]
        paraphrases = [p for p in paraphrases if p["id"].startswith(poisoned[0]["id"].replace("exp_poisoned_", "para_p"))][:1]
        if not paraphrases:
            paraphrases = load_paraphrases()[:1]
        queries = queries[:2]

    all_original = benign + poisoned
    original_ids = [r["id"] for r in all_original]
    poisoned_id_set = {r["id"] for r in poisoned}
    benign_ids = [r["id"] for r in benign]
    para_ids = [p["id"] for p in paraphrases]
    all_poisoned_ids = poisoned_id_set | set(para_ids)

    print(f"  Benign: {len(benign)}, Poisoned: {len(poisoned)}")
    print(f"  Paraphrases: {len(paraphrases)}, Eval queries: {len(queries)}")

    tombstone_sigs = []
    for r in poisoned:
        sig = signatures[r["id"]]
        tombstone_sigs.append(sig)

    embedder = Embedder(model_name=MODEL_NAME)

    result_a = run_condition_a(
        embedder, benign, poisoned, paraphrases, queries,
        all_original, original_ids, poisoned_id_set, benign_ids, para_ids, all_poisoned_ids
    )
    result_b = run_condition_b(
        embedder, benign, poisoned, paraphrases, queries, signatures,
        all_original, original_ids, poisoned_id_set, benign_ids, para_ids, all_poisoned_ids,
        tombstone_sigs
    )
    result_c = run_condition_c(
        embedder, benign, poisoned, paraphrases, queries, signatures,
        all_original, original_ids, poisoned_id_set, benign_ids, para_ids, all_poisoned_ids,
        tombstone_sigs
    )

    results = {
        "ablation": "alt_embedding_backend",
        "embedding_model": MODEL_NAME,
        "embedding_dim": 2560,
        "matching_strategy": "hazard_set_containment",
        "k": K,
        "delta": DELTA,
        "n_benign": len(benign),
        "n_poisoned_original": len(poisoned),
        "n_paraphrases": len(paraphrases),
        "n_eval_queries": len(queries),
        "dry_run": args.dry_run,
        "condition_a": result_a,
        "condition_b": result_b,
        "condition_c": result_c,
    }

    if not args.dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "ablation_alt_embedding_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    print("\n=== Alt Embedding Ablation Summary ===")
    for cond_name, cond_res in [("A", result_a), ("B", result_b), ("C", result_c)]:
        print(f"  Condition {cond_name}: PRP@3={cond_res['prp_at_3']:.4f}, "
              f"Benign Recall@3={cond_res['benign_recall_at_3']:.4f}")

    return results


if __name__ == "__main__":
    main()
