"""Prepare MSMARCO-10K: embed passages, build HNSW indices, compute true kNN.

Outputs:
  outputs/msmarco10k/vectors.npy       -- (10000, 768) float32 embeddings
  outputs/msmarco10k/passage_ids.json  -- sampled passage indices
  outputs/msmarco10k/adj_seed{seed}.pkl -- (adjacency_list, perm) per seed
  outputs/msmarco10k/true_knn_k20.npy  -- true 20-NN for all 10K vectors
"""

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.msmarco10k import load_msmarco10k
from evaluation.metrics import compute_true_knn
from graph.hnsw_builder import build_hnsw_index, extract_layer0_adjacency

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "msmarco10k")
SEEDS = [42, 123, 456]
M = 32
EF_CONSTRUCTION = 64
KNN_K = 20


def remap_adjacency(adjacency_list: dict, perm: np.ndarray) -> dict:
    remapped = {}
    for internal_id in range(len(perm)):
        orig_id = int(perm[internal_id])
        neighbors_internal = adjacency_list[internal_id]
        neighbors_orig = [int(perm[n]) for n in neighbors_internal]
        remapped[orig_id] = neighbors_orig
    return remapped


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vectors = load_msmarco10k(seed=42)
    print(f"Loaded MSMARCO-10K: {vectors.shape}")

    for seed in SEEDS:
        pkl_path = os.path.join(OUTPUT_DIR, f"adj_seed{seed}.pkl")
        if os.path.exists(pkl_path):
            print(f"  Seed {seed}: already exists, skipping.")
            continue

        print(f"  Building HNSW index (M={M}, efConstruction={EF_CONSTRUCTION}, seed={seed})...")
        index, perm = build_hnsw_index(vectors, M=M, efConstruction=EF_CONSTRUCTION, seed=seed)
        print(f"  Extracting layer-0 adjacency...")
        adjacency_list, degrees = extract_layer0_adjacency(index)

        remapped_adj = remap_adjacency(adjacency_list, perm)

        with open(pkl_path, "wb") as f:
            pickle.dump({"adjacency_list": remapped_adj, "perm": perm}, f)
        print(f"  Saved {pkl_path}")

        deg_vals = [len(remapped_adj[i]) for i in range(len(remapped_adj))]
        print(f"  Degree stats: min={min(deg_vals)}, max={max(deg_vals)}, mean={np.mean(deg_vals):.1f}")

    knn_path = os.path.join(OUTPUT_DIR, "true_knn_k20.npy")
    if os.path.exists(knn_path):
        print(f"True kNN already exists at {knn_path}")
    else:
        print(f"Computing true {KNN_K}-NN ground truth...")
        true_knn = compute_true_knn(vectors, k=KNN_K)
        knn_array = np.array([true_knn[i] for i in range(len(true_knn))], dtype=np.int64)
        np.save(knn_path, knn_array)
        print(f"Saved true kNN to {knn_path}, shape={knn_array.shape}")

    print("Done.")


if __name__ == "__main__":
    main()
