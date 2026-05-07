"""Constant-motion baseline: predict zero PCA coefficients (mean face) for every frame.
Reconstructs to pca_model['original_data_mean'] for all frames, computes LVE against GT.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scaffoldswap.data.biwi.dataset import BIWIDataset, biwi_collate_fn
from scaffoldswap.evaluate import inverse_pca, compute_lve, compute_mve, compute_ufve, compute_fdd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    pca_model = torch.load(os.path.join(args.data_dir, "pca_model.pt"), weights_only=False)
    for k in pca_model:
        pca_model[k] = pca_model[k].to(device)

    regions = torch.load(os.path.join(args.data_dir, "region_indices.pt"), weights_only=False)
    lip_indices = regions["lip"]
    upper_face_indices = regions["upper_face"]

    test_ds = BIWIDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             collate_fn=biwi_collate_fn, num_workers=2)

    all_lve, all_mve, all_ufve, all_fdd = [], [], [], []
    pca_dim = pca_model["components_to_data"].shape[0]

    for batch in test_loader:
        pca_target = batch["pca_target"].to(device)
        n_frames_list = batch["n_frames"]

        for i in range(pca_target.shape[0]):
            T = n_frames_list[i]
            gt_pca = pca_target[i, :T]

            pred_pca = torch.zeros_like(gt_pca)

            pred_verts = inverse_pca(pred_pca, pca_model)
            gt_verts = inverse_pca(gt_pca, pca_model)

            lve = compute_lve(pred_verts, gt_verts, lip_indices)
            mve = compute_mve(pred_verts, gt_verts)
            ufve = compute_ufve(pred_verts, gt_verts, upper_face_indices)
            fdd = compute_fdd(pred_verts, gt_verts, upper_face_indices) if T > 1 else 0.0

            all_lve.append(lve)
            all_mve.append(mve)
            all_ufve.append(ufve)
            all_fdd.append(fdd)

    results = {
        "baseline": "constant_motion_mean_face",
        "dataset": "biwi",
        "LVE": float(np.mean(all_lve)),
        "MVE": float(np.mean(all_mve)),
        "UFVE": float(np.mean(all_ufve)),
        "FDD": float(np.mean(all_fdd)),
        "LVE_std": float(np.std(all_lve)),
        "n_test_sequences": len(all_lve),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nConstant-motion baseline results:")
    print(f"  LVE:  {results['LVE']:.6f}")
    print(f"  MVE:  {results['MVE']:.6f}")
    print(f"  UFVE: {results['UFVE']:.6f}")
    print(f"  FDD:  {results['FDD']:.6f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
