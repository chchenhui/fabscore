#!/usr/bin/env python3
import scanpy as sc
import pandas as pd
import numpy as np

print("Loading SCENIC results file...")
adata = sc.read_h5ad('entero_hg38_scenic_full_results.h5ad')

print("\n=== DATASET OVERVIEW ===")
print(f"Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

print("\n=== CELL METADATA (adata.obs) ===")
print(f"Number of columns: {len(adata.obs.columns)}")
print(f"Columns: {list(adata.obs.columns[:20])}")
if len(adata.obs.columns) > 20:
    print(f"... and {len(adata.obs.columns) - 20} more columns")

print("\n=== REGULON ACTIVITIES ===")
regulon_cols = [col for col in adata.obs.columns if '(+)' in col or '(-)' in col]
print(f"Number of regulons found: {len(regulon_cols)}")
if regulon_cols:
    print(f"First 10 regulons: {regulon_cols[:10]}")

print("\n=== KEY METADATA COLUMNS ===")
key_cols = ['cell_type', 'batch', 'donor', 'disease', 'tissue', 'n_genes', 'n_counts']
for col in key_cols:
    if col in adata.obs.columns:
        if adata.obs[col].dtype == 'object' or adata.obs[col].dtype == 'category':
            unique_vals = adata.obs[col].nunique()
            print(f"{col}: {unique_vals} unique values")
            if unique_vals <= 10:
                print(f"  Values: {adata.obs[col].unique()[:10].tolist()}")
        else:
            print(f"{col}: min={adata.obs[col].min():.2f}, max={adata.obs[col].max():.2f}, mean={adata.obs[col].mean():.2f}")

print("\n=== GENE INFORMATION (adata.var) ===")
print(f"Gene var columns: {list(adata.var.columns)}")
print(f"First 10 genes: {list(adata.var.index[:10])}")

print("\n=== EMBEDDINGS (adata.obsm) ===")
if adata.obsm:
    for key in adata.obsm.keys():
        print(f"{key}: shape {adata.obsm[key].shape}")

print("\n=== ADDITIONAL SLOTS ===")
if adata.uns:
    print(f"adata.uns keys: {list(adata.uns.keys())[:10]}")
if adata.obsp:
    print(f"adata.obsp keys: {list(adata.obsp.keys())}")
if adata.varp:
    print(f"adata.varp keys: {list(adata.varp.keys())}")

print("\n=== REGULON ACTIVITY STATISTICS ===")
if regulon_cols:
    regulon_activities = adata.obs[regulon_cols].values
    print(f"Activity range: [{np.min(regulon_activities):.4f}, {np.max(regulon_activities):.4f}]")
    print(f"Mean activity: {np.mean(regulon_activities):.4f}")
    print(f"Median activity: {np.median(regulon_activities):.4f}")
    
    print("\n=== TOP 10 MOST ACTIVE REGULONS (by mean activity) ===")
    mean_activities = adata.obs[regulon_cols].mean().sort_values(ascending=False)
    for i, (reg, val) in enumerate(mean_activities.head(10).items(), 1):
        print(f"{i}. {reg}: {val:.4f}")