#!/usr/bin/env python3

import scanpy as sc
import pandas as pd

print("Checking SCENIC results structure...")

adata = sc.read_h5ad('entero_hg38_scenic_full_results.h5ad')
print(f"\nData shape: {adata.shape}")

print("\n=== Available data slots ===")
print(f"obs columns: {list(adata.obs.columns)[:10]}")
print(f"Total obs columns: {len(adata.obs.columns)}")

print(f"\nvar columns: {list(adata.var.columns)}")
print(f"\nobsm keys: {list(adata.obsm.keys())}")
print(f"\nvarm keys: {list(adata.varm.keys())}")
print(f"\nuns keys: {list(adata.uns.keys())}")

regulon_cols = [col for col in adata.obs.columns if '(+)' in col or '(-)' in col]
print(f"\n=== Regulon columns ===")
print(f"Found {len(regulon_cols)} regulon columns")
if regulon_cols:
    print(f"First 10 regulons: {regulon_cols[:10]}")
    
arid_cols = [col for col in adata.obs.columns if 'Arid' in col or 'ARID' in col]
print(f"\n=== Checking for Arid5b ===")
print(f"Columns containing 'Arid': {arid_cols}")

print("\n=== Checking all obs columns for patterns ===")
for col in adata.obs.columns[:20]:
    print(f"  {col}")