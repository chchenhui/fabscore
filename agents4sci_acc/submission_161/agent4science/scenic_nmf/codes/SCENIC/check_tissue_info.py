#!/usr/bin/env python3

import scanpy as sc
import pandas as pd

print("Checking tissue information in h5ad file...")

adata = sc.read_h5ad('entero_hg38_scenic_full_results.h5ad')
print(f"Data shape: {adata.shape}")

# Check for tissue-related columns
tissue_cols = [col for col in adata.obs.columns if 'tissue' in col.lower()]
print(f"\nTissue-related columns: {tissue_cols}")

# Check each tissue column
for col in tissue_cols:
    print(f"\n=== {col} ===")
    value_counts = adata.obs[col].value_counts()
    print(f"Number of unique values: {len(value_counts)}")
    print(f"Top 20 values:")
    print(value_counts.head(20).to_string())
    
# Also check for organ-related columns
organ_cols = [col for col in adata.obs.columns if 'organ' in col.lower()]
print(f"\n\nOrgan-related columns: {organ_cols}")

for col in organ_cols:
    print(f"\n=== {col} ===")
    value_counts = adata.obs[col].value_counts()
    print(f"Number of unique values: {len(value_counts)}")
    print(f"Top 10 values:")
    print(value_counts.head(10).to_string())