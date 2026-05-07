#!/usr/bin/env python
import scanpy as sc
import pandas as pd

# Load the h5ad file
adata = sc.read_h5ad('../sub_adata/strict_endocrine.h5ad')

print("Data shape:", adata.shape)
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")
print("\nData structure:")
print(f"X matrix type: {type(adata.X)}")
print(f"X matrix shape: {adata.X.shape}")

print("\nObs columns:", list(adata.obs.columns))
print("\nVar columns:", list(adata.var.columns))

# Check if data is normalized/log-transformed
import numpy as np
if hasattr(adata.X, 'data'):
    sample_values = adata.X.data[:100]
else:
    sample_values = adata.X.flatten()[:100]
    
print(f"\nSample values range: [{sample_values.min():.3f}, {sample_values.max():.3f}]")
print(f"Data appears to be log-transformed: {sample_values.min() >= 0 and sample_values.max() < 100}")