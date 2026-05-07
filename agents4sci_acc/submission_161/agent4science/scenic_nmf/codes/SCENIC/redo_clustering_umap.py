#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white')

print(f"Starting SCENIC-based clustering and UMAP at {datetime.now()}")

# Load the updated h5ad file
print("\nLoading h5ad file...")
adata = sc.read_h5ad('entero_hg38_scenic_full_results_with_tissue_combined.h5ad')
print(f"Loaded data with shape: {adata.shape}")

# Extract regulon activity matrix
print("\nExtracting regulon activity matrix...")
regulon_columns = [col for col in adata.obs.columns if '(+)' in col or '(-)' in col]
print(f"Found {len(regulon_columns)} regulons")

# Create a new AnnData object with regulon activities as the main matrix
regulon_matrix = adata.obs[regulon_columns].values
adata_scenic = sc.AnnData(X=regulon_matrix)
adata_scenic.obs_names = adata.obs_names
adata_scenic.var_names = regulon_columns

# Copy metadata
print("\nCopying metadata...")
metadata_cols = ['tissue', 'tissue_combined', 'cell_type', 'disease', 'inferred_cell_type', 
                 'donor_id', 'sex_ontology_term_id', 'donor_age', 'disease_ontology_term_id']
for col in metadata_cols:
    if col in adata.obs.columns:
        adata_scenic.obs[col] = adata.obs[col].values
        print(f"  Added {col}")

# Perform PCA on regulon activities
print("\nPerforming PCA on regulon activities...")
sc.tl.pca(adata_scenic, n_comps=50)

# Compute neighborhood graph
print("\nComputing neighborhood graph...")
sc.pp.neighbors(adata_scenic, n_neighbors=30, n_pcs=30, metric='euclidean')

# Perform UMAP embedding
print("\nComputing UMAP embedding...")
sc.tl.umap(adata_scenic, min_dist=0.3, spread=1.0)

# Perform Louvain clustering at multiple resolutions
print("\nPerforming Louvain clustering...")
resolutions = [0.5, 1.0, 1.5, 2.0]
for res in resolutions:
    sc.tl.louvain(adata_scenic, resolution=res, key_added=f'louvain_{res}')
    print(f"  Resolution {res}: {len(adata_scenic.obs[f'louvain_{res}'].unique())} clusters")

# Use resolution 1.0 as default
adata_scenic.obs['louvain'] = adata_scenic.obs['louvain_1.0']

# Store the new UMAP coordinates back to the original adata
print("\nTransferring SCENIC UMAP to original AnnData...")
adata.obsm['X_umap_scenic'] = adata_scenic.obsm['X_umap'].copy()

# Also store clustering results
for res in resolutions:
    adata.obs[f'scenic_louvain_{res}'] = adata_scenic.obs[f'louvain_{res}'].values
adata.obs['scenic_louvain'] = adata_scenic.obs['louvain'].values

# Save both updated files
print("\nSaving files...")
adata_scenic.write('entero_scenic_clustering_adata.h5ad')
print("  Saved SCENIC clustering AnnData to: entero_scenic_clustering_adata.h5ad")

adata.write('entero_hg38_scenic_full_results_with_clustering.h5ad')
print("  Saved full AnnData with SCENIC UMAP to: entero_hg38_scenic_full_results_with_clustering.h5ad")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Total cells: {adata_scenic.n_obs}")
print(f"Total regulons: {adata_scenic.n_vars}")
print(f"Number of clusters (resolution 1.0): {len(adata_scenic.obs['louvain'].unique())}")

# Show cluster distribution
print("\n=== Cluster Distribution (resolution 1.0) ===")
cluster_counts = adata_scenic.obs['louvain'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    percentage = (count / adata_scenic.n_obs) * 100
    print(f"  Cluster {cluster}: {count} cells ({percentage:.1f}%)")

print(f"\nCompleted at {datetime.now()}")