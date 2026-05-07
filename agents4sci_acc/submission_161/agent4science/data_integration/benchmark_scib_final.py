#!/usr/bin/env python3
"""
Final scib-metrics benchmarking with correct embeddings and proper method names
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# Set environment for multi-threading
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'

print("="*80)
print("FINAL SCIB-METRICS BENCHMARKING")
print("="*80)

# 1. Load the integrated dataset
print("\n1. Loading integrated dataset...")
adata = sc.read_h5ad('/scratch/rli/project/agent/results/data_integration_2025-08-25/scvi_unique_integration/neuroendocrine_scvi_integrated_unique.h5ad')
print(f"   Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

# 2. Prepare embeddings with proper names
print("\n2. Preparing embeddings with proper method names...")

# Ensure batch and label are categorical
adata.obs["batch"] = adata.obs["batch"].astype("category")
adata.obs["endocrine_type_simple"] = adata.obs["endocrine_type_simple"].astype("category")

# Copy embeddings with clean method names
embedding_keys = []

# 1. Unintegrated (PCA)
if "X_pca_unintegrated" in adata.obsm.keys():
    adata.obsm["Unintegrated"] = adata.obsm["X_pca_unintegrated"].copy()
    embedding_keys.append("Unintegrated")
    print(f"   ✓ Unintegrated: PCA {adata.obsm['Unintegrated'].shape}")
elif "X_pca" in adata.obsm.keys():
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"].copy()
    embedding_keys.append("Unintegrated")
    print(f"   ✓ Unintegrated: PCA {adata.obsm['Unintegrated'].shape}")
else:
    print("   Computing PCA for unintegrated data...")
    adata_tmp = adata.copy()
    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    sc.pp.scale(adata_tmp, max_value=10)
    sc.tl.pca(adata_tmp, n_comps=50, svd_solver='arpack')
    adata.obsm["Unintegrated"] = adata_tmp.obsm["X_pca"].copy()
    embedding_keys.append("Unintegrated")
    del adata_tmp
    print(f"   ✓ Unintegrated: Computed PCA {adata.obsm['Unintegrated'].shape}")

# 2. ComBat (PCA after ComBat correction)
if "X_pca_combat" in adata.obsm.keys():
    adata.obsm["ComBat"] = adata.obsm["X_pca_combat"].copy()
    embedding_keys.append("ComBat")
    print(f"   ✓ ComBat: PCA after correction {adata.obsm['ComBat'].shape}")
elif "combat" in adata.layers.keys():
    print("   Computing PCA for ComBat corrected data...")
    adata_tmp = adata.copy()
    adata_tmp.X = adata.layers["combat"]
    sc.pp.scale(adata_tmp, max_value=10)
    sc.tl.pca(adata_tmp, n_comps=50, svd_solver='arpack')
    adata.obsm["ComBat"] = adata_tmp.obsm["X_pca"].copy()
    embedding_keys.append("ComBat")
    del adata_tmp
    print(f"   ✓ ComBat: Computed PCA {adata.obsm['ComBat'].shape}")
else:
    print("   ⚠ No ComBat corrected data found")

# 3. scVI (latent representation)
if "X_scvi" in adata.obsm.keys():
    adata.obsm["scVI"] = adata.obsm["X_scvi"].copy()
    embedding_keys.append("scVI")
    print(f"   ✓ scVI: Latent representation {adata.obsm['scVI'].shape}")
elif "X_scVI" in adata.obsm.keys():
    adata.obsm["scVI"] = adata.obsm["X_scVI"].copy()
    embedding_keys.append("scVI")
    print(f"   ✓ scVI: Latent representation {adata.obsm['scVI'].shape}")
else:
    print("   ⚠ No scVI latent representation found")

print(f"\n   Methods to benchmark: {embedding_keys}")

if len(embedding_keys) < 2:
    raise ValueError(f"Need at least 2 embeddings for comparison, only found: {embedding_keys}")

# 3. Create subset for faster computation
print("\n3. Creating subset for benchmarking...")
n_subset = 30000
np.random.seed(42)
if len(adata) > n_subset:
    indices = np.random.choice(len(adata), n_subset, replace=False)
    adata_subset = adata[indices].copy()
    print(f"   Using {n_subset:,} cells (from {len(adata):,} total)")
else:
    adata_subset = adata.copy()
    print(f"   Using full dataset ({len(adata):,} cells)")

# 4. Initialize and run benchmarker with proper method names
print("\n4. Setting up Benchmarker...")
print(f"   Batch key: batch")
print(f"   Label key: endocrine_type_simple")
print(f"   Integration methods: {', '.join(embedding_keys)}")
print(f"   Using 32 CPU cores")

bm = Benchmarker(
    adata_subset,
    batch_key="batch",
    label_key="endocrine_type_simple",
    bio_conservation_metrics=BioConservation(),
    batch_correction_metrics=BatchCorrection(),
    embedding_obsm_keys=embedding_keys,
    n_jobs=32
)

print("\n5. Running benchmark...")
print("   Computing metrics for each integration method...")
print("   (This may take a few minutes...)")
bm.benchmark()

print("\n6. Benchmark completed successfully!")
print("="*80)

# 7. Generate result tables using bm.plot_results_table()
output_dir = '/scratch/rli/project/agent/data_integration/results/'

print("\n7. Generating result tables with proper method names...")

# Generate scaled results table
print("\n   A. Generating SCALED results table (0-1 normalized)...")
try:
    bm.plot_results_table(
        save_dir=output_dir,
        min_max_scale=True,
        show=False
    )
    import shutil
    src = os.path.join(output_dir, 'scib_results.svg')
    dst = os.path.join(output_dir, 'scib_results_table_scaled_final.svg')
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"   ✓ Saved as 'scib_results_table_scaled_final.svg'")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Generate raw results table  
print("\n   B. Generating RAW results table...")
try:
    bm.plot_results_table(
        save_dir=output_dir,
        min_max_scale=False,
        show=False
    )
    src = os.path.join(output_dir, 'scib_results.svg')
    dst = os.path.join(output_dir, 'scib_results_table_raw_final.svg')
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"   ✓ Saved as 'scib_results_table_raw_final.svg'")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 8. Save CSV results with proper names
print("\n8. Saving CSV results with proper method names...")

# Get results - they should already have the correct names
results_raw = bm.get_results(min_max_scale=False)
results_scaled = bm.get_results(min_max_scale=True)

# Clean DataFrames - remove Metric Type row if it exists
if 'Metric Type' in results_raw.index:
    results_raw = results_raw.drop('Metric Type')
if 'Metric Type' in results_scaled.index:
    results_scaled = results_scaled.drop('Metric Type')

# The index should already be 'Unintegrated', 'ComBat', 'scVI'
print(f"   Methods in results: {list(results_raw.index)}")

# Save to CSV
results_raw.to_csv(os.path.join(output_dir, 'scib_results_raw_final.csv'))
print("   ✓ Raw results saved")

results_scaled.to_csv(os.path.join(output_dir, 'scib_results_scaled_final.csv'))
print("   ✓ Scaled results saved")

# 9. Display summary with proper names
print("\n9. Results Summary:")
print("="*80)

# Clean up for display
results_scaled_clean = results_scaled.apply(pd.to_numeric, errors='coerce')

# Calculate aggregate scores
bio_metrics = ['Isolated labels', 'KMeans NMI', 'KMeans ARI', 'Silhouette label', 'cLISI']
batch_metrics = ['BRAS', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']

# Filter to existing columns
bio_metrics = [m for m in bio_metrics if m in results_scaled_clean.columns]
batch_metrics = [m for m in batch_metrics if m in results_scaled_clean.columns]

if bio_metrics:
    bio_scores = results_scaled_clean[bio_metrics].mean(axis=1)
    print("\nBiological Conservation (mean of scaled scores):")
    for method, score in bio_scores.items():
        print(f"   {method:12s}: {score:.4f}")

if batch_metrics:
    batch_scores = results_scaled_clean[batch_metrics].mean(axis=1)
    print("\nBatch Correction (mean of scaled scores):")
    for method, score in batch_scores.items():
        print(f"   {method:12s}: {score:.4f}")

overall_scores = results_scaled_clean[bio_metrics + batch_metrics].mean(axis=1)
print("\nOverall Score (mean of all metrics):")
for method, score in overall_scores.items():
    print(f"   {method:12s}: {score:.4f}")

# Find best method
best_overall = overall_scores.idxmax()
print(f"\n🏆 Best Integration Method: {best_overall} (score: {overall_scores[best_overall]:.3f})")

# Create summary DataFrame with proper names
summary_df = pd.DataFrame({
    'Method': overall_scores.index,
    'Bio_Conservation': bio_scores.values if not bio_scores.empty else [0]*len(overall_scores),
    'Batch_Correction': batch_scores.values if not batch_scores.empty else [0]*len(overall_scores),
    'Overall_Score': overall_scores.values
})
summary_df.to_csv(os.path.join(output_dir, 'scib_summary_final.csv'), index=False)
print("\n✓ Saved summary to 'scib_summary_final.csv'")

# Print detailed metrics table
print("\n" + "="*80)
print("DETAILED METRICS (Scaled 0-1)")
print("="*80)

# Create a nice display table
print("\nMetric Type | " + " | ".join([f"{m:^11s}" for m in results_scaled.index[:3]]))
print("-" * (13 + 14 * min(3, len(results_scaled.index))))

for metric in results_scaled.columns[:10]:  # Show first 10 metrics
    if metric in ['Batch correction', 'Bio conservation', 'Total']:
        continue  # Skip aggregate columns
    values = [results_scaled.loc[method, metric] for method in results_scaled.index[:3]]
    metric_short = metric[:11] if len(metric) > 11 else metric
    print(f"{metric_short:11s} | " + " | ".join([f"{v:^11.3f}" for v in values]))

print("\n" + "="*80)
print("BENCHMARKING COMPLETE!")
print("="*80)

print("\n✅ Successfully benchmarked with proper method names:")
print("   • Unintegrated - Raw PCA (50 dimensions)")
print("   • ComBat - PCA after ComBat correction (30 dimensions)")
print("   • scVI - Latent representation (30 dimensions)")

print("\nAll files saved with '_final' suffix in:")
print(f"   {output_dir}")

print("\n🎉 Done!")