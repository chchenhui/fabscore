#!/usr/bin/env python3
"""
Neuroendocrine Dataset - scVI Batch Integration with Three-way Comparison (UNIQUE DATASETS)
Includes unintegrated vs ComBat vs scVI comparisons using unique datasets only
Adapted from proven scVI pipeline for deduplicated data
"""

import os
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import gc
from scipy.stats import entropy
from sklearn.metrics import silhouette_score
from datetime import datetime
import scvi
warnings.filterwarnings('ignore')

# Configure scanpy and plotting settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=300, facecolor='white', frameon=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_style("whitegrid")
sns.set_palette("husl")

# Define paths - Updated for unique datasets
BASE_DIR = Path("/scratch/rli/project/agent/data_integration")
INPUT_FILE = Path("/scratch/rli/project/agent/results/data_integration_2025-08-21/merged_data/merged_endocrine_dataset_unique.h5ad")  # Use merged dataset directly
OUTPUT_DIR = BASE_DIR / "results"

# Create organized output directories
INTEGRATION_DIR = OUTPUT_DIR / "scvi_unique_integration"
COMPARISON_DIR = INTEGRATION_DIR / "comparison_plots"
DENSITY_DIR = INTEGRATION_DIR / "density_analysis"
CONSISTENCY_DIR = INTEGRATION_DIR / "consistency_evaluation"
FIGURES_DIR = INTEGRATION_DIR / "figures"

for dir_path in [INTEGRATION_DIR, COMPARISON_DIR, DENSITY_DIR, CONSISTENCY_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Integration parameters
N_HVG = 3000
BATCH_KEY = 'batch'  # Using batch as the key for unique datasets
N_PCS = 30  # For comparison with ComBat
N_LATENT = 30  # scVI latent dimensions
UMAP_MIN_DIST = 0.3
SCVI_MAX_EPOCHS = 50  # Reduced from 100 for faster execution

def load_and_prepare_dataset():
    """Load and prepare the dataset for integration"""
    print("Loading analyzed unique dataset...")
    
    # Try alternative locations if analysis hasn't been run yet
    if not INPUT_FILE.exists():
        alternative_file = Path("/scratch/rli/project/agent/results/data_integration_2025-08-21/merged_data/merged_endocrine_dataset_unique.h5ad")
        if alternative_file.exists():
            print("Using merged dataset (analysis will be performed inline)")
            adata = sc.read_h5ad(alternative_file)
            
            # Perform basic analysis if not done
            if 'highly_variable' not in adata.var.columns:
                print("Performing HVG selection...")
                sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, batch_key=BATCH_KEY)
                adata.raw = adata
                sc.pp.scale(adata, max_value=10)
                sc.tl.pca(adata, svd_solver='arpack')
        else:
            print(f"ERROR: No suitable input file found.")
            print(f"Expected: {INPUT_FILE}")
            print(f"Alternative: {alternative_file}")
            return None
    else:
        adata = sc.read_h5ad(INPUT_FILE)
    
    print(f"Loaded dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Verify required metadata columns
    required_cols = [BATCH_KEY]
    missing_cols = [col for col in required_cols if col not in adata.obs.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    # Create simplified metadata for visualization
    if 'endocrine_type_simple' not in adata.obs.columns:
        if 'endocrine_cell_types' in adata.obs.columns:
            adata.obs['endocrine_type_simple'] = adata.obs['endocrine_cell_types'].apply(
                lambda x: str(x).split(';')[0].strip() if pd.notna(x) and str(x) != 'Unknown' else 'Unknown'
            )
        else:
            adata.obs['endocrine_type_simple'] = 'Unknown'
    
    # Create tissue metadata if available
    if 'tissues' in adata.obs.columns and 'tissue' not in adata.obs.columns:
        adata.obs['tissue'] = adata.obs['tissues'].apply(
            lambda x: str(x).split(';')[0].strip() if pd.notna(x) else 'Unknown'
        )
    elif 'tissue' not in adata.obs.columns:
        adata.obs['tissue'] = 'Unknown'
    
    # Create disease metadata if available
    if 'diseases' in adata.obs.columns and 'disease' not in adata.obs.columns:
        adata.obs['disease'] = adata.obs['diseases'].apply(
            lambda x: str(x).split(';')[0].strip() if pd.notna(x) else 'Unknown'
        )
    elif 'disease' not in adata.obs.columns:
        adata.obs['disease'] = 'Unknown'
    
    print(f"Unique batches: {adata.obs[BATCH_KEY].nunique()}")
    print(f"Unique tissues: {adata.obs['tissue'].nunique()}")
    print(f"Unique endocrine types: {adata.obs['endocrine_type_simple'].nunique()}")
    
    return adata

def perform_combat_integration(adata):
    """Perform ComBat batch correction for comparison"""
    print(f"Performing ComBat integration for comparison using {N_HVG} HVGs...")
    
    # Store original data
    adata_combat = adata.copy()
    
    # Ensure we have HVGs
    if 'highly_variable' not in adata_combat.var.columns:
        print("Identifying highly variable genes for ComBat...")
        sc.pp.highly_variable_genes(adata_combat, n_top_genes=N_HVG, batch_key=BATCH_KEY, subset=False)
    
    print(f"Using {adata_combat.var.highly_variable.sum()} highly variable genes")
    
    # Keep only HVGs for integration
    adata_hvg = adata_combat[:, adata_combat.var.highly_variable].copy()
    
    # Normalize and log-transform
    print("Normalizing and scaling data for ComBat...")
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)
    sc.pp.scale(adata_hvg)
    
    # Perform PCA
    print(f"Computing {N_PCS} principal components...")
    sc.tl.pca(adata_hvg, n_comps=N_PCS)
    
    # ComBat batch correction
    print("Applying ComBat batch correction...")
    sc.pp.combat(adata_hvg, key=BATCH_KEY)
    
    # Compute UMAP for ComBat
    print("Computing UMAP for ComBat-corrected data...")
    sc.pp.neighbors(adata_hvg, n_pcs=N_PCS)
    sc.tl.umap(adata_hvg, min_dist=UMAP_MIN_DIST)
    
    # Store ComBat results in original adata
    adata.obsm['X_pca_combat'] = adata_hvg.obsm['X_pca']
    adata.obsm['X_umap_combat'] = adata_hvg.obsm['X_umap']
    adata.uns['combat'] = {'params': {'n_hvg': N_HVG, 'n_pcs': N_PCS}}
    
    print("✓ ComBat integration completed")
    return adata

def perform_scvi_integration(adata):
    """Perform scVI integration"""
    print(f"Performing scVI integration with {N_LATENT} latent dimensions...")
    
    # Prepare data for scVI
    adata_scvi = adata.copy()
    
    # Set up scVI
    print("Setting up scVI model...")
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        batch_key=BATCH_KEY,
        layer=None  # Use .X matrix
    )
    
    # Create and train scVI model
    print(f"Training scVI model for {SCVI_MAX_EPOCHS} epochs...")
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=N_LATENT,
        n_layers=2,
        n_hidden=128
    )
    
    # Train the model
    model.train(max_epochs=SCVI_MAX_EPOCHS, batch_size=2048, early_stopping=True)
    
    # Get latent representation
    print("Extracting scVI latent representation...")
    adata.obsm['X_scvi'] = model.get_latent_representation()
    
    # Compute UMAP on scVI latent space
    print("Computing UMAP for scVI latent space...")
    sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=15)
    # Store neighbors for scVI
    adata.uns['scvi'] = adata.uns['neighbors'].copy()
    sc.tl.umap(adata, min_dist=UMAP_MIN_DIST)
    adata.obsm['X_umap_scvi'] = adata.obsm['X_umap']
    
    print("✓ scVI integration completed")
    return adata, model

def compute_unintegrated_embedding(adata):
    """Compute unintegrated UMAP for comparison"""
    print("Computing unintegrated UMAP for comparison...")
    
    # Use original PCA if available, otherwise compute
    if 'X_pca' not in adata.obsm:
        print("Computing PCA for unintegrated data...")
        adata_temp = adata.copy()
        sc.pp.scale(adata_temp)
        sc.tl.pca(adata_temp, n_comps=50)
        adata.obsm['X_pca_unintegrated'] = adata_temp.obsm['X_pca']
    else:
        adata.obsm['X_pca_unintegrated'] = adata.obsm['X_pca']
    
    # Compute unintegrated UMAP
    sc.pp.neighbors(adata, use_rep='X_pca_unintegrated', n_neighbors=15)
    sc.tl.umap(adata, min_dist=UMAP_MIN_DIST)
    adata.obsm['X_umap_unintegrated'] = adata.obsm['X_umap']
    
    print("✓ Unintegrated embedding completed")
    return adata

def perform_clustering_analysis(adata):
    """Perform clustering on scVI latent space"""
    print("Performing Leiden clustering on scVI latent space...")
    
    # Ensure we have scVI neighbors
    if 'scvi' in adata.uns:
        adata.uns['neighbors'] = adata.uns['scvi'].copy()
    else:
        # Recompute neighbors if needed
        sc.pp.neighbors(adata, use_rep='X_scvi', n_neighbors=15)
    
    # Perform clustering
    sc.tl.leiden(adata, resolution=0.8)
    n_clusters = adata.obs['leiden'].nunique()
    
    print(f"Found {n_clusters} clusters")
    
    # Store clustering info
    adata.uns['clustering'] = {
        'method': 'leiden',
        'resolution': 0.8,
        'n_clusters': n_clusters,
        'representation': 'scvi'
    }
    
    return adata

def calculate_integration_metrics(adata):
    """Calculate integration quality metrics"""
    print("Calculating integration quality metrics...")
    
    metrics = {}
    
    # For each method, calculate batch mixing and biological conservation
    methods = {
        'unintegrated': 'X_umap_unintegrated',
        'combat': 'X_umap_combat', 
        'scvi': 'X_umap_scvi'
    }
    
    for method_name, umap_key in methods.items():
        if umap_key in adata.obsm:
            embedding = adata.obsm[umap_key]
            
            # Calculate tissue centroid distances (biological conservation)
            tissue_centroids = []
            tissues = adata.obs['tissue'].unique()
            for tissue in tissues:
                if tissue != 'Unknown':
                    mask = adata.obs['tissue'] == tissue
                    if mask.sum() > 5:  # At least 5 cells
                        centroid = embedding[mask].mean(axis=0)
                        tissue_centroids.append(centroid)
            
            if len(tissue_centroids) > 1:
                tissue_distances = []
                for i in range(len(tissue_centroids)):
                    for j in range(i+1, len(tissue_centroids)):
                        dist = np.linalg.norm(tissue_centroids[i] - tissue_centroids[j])
                        tissue_distances.append(dist)
                avg_tissue_dist = np.mean(tissue_distances)
            else:
                avg_tissue_dist = np.nan
            
            # Calculate batch centroid distances (batch mixing)
            batch_centroids = []
            batches = adata.obs[BATCH_KEY].unique()
            for batch in batches:
                mask = adata.obs[BATCH_KEY] == batch
                if mask.sum() > 5:  # At least 5 cells
                    centroid = embedding[mask].mean(axis=0)
                    batch_centroids.append(centroid)
            
            if len(batch_centroids) > 1:
                batch_distances = []
                for i in range(len(batch_centroids)):
                    for j in range(i+1, len(batch_centroids)):
                        dist = np.linalg.norm(batch_centroids[i] - batch_centroids[j])
                        batch_distances.append(dist)
                avg_batch_dist = np.mean(batch_distances)
            else:
                avg_batch_dist = np.nan
            
            metrics[method_name] = {
                'avg_tissue_distance': avg_tissue_dist,
                'avg_batch_distance': avg_batch_dist
            }
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics).T
    metrics_file = CONSISTENCY_DIR / "integration_metrics_unique.csv"
    metrics_df.to_csv(metrics_file)
    
    print("Integration quality metrics:")
    for method, values in metrics.items():
        print(f"  {method}:")
        print(f"    Tissue separation: {values['avg_tissue_distance']:.2f}")
        print(f"    Batch mixing: {values['avg_batch_distance']:.2f}")
    
    return metrics

def create_comprehensive_comparison_plots(adata):
    """Create comprehensive three-way comparison plots"""
    print("Creating comprehensive comparison plots...")
    
    # Categories to compare
    categories = [
        (BATCH_KEY, 'Batch Effects'),
        ('endocrine_type_simple', 'Cell Types'), 
        ('tissue', 'Tissues'),
        ('disease', 'Disease States')
    ]
    
    # Create 4x3 comparison figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    
    for row, (cat_key, cat_name) in enumerate(categories):
        if cat_key not in adata.obs.columns:
            continue
            
        # Check category count for legend decision
        n_categories = adata.obs[cat_key].nunique()
        show_legend = n_categories <= 20
        legend_loc = 'right margin' if show_legend else None
        
        # Column 1: Unintegrated
        if 'X_umap_unintegrated' in adata.obsm:
            adata.obsm['X_umap'] = adata.obsm['X_umap_unintegrated']
            sc.pl.umap(adata, color=cat_key, ax=axes[row,0], show=False, 
                      title=f'Unintegrated: {cat_name}',
                      legend_loc=legend_loc, legend_fontsize=6, frameon=False,
                      s=1)
        
        # Column 2: ComBat
        if 'X_umap_combat' in adata.obsm:
            adata.obsm['X_umap'] = adata.obsm['X_umap_combat']
            sc.pl.umap(adata, color=cat_key, ax=axes[row,1], show=False,
                      title=f'ComBat: {cat_name}',
                      legend_loc=legend_loc, legend_fontsize=6, frameon=False,
                      s=1)
        
        # Column 3: scVI
        if 'X_umap_scvi' in adata.obsm:
            adata.obsm['X_umap'] = adata.obsm['X_umap_scvi']
            sc.pl.umap(adata, color=cat_key, ax=axes[row,2], show=False,
                      title=f'scVI: {cat_name}',
                      legend_loc=legend_loc, legend_fontsize=6, frameon=False,
                      s=1)
    
    plt.suptitle('Three-way Integration Comparison (Unique Datasets)', fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(COMPARISON_DIR / 'three_way_integration_comparison_unique.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: three_way_integration_comparison_unique.png")

def create_integration_overview(adata):
    """Create scVI integration overview with 2x4 grid"""
    print("Creating scVI integration overview...")
    
    if 'X_umap_scvi' not in adata.obsm:
        print("No scVI UMAP found, skipping overview")
        return
    
    # Set scVI as main view
    adata.obsm['X_umap'] = adata.obsm['X_umap_scvi']
    
    # Create 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Main biological annotations
    row1_categories = [
        (BATCH_KEY, 'Batch Effects'),
        ('tissue', 'Tissues'),
        ('endocrine_type_simple', 'Cell Types'),
        ('disease', 'Disease States')
    ]
    
    for col, (cat_key, cat_name) in enumerate(row1_categories):
        if cat_key in adata.obs.columns:
            n_categories = adata.obs[cat_key].nunique()
            show_legend = n_categories <= 15
            legend_loc = 'right margin' if show_legend else None
            
            sc.pl.umap(adata, color=cat_key, ax=axes[0,col], show=False,
                      title=f'scVI: {cat_name}',
                      legend_loc=legend_loc, legend_fontsize=6, frameon=False,
                      s=1)
    
    # Row 2: Technical annotations
    row2_specs = [
        ('assay', 'Assay', None),
        ('leiden', 'Leiden Clusters', None), 
        ('n_genes_by_counts', 'N Genes', 'viridis'),
        ('total_counts', 'Total Counts', 'YlOrRd')
    ]
    
    for col, (cat_key, cat_name, cmap) in enumerate(row2_specs):
        if cat_key in adata.obs.columns:
            if cmap:  # Continuous variables
                sc.pl.umap(adata, color=cat_key, ax=axes[1,col], show=False,
                          title=f'scVI: {cat_name}', color_map=cmap,
                          s=1)
            else:  # Categorical variables
                n_categories = adata.obs[cat_key].nunique()
                show_legend = n_categories <= 15
                legend_loc = 'right margin' if show_legend else None
                
                sc.pl.umap(adata, color=cat_key, ax=axes[1,col], show=False,
                          title=f'scVI: {cat_name}',
                          legend_loc=legend_loc, legend_fontsize=6, frameon=False,
                          s=1)
    
    plt.suptitle('scVI Integration Overview (Unique Datasets)', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURES_DIR / 'scvi_integration_overview_unique.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: scvi_integration_overview_unique.png")

def save_integration_results(adata, model):
    """Save integration results"""
    print("Saving integration results...")
    
    # Save integrated dataset
    output_file = INTEGRATION_DIR / "neuroendocrine_scvi_integrated_unique.h5ad"
    
    # Note: scVI models cannot be saved in h5ad format
    # Remove model from adata if present
    if hasattr(adata, 'uns') and 'scvi_model' in adata.uns:
        del adata.uns['scvi_model']
    
    adata.write_h5ad(output_file, compression='gzip')
    print(f"Saved integrated dataset: {output_file}")
    
    # Save scVI model separately
    model_dir = INTEGRATION_DIR / "scvi_model"
    model.save(model_dir, overwrite=True)
    print(f"Saved scVI model: {model_dir}")
    
    # Generate integration summary
    integration_summary = {
        'total_cells': adata.n_obs,
        'total_genes': adata.n_vars,
        'unique_batches': adata.obs[BATCH_KEY].nunique(),
        'unique_datasets': True,
        'duplicates_removed': 9,
        'integration_methods': ['unintegrated', 'combat', 'scvi'],
        'scvi_latent_dims': N_LATENT,
        'scvi_max_epochs': SCVI_MAX_EPOCHS,
        'clusters_found': adata.obs['leiden'].nunique() if 'leiden' in adata.obs.columns else 0,
        'integration_date': datetime.now().isoformat()
    }
    
    summary_df = pd.DataFrame.from_dict(integration_summary, orient='index', columns=['Value'])
    summary_file = INTEGRATION_DIR / "integration_summary_unique.csv"
    summary_df.to_csv(summary_file)
    print(f"Saved integration summary: {summary_file}")
    
    return output_file

def main():
    print("="*80)
    print("Neuroendocrine Dataset - scVI Integration with Three-way Comparison")
    print("UNIQUE DATASETS ONLY (No Duplicates)")
    print("="*80)
    print(f"Integration parameters:")
    print(f"  - HVGs for ComBat: {N_HVG}")
    print(f"  - scVI latent dimensions: {N_LATENT}")
    print(f"  - scVI max epochs: {SCVI_MAX_EPOCHS}")
    print(f"  - Batch key: {BATCH_KEY}")
    print(f"  - UMAP min_dist: {UMAP_MIN_DIST}")
    print("="*80)
    
    # Load and prepare dataset
    adata = load_and_prepare_dataset()
    if adata is None:
        return
    
    # Compute unintegrated embedding
    adata = compute_unintegrated_embedding(adata)
    
    # Perform ComBat integration
    adata = perform_combat_integration(adata)
    
    # Perform scVI integration
    adata, model = perform_scvi_integration(adata)
    
    # Perform clustering analysis
    adata = perform_clustering_analysis(adata)
    
    # Calculate integration metrics
    metrics = calculate_integration_metrics(adata)
    
    # SAVE RESULTS IMMEDIATELY AFTER INTEGRATION
    print("\n" + "="*80)
    print("SAVING INTEGRATED DATA BEFORE VISUALIZATION")
    print("="*80)
    output_file = save_integration_results(adata, model)
    print(f"✓ Data successfully saved to: {output_file}")
    print("✓ Proceeding with visualization (errors here won't affect saved data)")
    print("="*80 + "\n")
    
    # Now attempt visualization (if these fail, we still have saved data)
    try:
        create_comprehensive_comparison_plots(adata)
    except Exception as e:
        print(f"Warning: Error in comprehensive comparison plots: {e}")
        print("Continuing...")
    
    try:
        create_integration_overview(adata)
    except Exception as e:
        print(f"Warning: Error in integration overview: {e}")
        print("Continuing...")
    
    print("\n" + "="*80)
    print("scVI INTEGRATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"✓ Processed dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"✓ Unique batches integrated: {adata.obs[BATCH_KEY].nunique()}")
    print(f"✓ Clusters found: {adata.obs['leiden'].nunique()}")
    print(f"✓ Integration methods: Unintegrated, ComBat, scVI")
    print(f"✓ Results saved to: {output_file}")
    print("\n✓ Three-way integration comparison completed successfully!")
    print("✓ Ready for downstream analysis and interpretation!")

if __name__ == "__main__":
    main()