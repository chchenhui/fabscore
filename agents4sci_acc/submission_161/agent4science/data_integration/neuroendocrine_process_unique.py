#!/usr/bin/env python3
"""
Complete Neuroendocrine Dataset Integration Pipeline - UNIQUE DATASETS ONLY
Adapted from proven pipeline to use deduplicated metadata (55 unique datasets)
"""

import os
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import gc
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure scanpy and plotting settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_style("whitegrid")
sns.set_palette("husl")

# Define paths
DATA_DIR = Path("/scratch/rli/data/neuroendocrine_dataset/endocrine_datasets")
METADATA_PATH = Path("/scratch/rli/project/agent/data_integration/unique_datasets_metadata_final.csv")  # Use unique metadata
BASE_DIR = Path("/scratch/rli/project/agent/data_integration")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / f"processed_{datetime.now().strftime('%Y-%m-%d')}"

# Create organized output directories
QC_DIR = RESULTS_DIR  # QC plots go directly to results
PROCESSED_DIR = OUTPUT_DIR / "processed_data"
MERGED_DIR = OUTPUT_DIR / "merged_data"
FIGURES_DIR = RESULTS_DIR  # Figures go directly to results
STATS_DIR = RESULTS_DIR  # Statistics go directly to results

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for dir_path in [PROCESSED_DIR, MERGED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Processing parameters
MIN_ENDOCRINE_PERCENTAGE = 1.0  # Skip datasets with less than 1.0% endocrine cells
MAX_CELLS_BEFORE_SUBSAMPLE = 100000  # Subsample datasets larger than this
SUBSAMPLE_SIZE = 50000  # Target size after subsampling
SUBSAMPLE_SEED = 42  # Random seed for reproducibility

def calculate_qc_metrics(adata, batch_name):
    """Calculate quality control metrics for a dataset"""
    
    # Calculate mitochondrial gene percentage
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Calculate ribosomal gene percentage  
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['ribo'], percent_top=None, log1p=False, inplace=True)
    
    # Calculate hemoglobin gene percentage
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['hb'], percent_top=None, log1p=False, inplace=True)
    
    # Store batch information
    adata.obs['batch'] = batch_name
    
    return adata

def filter_cells_and_genes(adata, min_genes=200, min_cells=3, max_genes=2500, max_mt_percent=20):
    """Apply quality control filters"""
    
    print(f"  Before filtering: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :]
    adata = adata[adata.obs['pct_counts_mt'] < max_mt_percent, :]
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    print(f"  After filtering: {adata.n_obs} cells, {adata.n_vars} genes")
    
    return adata

def normalize_and_log_transform(adata):
    """Normalize and log-transform the data"""
    
    # Store raw counts
    adata.raw = adata
    
    # Normalize to 10,000 reads per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    sc.pp.log1p(adata)
    
    return adata

def process_dataset(row):
    """Process a single dataset with QC and normalization"""
    
    dataset_id = row['dataset_id']
    endocrine_percentage = float(row['endocrine_percentage'])
    total_cells = int(row['dataset_total_cell_count'])
    
    # Skip datasets with very low endocrine percentage
    if endocrine_percentage < MIN_ENDOCRINE_PERCENTAGE:
        print(f"  Skipping: Low endocrine percentage ({endocrine_percentage}%)")
        return None
    
    h5ad_file = DATA_DIR / f"{dataset_id}.h5ad"
    
    if not h5ad_file.exists():
        print(f"  Warning: File not found: {h5ad_file}")
        return None
    
    try:
        # Check if subsampling is needed
        if total_cells > MAX_CELLS_BEFORE_SUBSAMPLE:
            print(f"  Large dataset ({total_cells} cells) - will subsample to {SUBSAMPLE_SIZE}")
            # Read with backed mode first
            adata_backed = sc.read_h5ad(h5ad_file, backed='r')
            n_cells = adata_backed.shape[0]
            
            # Randomly subsample cells
            np.random.seed(SUBSAMPLE_SEED)
            subsample_idx = np.random.choice(n_cells, size=min(SUBSAMPLE_SIZE, n_cells), replace=False)
            subsample_idx = np.sort(subsample_idx)
            
            # Load subsampled data
            adata = adata_backed[subsample_idx].to_memory()
            print(f"  Subsampled to: {adata.shape[0]} cells")
        else:
            # Load the full dataset
            adata = sc.read_h5ad(h5ad_file)
            print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
        
        # Calculate QC metrics
        adata = calculate_qc_metrics(adata, dataset_id)
        
        # Add metadata from CSV
        for col in ['dataset_id', 'collection_name', 'dataset_title', 'tissues', 
                   'assays', 'diseases', 'endocrine_cell_types', 'endocrine_cell_count',
                   'dataset_total_cell_count', 'endocrine_percentage']:
            if col in row:
                adata.obs[col] = str(row[col]) if pd.notna(row[col]) else 'Unknown'
        
        # Store dataset metadata in uns
        adata.uns['dataset_metadata'] = {
            'dataset_id': dataset_id,
            'collection_id': str(row['collection_id']),
            'collection_name': str(row['collection_name']),
            'collection_doi': str(row['collection_doi']) if 'collection_doi' in row else 'Unknown',
            'dataset_title': str(row['dataset_title']),
            'was_subsampled': total_cells > MAX_CELLS_BEFORE_SUBSAMPLE
        }
        
        # Apply QC filters
        adata = filter_cells_and_genes(adata)
        
        # Skip if no cells remain after filtering
        if adata.n_obs == 0:
            print(f"  Warning: No cells remaining after QC filtering")
            return None
        
        # Normalize and log-transform
        adata = normalize_and_log_transform(adata)
        
        return adata
        
    except Exception as e:
        print(f"  Error processing {dataset_id}: {str(e)}")
        return None

def clean_dataset_name(dataset_id, metadata_df):
    """Create a clean, elegant dataset name for plotting"""
    row = metadata_df[metadata_df['dataset_id'] == dataset_id]
    if not row.empty:
        title = row.iloc[0]['dataset_title']
        if len(title) > 25:
            title = title[:22] + "..."
        tissue = row.iloc[0]['tissues'].split(';')[0].strip() if pd.notna(row.iloc[0]['tissues']) else "Unknown"
        tissue = tissue[:15] if len(tissue) > 15 else tissue
        return f"{tissue} - {title}"
    return dataset_id[:12]

def plot_qc_violin_genes(adata, metadata_df, save_dir):
    """Professional violin plot for number of genes per cell"""
    
    # Clean batch names for top 12 batches
    top_batches = adata.obs['batch'].value_counts().head(12).index
    adata_subset = adata[adata.obs['batch'].isin(top_batches)].copy()
    
    batch_mapping = {}
    for batch in adata_subset.obs['batch'].unique():
        batch_mapping[batch] = clean_dataset_name(batch, metadata_df)
    adata_subset.obs['batch_clean'] = adata_subset.obs['batch'].map(batch_mapping)
    
    # Create professional violin plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sc.pl.violin(adata_subset, 'n_genes_by_counts', groupby='batch_clean',
                 rotation=45, ax=ax, show=False, stripplot=False, 
                 palette='Set2', size=1.2)
    
    ax.set_title('Number of Genes Expressed per Cell (Unique Datasets)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Genes', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    median_genes = np.median(adata.obs['n_genes_by_counts'])
    ax.axhline(y=median_genes, color='red', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Overall Median: {median_genes:.0f}')
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'qc_violin_genes_per_cell_unique.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: qc_violin_genes_per_cell_unique.png")

def plot_qc_violin_counts(adata, metadata_df, save_dir):
    """Professional violin plot for total counts per cell"""
    
    # Clean batch names for top 12 batches
    top_batches = adata.obs['batch'].value_counts().head(12).index
    adata_subset = adata[adata.obs['batch'].isin(top_batches)].copy()
    
    batch_mapping = {}
    for batch in adata_subset.obs['batch'].unique():
        batch_mapping[batch] = clean_dataset_name(batch, metadata_df)
    adata_subset.obs['batch_clean'] = adata_subset.obs['batch'].map(batch_mapping)
    
    # Create professional violin plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sc.pl.violin(adata_subset, 'total_counts', groupby='batch_clean',
                 rotation=45, ax=ax, show=False, stripplot=False,
                 palette='viridis', size=1.2)
    
    ax.set_title('Total UMI Counts per Cell (Unique Datasets)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total UMI Counts', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    median_counts = np.median(adata.obs['total_counts'])
    ax.axhline(y=median_counts, color='red', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Overall Median: {median_counts:.0f}')
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'qc_violin_total_counts_unique.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: qc_violin_total_counts_unique.png")

def plot_qc_violin_mitochondrial(adata, metadata_df, save_dir):
    """Professional violin plot for mitochondrial gene percentage"""
    
    # Clean batch names for top 12 batches
    top_batches = adata.obs['batch'].value_counts().head(12).index
    adata_subset = adata[adata.obs['batch'].isin(top_batches)].copy()
    
    batch_mapping = {}
    for batch in adata_subset.obs['batch'].unique():
        batch_mapping[batch] = clean_dataset_name(batch, metadata_df)
    adata_subset.obs['batch_clean'] = adata_subset.obs['batch'].map(batch_mapping)
    
    # Create professional violin plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sc.pl.violin(adata_subset, 'pct_counts_mt', groupby='batch_clean',
                 rotation=45, ax=ax, show=False, stripplot=False,
                 palette='plasma', size=1.2)
    
    ax.set_title('Mitochondrial Gene Percentage per Cell (Unique Datasets)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mitochondrial Gene Percentage (%)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=11)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    median_mt = np.median(adata.obs['pct_counts_mt'])
    ax.axhline(y=median_mt, color='red', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Overall Median: {median_mt:.2f}%')
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'qc_violin_mitochondrial_pct_unique.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: qc_violin_mitochondrial_pct_unique.png")

def plot_qc_scatter_colored_by_mt(adata, save_dir):
    """Professional scatter plot of QC metrics colored by mitochondrial percentage"""
    
    # Sample data for better visualization if too many cells
    n_cells = adata.n_obs
    if n_cells > 10000:
        sample_idx = np.random.choice(n_cells, size=10000, replace=False)
        adata_plot = adata[sample_idx]
    else:
        adata_plot = adata
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create scatter plot
    scatter = ax.scatter(adata_plot.obs['total_counts'], 
                        adata_plot.obs['n_genes_by_counts'],
                        c=adata_plot.obs['pct_counts_mt'], 
                        cmap='RdYlBu_r', 
                        alpha=0.6, 
                        s=15,
                        edgecolors='none')
    
    ax.set_xlabel('Total UMI Counts', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Genes', fontsize=14, fontweight='bold')
    ax.set_title('QC Metrics Scatter Plot (Unique Datasets)', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mitochondrial Gene Percentage (%)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Add correlation coefficient
    corr = np.corrcoef(adata_plot.obs['total_counts'], adata_plot.obs['n_genes_by_counts'])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, 
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add statistics box
    stats_text = f"""Dataset Statistics:
Total Cells: {n_cells:,}
Plotted: {len(adata_plot):,}
Med. Genes: {np.median(adata.obs['n_genes_by_counts']):.0f}
Med. UMI: {np.median(adata.obs['total_counts']):.0f}
Med. MT%: {np.median(adata.obs['pct_counts_mt']):.2f}"""
    
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / 'qc_scatter_genes_vs_counts_colored_by_mt_unique.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: qc_scatter_genes_vs_counts_colored_by_mt_unique.png")

def main():
    print("="*60)
    print("Neuroendocrine Dataset Integration Pipeline - UNIQUE DATASETS")
    print("="*60)
    print(f"Using deduplicated metadata: 55 unique datasets")
    print(f"Settings:")
    print(f"  - Min endocrine percentage: {MIN_ENDOCRINE_PERCENTAGE}%")
    print(f"  - Max cells before subsample: {MAX_CELLS_BEFORE_SUBSAMPLE}")
    print(f"  - Subsample size: {SUBSAMPLE_SIZE}")
    print(f"  - Output directory: {RESULTS_DIR}")
    print("="*60)
    
    # Load unique metadata
    print("\nLoading unique metadata...")
    if not METADATA_PATH.exists():
        print(f"ERROR: Unique metadata file not found: {METADATA_PATH}")
        print("Please run the deduplication analysis first.")
        return
        
    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Sort by endocrine percentage (prioritize higher percentage) and cell count
    metadata_df = metadata_df.sort_values(['endocrine_percentage', 'endocrine_cell_count'], 
                                         ascending=[False, True])
    
    print(f"Found {len(metadata_df)} unique datasets in metadata")
    eligible_datasets = (metadata_df['endocrine_percentage'] >= MIN_ENDOCRINE_PERCENTAGE).sum()
    print(f"Datasets with >= {MIN_ENDOCRINE_PERCENTAGE}% endocrine cells: {eligible_datasets}")
    
    # Process datasets
    processed_datasets = []
    failed_datasets = []
    skipped_datasets = []
    
    print("\n" + "="*60)
    print("Processing unique datasets...")
    print("="*60)
    
    for idx, row in metadata_df.iterrows():
        dataset_id = row['dataset_id']
        endocrine_pct = float(row['endocrine_percentage'])
        
        print(f"\n[{len(processed_datasets)+len(skipped_datasets)+1}/{len(metadata_df)}] Processing {dataset_id} (endocrine: {endocrine_pct}%)...")
        
        if endocrine_pct < MIN_ENDOCRINE_PERCENTAGE:
            print(f"  Skipping: Low endocrine percentage")
            skipped_datasets.append(dataset_id)
            continue
        
        adata = process_dataset(row)
        
        if adata is not None:
            processed_datasets.append(adata)
            
            # Clear memory periodically
            if len(processed_datasets) % 10 == 0:
                gc.collect()
                print(f"  Memory cleared after {len(processed_datasets)} datasets")
        else:
            failed_datasets.append(dataset_id)
    
    print("\n" + "="*60)
    print(f"Successfully processed: {len(processed_datasets)} datasets")
    print(f"Skipped (low endocrine %): {len(skipped_datasets)} datasets")
    print(f"Failed to process: {len(failed_datasets)} datasets")
    print("="*60)
    
    if len(processed_datasets) == 0:
        print("No datasets to merge!")
        return
    
    # Merge datasets
    print("\nMerging processed datasets...")
    adata_merged = ad.concat(
        processed_datasets,
        join='outer',
        merge='same',
        label='batch',
        keys=[adata.obs['batch'].iloc[0] for adata in processed_datasets],
        index_unique='-',
        fill_value=0
    )
    
    # Clear memory
    del processed_datasets
    gc.collect()
    
    print(f"Merged dataset shape: {adata_merged.n_obs} cells × {adata_merged.n_vars} genes")
    
    # Fix data types for saving
    print("Fixing data types...")
    for col in adata_merged.obs.columns:
        if adata_merged.obs[col].dtype == 'object':
            adata_merged.obs[col] = adata_merged.obs[col].astype(str)
    
    # Add merge information
    adata_merged.uns['merge_info'] = {
        'n_batches': adata_merged.obs['batch'].nunique(),
        'total_cells': adata_merged.n_obs,
        'total_genes': adata_merged.n_vars,
        'failed_datasets': failed_datasets,
        'skipped_datasets': skipped_datasets,
        'unique_datasets_only': True,
        'duplicates_removed': 9,  # From our analysis
        'preprocessing_params': {
            'min_genes': 200,
            'min_cells': 3,
            'max_genes': 2500,
            'max_mt_percent': 20,
            'normalization_target_sum': 10000,
            'min_endocrine_percentage': MIN_ENDOCRINE_PERCENTAGE,
            'max_cells_before_subsample': MAX_CELLS_BEFORE_SUBSAMPLE,
            'subsample_size': SUBSAMPLE_SIZE
        }
    }
    
    # Calculate QC metrics for merged dataset
    print("Calculating QC metrics for merged dataset...")
    adata_merged.var['mt'] = adata_merged.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata_merged, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    adata_merged.var['ribo'] = adata_merged.var_names.str.startswith(('RPS', 'RPL'))
    sc.pp.calculate_qc_metrics(adata_merged, qc_vars=['ribo'], percent_top=None, log1p=False, inplace=True)
    
    adata_merged.var['hb'] = adata_merged.var_names.str.contains('^HB[^(P)]')
    sc.pp.calculate_qc_metrics(adata_merged, qc_vars=['hb'], percent_top=None, log1p=False, inplace=True)
    
    # Generate professional QC plots
    print("\nGenerating professional QC plots...")
    plot_qc_violin_genes(adata_merged, metadata_df, QC_DIR)
    plot_qc_violin_counts(adata_merged, metadata_df, QC_DIR)
    plot_qc_violin_mitochondrial(adata_merged, metadata_df, QC_DIR)
    plot_qc_scatter_colored_by_mt(adata_merged, QC_DIR)
    
    # Save merged dataset
    output_file = MERGED_DIR / "merged_endocrine_dataset_unique.h5ad"
    print(f"\nSaving merged dataset to: {output_file}")
    adata_merged.write_h5ad(output_file, compression='gzip')
    
    # Generate summary statistics
    print("Generating summary statistics...")
    
    # Summary statistics
    summary_stats = {
        'Total cells after QC': adata_merged.n_obs,
        'Total genes after QC': adata_merged.n_vars,
        'Number of batches': adata_merged.obs['batch'].nunique(),
        'Datasets processed': len(set(adata_merged.obs['batch'].unique())),
        'Datasets skipped (low endocrine)': len(skipped_datasets),
        'Datasets failed': len(failed_datasets),
        'Original datasets': 64,
        'Duplicates removed': 9,
        'Unique datasets available': 55,
        'Median genes per cell': np.median(adata_merged.obs['n_genes_by_counts']),
        'Median counts per cell': np.median(adata_merged.obs['total_counts']),
        'Median % mitochondrial': np.median(adata_merged.obs['pct_counts_mt'])
    }
    
    # Save summary report
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index', columns=['Value'])
    summary_file = STATS_DIR / "integration_summary_unique.csv"
    summary_df.to_csv(summary_file)
    
    # Save batch information
    batch_info = adata_merged.obs.groupby('batch').agg({
        'dataset_title': 'first',
        'endocrine_cell_count': 'first',
        'endocrine_percentage': 'first',
        'tissues': 'first',
        'assays': 'first',
        'diseases': 'first',
        'endocrine_cell_types': 'first',
        'n_genes_by_counts': 'median',
        'total_counts': 'median',
        'pct_counts_mt': 'median'
    }).reset_index()
    
    batch_info.columns = ['batch', 'dataset_title', 'original_endocrine_count', 'endocrine_percentage',
                          'tissues', 'assays', 'diseases', 'cell_types', 'median_genes', 
                          'median_counts', 'median_pct_mt']
    
    # Add current cell count after QC
    batch_counts = adata_merged.obs['batch'].value_counts()
    batch_info['cells_after_qc'] = batch_info['batch'].map(batch_counts)
    
    batch_info_file = STATS_DIR / "batch_information_unique.csv"
    batch_info.to_csv(batch_info_file, index=False)
    
    print("\n" + "="*60)
    print("UNIQUE DATASETS INTEGRATION SUMMARY")
    print("="*60)
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"✓ Merged dataset: {output_file}")
    print(f"✓ Summary statistics: {summary_file}")
    print(f"✓ Batch information: {batch_info_file}")
    
    # List QC plots
    qc_plots = list(QC_DIR.glob('*.png'))
    print(f"✓ Professional QC plots ({len(qc_plots)}):")
    for plot in sorted(qc_plots):
        print(f"  - {plot.name}")
    
    print("\n✓ Complete neuroendocrine dataset integration (unique datasets) finished successfully!")
    print(f"✓ Ready for next step: Integration analysis and scVI pipeline")

if __name__ == "__main__":
    main()