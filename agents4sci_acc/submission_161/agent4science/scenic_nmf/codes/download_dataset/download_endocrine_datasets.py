#!/usr/bin/env python3
"""
Download single-cell datasets containing endocrine cells from CZ CELLxGENE Census
"""

import cellxgene_census
import pandas as pd
import os
from typing import Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def find_endocrine_datasets(census_version: str = "2025-01-30"):
    """
    Find all datasets containing endocrine cells in human data
    """
    print(f"Connecting to CZ CELLxGENE Census version {census_version}...")
    
    with cellxgene_census.open_soma(census_version=census_version) as census:
        # Get human data
        human_data = census["census_data"]["homo_sapiens"]
        
        # Query for cells with "endocrine" in cell_type
        print("Querying for endocrine cells...")
        obs_df = human_data.obs.read(
            column_names=["soma_joinid", "dataset_id", "cell_type", "tissue", 
                         "assay", "disease", "sex", "development_stage", 
                         "cell_type_ontology_term_id", "tissue_ontology_term_id"]
        ).concat().to_pandas()
        
        # Filter for endocrine cells
        endocrine_mask = obs_df['cell_type'].str.contains('endocrine', case=False, na=False)
        endocrine_cells = obs_df[endocrine_mask].copy()
        
        print(f"Found {len(endocrine_cells):,} endocrine cells across datasets")
        
        # Get dataset metadata
        datasets_df = census["census_info"]["datasets"].read().concat().to_pandas()
        
        return endocrine_cells, datasets_df

def aggregate_by_dataset(endocrine_cells: pd.DataFrame, datasets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate endocrine cell data by dataset
    """
    print("Aggregating data by dataset...")
    
    # Group by dataset to get statistics
    dataset_stats = endocrine_cells.groupby('dataset_id').agg({
        'soma_joinid': 'count',  # Number of endocrine cells
        'cell_type': lambda x: list(x.unique()),  # Unique endocrine cell types
        'tissue': lambda x: list(x.unique()),  # Unique tissues
        'assay': lambda x: list(x.unique()),  # Unique assays
        'disease': lambda x: list(x.unique()),  # Unique diseases
    }).reset_index()
    
    dataset_stats.columns = ['dataset_id', 'endocrine_cell_count', 
                            'endocrine_cell_types', 'tissues', 'assays', 'diseases']
    
    # Convert lists to strings for better CSV storage
    for col in ['endocrine_cell_types', 'tissues', 'assays', 'diseases']:
        dataset_stats[col] = dataset_stats[col].apply(lambda x: '; '.join(x) if x else '')
    
    # Merge with dataset metadata
    summary_df = pd.merge(
        dataset_stats,
        datasets_df[['dataset_id', 'collection_id', 'collection_name', 
                     'collection_doi', 'dataset_title', 'dataset_h5ad_path', 
                     'dataset_total_cell_count']],
        on='dataset_id',
        how='left'
    )
    
    # Calculate percentage of endocrine cells
    summary_df['endocrine_percentage'] = (
        summary_df['endocrine_cell_count'] / summary_df['dataset_total_cell_count'] * 100
    ).round(2)
    
    # Sort by endocrine cell count
    summary_df = summary_df.sort_values('endocrine_cell_count', ascending=False)
    
    # Ensure endocrine_cell_types is not blank
    summary_df = summary_df[summary_df['endocrine_cell_types'].str.len() > 0]
    
    return summary_df

def download_datasets(summary_df: pd.DataFrame, output_dir: str = "endocrine_datasets", census_version: str = "2025-01-30"):
    """
    Download h5ad files for all datasets using cellxgene_census
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nDownloading {len(summary_df)} datasets to {output_dir}/...")
    
    success_count = 0
    failed_downloads = []
    
    for idx, row in summary_df.iterrows():
        dataset_id = row['dataset_id']
        
        # Create filename from dataset_id
        filename = f"{dataset_id}.h5ad"
        filepath = os.path.join(output_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"\nSkipping {dataset_id}: Already downloaded")
            success_count += 1
            continue
        
        print(f"\nDownloading dataset {idx+1}/{len(summary_df)}: {dataset_id}")
        print(f"  Title: {row['dataset_title'][:80]}...")
        print(f"  Endocrine cells: {row['endocrine_cell_count']:,} ({row['endocrine_percentage']:.1f}%)")
        
        try:
            # Use cellxgene_census download_source_h5ad method
            cellxgene_census.download_source_h5ad(
                dataset_id=dataset_id,
                to_path=filepath,
                census_version=census_version
            )
            success_count += 1
            print(f"  Successfully downloaded to {filename}")
        except Exception as e:
            print(f"  Failed to download: {e}")
            failed_downloads.append(dataset_id)
    
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"  Successfully downloaded: {success_count}/{len(summary_df)} datasets")
    if failed_downloads:
        print(f"  Failed downloads: {', '.join(failed_downloads)}")
    
    return success_count, failed_downloads

def main():
    """
    Main function to orchestrate the download process
    """
    # Find endocrine datasets
    endocrine_cells, datasets_df = find_endocrine_datasets(census_version="2025-01-30")
    
    # Aggregate by dataset
    summary_df = aggregate_by_dataset(endocrine_cells, datasets_df)
    
    print(f"\nFound {len(summary_df)} datasets containing endocrine cells")
    
    # Save summary to CSV
    csv_filename = "endocrine_datasets_summary.csv"
    summary_df.to_csv(csv_filename, index=False)
    print(f"Summary saved to {csv_filename}")
    
    # Display top datasets
    print("\nTop 10 datasets by endocrine cell count:")
    display_cols = ['dataset_id', 'dataset_title', 'endocrine_cell_count', 
                   'endocrine_percentage', 'endocrine_cell_types']
    
    for idx, row in summary_df.head(10).iterrows():
        print(f"\n{idx+1}. {row['dataset_id']}")
        print(f"   Title: {row['dataset_title'][:70]}...")
        print(f"   Endocrine cells: {row['endocrine_cell_count']:,} ({row['endocrine_percentage']:.1f}%)")
        print(f"   Cell types: {row['endocrine_cell_types'][:100]}...")
    
    # Download all datasets
    print("\n" + "="*60)
    print("Starting dataset downloads...")
    success_count, failed_downloads = download_datasets(summary_df, census_version="2025-01-30")
    
    print("\nProcess complete!")
    print(f"Summary CSV: {csv_filename}")
    print(f"Downloaded files: endocrine_datasets/")
    
    return summary_df

if __name__ == "__main__":
    summary_df = main()