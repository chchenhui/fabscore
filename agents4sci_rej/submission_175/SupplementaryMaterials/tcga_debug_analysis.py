#!/usr/bin/env python3
"""
TCGA Dataset Debug and Analysis
===============================
Debug the .rds file structure and then perform comprehensive analysis
"""

import pyreadr
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def debug_rds_files():
    """Debug the structure of RDS files"""
    data_dir = Path("agent4science/data")
    
    print("DEBUGGING RDS FILE STRUCTURE")
    print("=" * 50)
    
    # Test clinical data
    clinical_file = data_dir / "clinical_data" / "ALL_Cancer_clinical.rds"
    print(f"Clinical file exists: {clinical_file.exists()}")
    
    if clinical_file.exists():
        try:
            clinical_result = pyreadr.read_r(str(clinical_file))
            print(f"Clinical keys: {list(clinical_result.keys())}")
            print(f"Clinical result type: {type(clinical_result)}")
            
            # Try different access methods
            if len(clinical_result) > 0:
                for key in clinical_result.keys():
                    print(f"Key '{key}': shape {clinical_result[key].shape}")
        except Exception as e:
            print(f"Clinical data error: {e}")
    
    # Test RNA-seq data
    rnaseq_dir = data_dir / "RNAseq_data"
    test_files = ["BRCA_data.rds", "ACC_data.rds"]
    
    for test_file in test_files:
        file_path = rnaseq_dir / test_file
        print(f"\nTesting {test_file}:")
        
        if file_path.exists():
            try:
                result = pyreadr.read_r(str(file_path))
                print(f"  Keys: {list(result.keys())}")
                
                if len(result) > 0:
                    for key in result.keys():
                        data = result[key]
                        print(f"  Key '{key}': shape {data.shape}, type {type(data)}")
                        if hasattr(data, 'columns'):
                            print(f"    Sample columns: {list(data.columns[:5])}")
                        if hasattr(data, 'index'):
                            print(f"    Sample rows: {list(data.index[:3])}")
            except Exception as e:
                print(f"  Error: {e}")

def simple_analysis():
    """Simplified analysis that handles RDS files properly"""
    data_dir = Path("agent4science/data")
    
    print("\n" + "=" * 60)
    print("SIMPLIFIED TCGA ANALYSIS")
    print("=" * 60)
    
    # Analyze clinical data
    print("\n1. CLINICAL DATA ANALYSIS")
    print("-" * 30)
    
    clinical_file = data_dir / "clinical_data" / "ALL_Cancer_clinical.rds"
    clinical_data = None
    
    try:
        clinical_result = pyreadr.read_r(str(clinical_file))
        
        # Handle different RDS structures
        if len(clinical_result) == 0:
            print("Clinical data is empty or not accessible")
        else:
            # Get the data (it might be stored under different keys)
            for key in clinical_result.keys():
                data = clinical_result[key]
                if hasattr(data, 'shape') and len(data.shape) == 2:
                    clinical_data = data
                    break
            
            if clinical_data is not None:
                print(f"Clinical data loaded successfully")
                print(f"Shape: {clinical_data.shape}")
                print(f"Columns: {len(clinical_data.columns)}")
                
                # Show some column names
                print(f"Sample columns: {list(clinical_data.columns[:10])}")
                
                # Look for key survival variables
                survival_keywords = ['OS', 'survival', 'time', 'event', 'vital', 'status']
                survival_cols = []
                for col in clinical_data.columns:
                    if any(keyword.lower() in col.lower() for keyword in survival_keywords):
                        survival_cols.append(col)
                
                print(f"Potential survival columns ({len(survival_cols)}): {survival_cols[:10]}")
                
                # Look for staging variables
                staging_keywords = ['stage', 'grade', 'tumor']
                staging_cols = []
                for col in clinical_data.columns:
                    if any(keyword.lower() in col.lower() for keyword in staging_keywords):
                        staging_cols.append(col)
                
                print(f"Potential staging columns ({len(staging_cols)}): {staging_cols[:10]}")
            else:
                print("Could not find suitable clinical data structure")
                
    except Exception as e:
        print(f"Error loading clinical data: {e}")
    
    # Analyze RNA-seq data
    print("\n2. RNA-SEQ DATA ANALYSIS")
    print("-" * 30)
    
    rnaseq_dir = data_dir / "RNAseq_data"
    rds_files = list(rnaseq_dir.glob("*.rds"))
    
    print(f"Found {len(rds_files)} cancer type files")
    
    sample_counts = {}
    gene_counts = {}
    successful_loads = 0
    
    # Analyze each cancer type
    for rds_file in rds_files:
        cancer_type = rds_file.stem.replace('_data', '')
        
        try:
            result = pyreadr.read_r(str(rds_file))
            
            # Find the data
            data = None
            for key in result.keys():
                candidate = result[key]
                if hasattr(candidate, 'shape') and len(candidate.shape) == 2:
                    data = candidate
                    break
            
            if data is not None:
                sample_counts[cancer_type] = data.shape[0]
                gene_counts[cancer_type] = data.shape[1]
                successful_loads += 1
                
                # Detailed analysis for a few cancer types
                if cancer_type in ['BRCA', 'LUAD', 'ACC']:
                    print(f"\n{cancer_type} analysis:")
                    print(f"  Samples: {data.shape[0]}")
                    print(f"  Features: {data.shape[1]}")
                    print(f"  Data type: {data.dtypes.iloc[0] if hasattr(data, 'dtypes') else 'unknown'}")
                    
                    # Check if it's gene expression data
                    if hasattr(data, 'columns'):
                        sample_genes = list(data.columns[:5])
                        print(f"  Sample features: {sample_genes}")
                        
                        # Basic statistics
                        if data.shape[1] > 0:
                            first_col = data.iloc[:, 0]
                            print(f"  First feature stats: min={first_col.min():.3f}, max={first_col.max():.3f}, mean={first_col.mean():.3f}")
            else:
                print(f"Could not extract data from {cancer_type}")
                
        except Exception as e:
            print(f"Error loading {cancer_type}: {e}")
    
    print(f"\nSuccessfully loaded {successful_loads}/{len(rds_files)} cancer types")
    
    if sample_counts:
        print(f"\nSample size statistics:")
        sizes = list(sample_counts.values())
        print(f"  Range: {min(sizes)} - {max(sizes)}")
        print(f"  Mean: {np.mean(sizes):.1f}")
        print(f"  Median: {np.median(sizes):.1f}")
        
        # Top cancer types by sample size
        sorted_cancers = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 10 cancer types by sample size:")
        for cancer, count in sorted_cancers[:10]:
            print(f"  {cancer}: {count} samples")
        
        # Categories for ML feasibility
        large_cancers = {k: v for k, v in sample_counts.items() if v >= 100}
        medium_cancers = {k: v for k, v in sample_counts.items() if 50 <= v < 100}
        small_cancers = {k: v for k, v in sample_counts.items() if v < 50}
        
        print(f"\nML Feasibility Categories:")
        print(f"  Large (≥100 samples): {len(large_cancers)} types - {list(large_cancers.keys())}")
        print(f"  Medium (50-99 samples): {len(medium_cancers)} types")
        print(f"  Small (<50 samples): {len(small_cancers)} types")
    
    # Gene count consistency check
    if gene_counts:
        unique_gene_counts = set(gene_counts.values())
        print(f"\nGene count consistency:")
        print(f"  Unique gene counts: {len(unique_gene_counts)}")
        if len(unique_gene_counts) <= 5:
            print(f"  Gene counts: {sorted(unique_gene_counts)}")
    
    return sample_counts, gene_counts, clinical_data, large_cancers if 'large_cancers' in locals() else {}

def ai_feasibility_assessment(sample_counts, large_cancers, clinical_data):
    """Assess feasibility for AI approaches"""
    print("\n" + "=" * 60)
    print("AI MODELING FEASIBILITY ASSESSMENT")
    print("=" * 60)
    
    total_samples = sum(sample_counts.values()) if sample_counts else 0
    
    print("1. CAUSAL DIFFUSION NETWORKS")
    print("-" * 30)
    
    # Requirements: Large sample sizes, clinical outcomes
    suitable_for_causal = [(k, v) for k, v in large_cancers.items() if v >= 200]
    print(f"Cancer types with ≥200 samples: {len(suitable_for_causal)}")
    for cancer, count in sorted(suitable_for_causal, key=lambda x: x[1], reverse=True):
        print(f"  {cancer}: {count} samples")
    
    has_clinical = clinical_data is not None and hasattr(clinical_data, 'shape')
    print(f"Clinical data available: {'Yes' if has_clinical else 'No'}")
    
    causal_feasible = len(suitable_for_causal) >= 3 and has_clinical
    print(f"Causal modeling feasibility: {'HIGH' if causal_feasible else 'MODERATE'}")
    
    print("\n2. BioCLR CONTRASTIVE LEARNING")
    print("-" * 30)
    
    print(f"Total samples available: {total_samples:,}")
    print(f"Large cancer types: {len(large_cancers)}")
    
    contrastive_feasible = total_samples >= 1000 and len(large_cancers) >= 5
    print(f"Contrastive learning feasibility: {'HIGH' if contrastive_feasible else 'MODERATE'}")
    
    print("Potential augmentation strategies:")
    print("  - Gene dropout (simulate missing data)")
    print("  - Cross-cancer type contrasts")
    print("  - Pathway-based masking")
    print("  - Expression noise injection")
    
    print("\n3. PSEUDO-TEMPORAL MODELING")
    print("-" * 30)
    
    # This requires staging/progression information
    if has_clinical:
        print("Requires staging/progression markers in clinical data")
        print("Cross-sectional data limits temporal resolution")
    else:
        print("Limited without clinical staging information")
    
    print("Feasibility: MODERATE (depends on staging data quality)")
    
    print("\n4. GRAPH-BASED APPROACHES")
    print("-" * 30)
    
    if sample_counts:
        avg_genes = np.mean(list(gene_counts.values())) if gene_counts else 0
        print(f"Average gene/feature count: {avg_genes:.0f}")
    
    print("Gene networks can be constructed from:")
    print("  - KEGG pathways")
    print("  - GO biological processes")
    print("  - STRING protein interactions")
    print("  - Co-expression networks from data")
    
    print("Graph modeling feasibility: HIGH")

def generate_recommendations(sample_counts, large_cancers):
    """Generate specific recommendations"""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR AI RESEARCH")
    print("=" * 60)
    
    if large_cancers:
        top_cancers = sorted(large_cancers.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print("1. PRIORITY CANCER TYPES FOR MODEL DEVELOPMENT:")
        for i, (cancer, count) in enumerate(top_cancers, 1):
            print(f"  {i}. {cancer}: {count} samples")
        
        print("\n2. RECOMMENDED RESEARCH SEQUENCE:")
        print("  Phase 1: BioCLR implementation (highest success probability)")
        print("    - Start with largest cancer types")
        print("    - Develop robust augmentation strategies")
        print("    - Cross-cancer validation")
        
        print("  Phase 2: Graph-based approaches")
        print("    - Integrate pathway knowledge")
        print("    - Co-expression network analysis")
        print("    - Multi-scale graph representations")
        
        print("  Phase 3: Causal modeling (if clinical data supports)")
        print("    - Causal discovery on molecular data")
        print("    - Clinical outcome prediction")
        print("    - Treatment effect estimation")
        
        print("\n3. TECHNICAL CONSIDERATIONS:")
        print("  - Implement batch effect correction")
        print("  - Use cross-cancer validation for generalizability")
        print("  - Consider multi-modal integration if other omics available")
        print("  - Plan for external validation datasets")
        
        print("\n4. DATA PREPROCESSING PIPELINE:")
        print("  - Quality control and outlier detection")
        print("  - Normalization (already appears processed)")
        print("  - Batch effect correction (ComBat/limma)")
        print("  - Feature selection/dimensionality reduction")
        
        total_large_samples = sum(large_cancers.values())
        print(f"\nTOTAL USABLE SAMPLES: {total_large_samples:,} (from cancer types with ≥100 samples)")

if __name__ == "__main__":
    print("TCGA Dataset Comprehensive Analysis")
    print("===================================")
    
    # First debug the file structure
    debug_rds_files()
    
    # Then run simplified analysis
    sample_counts, gene_counts, clinical_data, large_cancers = simple_analysis()
    
    # Assess AI feasibility
    if sample_counts:
        ai_feasibility_assessment(sample_counts, large_cancers, clinical_data)
        generate_recommendations(sample_counts, large_cancers)
    else:
        print("Could not load sufficient data for analysis")