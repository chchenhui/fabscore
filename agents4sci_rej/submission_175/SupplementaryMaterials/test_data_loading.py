#!/usr/bin/env python3
"""
Simple test script to verify TCGA data loading and preprocessing
"""
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'code' / 'src'))

from data.preprocessing import TCGAPathwayDataset

def test_data_loading():
    """Test TCGA data loading functionality"""
    print("="*50)
    print("TESTING TCGA DATA LOADING")
    print("="*50)
    
    try:
        # Initialize dataset
        dataset = TCGAPathwayDataset(
            data_dir='data/RNAseq_data',
            normalization='quantile',
            min_samples_per_cancer=30,  # Lower for testing
            test_size=0.2,
            val_size=0.1
        )
        
        print("✓ TCGAPathwayDataset initialized successfully")
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        data_splits = dataset.load_and_preprocess()
        
        print("✓ Data loading completed!")
        
        # Display summary statistics
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        
        for split_name, split_data in data_splits.items():
            n_samples = len(split_data['cancer_types'])
            n_features = split_data['pathway_data'].shape[1]
            n_cancer_types = len(set(split_data['cancer_types']))
            
            print(f"\n{split_name.upper()} SET:")
            print(f"  Samples: {n_samples}")
            print(f"  Features (pathways): {n_features}")
            print(f"  Cancer types: {n_cancer_types}")
            
            # Show cancer type distribution
            from collections import Counter
            cancer_counts = Counter(split_data['cancer_types'])
            print(f"  Cancer distribution: {dict(cancer_counts)}")
        
        print("\n✅ SUCCESS: Data preprocessing completed successfully!")
        print(f"✅ Total pathway features: {n_features}")
        print(f"✅ Data ready for meta-learning!")
        
        return data_splits
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test data loading
    data_splits = test_data_loading()
    
    if data_splits:
        print(f"\n🎉 READY TO PROCEED WITH FULL PIPELINE!")
        print("Next steps:")
        print("1. Run baseline models")
        print("2. Train hierarchical meta-learning model")
        print("3. Evaluate and analyze results")
    else:
        print(f"\n❌ DATA LOADING FAILED - Please check data directory and files")