"""
Test data exploration and validation.
Tests the collected pharmaceutical data for quality and completeness.
"""

import pytest
import pandas as pd
from pathlib import Path


class TestDataExploration:
    """Test data exploration and validation."""
    
    def setup_method(self):
        """Setup data paths."""
        self.data_dir = Path(__file__).parent.parent / "data_proc"
        self.launches_path = self.data_dir / "launches.parquet"
        self.revenues_path = self.data_dir / "launch_revenues.parquet"
    
    def test_data_files_exist(self):
        """Test that data files exist."""
        assert self.launches_path.exists(), "launches.parquet not found"
        assert self.revenues_path.exists(), "launch_revenues.parquet not found"
    
    def test_launches_data_structure(self):
        """Test launches data structure and quality."""
        if not self.launches_path.exists():
            pytest.skip("launches.parquet not found")
        
        df_l = pd.read_parquet(self.launches_path)
        
        # Check basic structure
        assert len(df_l) > 0, "No launches data found"
        assert len(df_l.columns) > 0, "No columns in launches data"
        
        # Check required columns
        required_columns = ['launch_id', 'drug_name', 'company', 'therapeutic_area']
        for col in required_columns:
            assert col in df_l.columns, f"Missing required column: {col}"
        
        # Check data quality
        assert df_l['launch_id'].nunique() == len(df_l), "Duplicate launch_ids found"
        assert df_l['drug_name'].notna().all(), "Missing drug names"
        assert df_l['therapeutic_area'].notna().all(), "Missing therapeutic areas"
    
    def test_revenues_data_structure(self):
        """Test revenues data structure and quality."""
        if not self.revenues_path.exists():
            pytest.skip("launch_revenues.parquet not found")
        
        df_r = pd.read_parquet(self.revenues_path)
        
        # Check basic structure
        assert len(df_r) > 0, "No revenues data found"
        assert len(df_r.columns) > 0, "No columns in revenues data"
        
        # Check required columns
        required_columns = ['launch_id', 'year_since_launch', 'revenue_usd']
        for col in required_columns:
            assert col in df_r.columns, f"Missing required column: {col}"
        
        # Check data quality
        assert df_r['revenue_usd'].notna().all(), "Missing revenue values"
        assert (df_r['revenue_usd'] >= 0).all(), "Negative revenue values found"
        assert df_r['year_since_launch'].notna().all(), "Missing year_since_launch values"
    
    def test_data_consistency(self):
        """Test consistency between launches and revenues data."""
        if not self.launches_path.exists() or not self.revenues_path.exists():
            pytest.skip("Data files not found")
        
        df_l = pd.read_parquet(self.launches_path)
        df_r = pd.read_parquet(self.revenues_path)
        
        # Check that all revenue launch_ids exist in launches
        revenue_launch_ids = set(df_r['launch_id'].unique())
        launch_ids = set(df_l['launch_id'].unique())
        
        missing_launches = revenue_launch_ids - launch_ids
        assert len(missing_launches) == 0, f"Revenue data for non-existent launches: {missing_launches}"
    
    def test_data_completeness(self):
        """Test data completeness for G1 gate requirements."""
        if not self.launches_path.exists() or not self.revenues_path.exists():
            pytest.skip("Data files not found")
        
        df_l = pd.read_parquet(self.launches_path)
        df_r = pd.read_parquet(self.revenues_path)
        
        # G1 Gate: N ≥ 50 launches
        assert len(df_l) >= 50, f"G1 Gate failed: Only {len(df_l)} launches (need ≥50)"
        
        # G1 Gate: ≥ 5 therapeutic areas
        therapeutic_areas = df_l['therapeutic_area'].nunique()
        assert therapeutic_areas >= 5, f"G1 Gate failed: Only {therapeutic_areas} therapeutic areas (need ≥5)"
        
        # G1 Gate: Y1–Y5 revenue ground truth
        years_covered = df_r['year_since_launch'].unique()
        required_years = set(range(6))  # 0-5 years
        missing_years = required_years - set(years_covered)
        assert len(missing_years) == 0, f"G1 Gate failed: Missing years {missing_years}"
    
    def test_data_statistics(self):
        """Test data statistics and quality metrics."""
        if not self.launches_path.exists() or not self.revenues_path.exists():
            pytest.skip("Data files not found")
        
        df_l = pd.read_parquet(self.launches_path)
        df_r = pd.read_parquet(self.revenues_path)
        
        # Print data summary (for manual inspection)
        print(f"\n=== DATA SUMMARY ===")
        print(f"Launches: {len(df_l)} drugs")
        print(f"Revenue records: {len(df_r)}")
        print(f"Therapeutic areas: {df_l['therapeutic_area'].value_counts().to_dict()}")
        print(f"Total revenue: ${df_r['revenue_usd'].sum():,.2f}")
        print(f"Average annual revenue: ${df_r['revenue_usd'].mean():,.2f}")
        print(f"Median annual revenue: ${df_r['revenue_usd'].median():,.2f}")
        
        # Check for reasonable revenue ranges
        max_revenue = df_r['revenue_usd'].max()
        min_revenue = df_r['revenue_usd'].min()
        
        assert max_revenue > 0, "No positive revenue values found"
        assert min_revenue >= 0, "Negative revenue values found"
        
        # Check for reasonable revenue distribution
        assert max_revenue < 1e12, "Unrealistically high revenue values found (>$1T)"
        assert min_revenue < 1e9, "All revenue values seem too high (>$1B)"


def explore_data():
    """Standalone data exploration function (for manual use)."""
    print('=== EXPLORING REAL PHARMACEUTICAL DATA ===')
    print()
    
    data_dir = Path(__file__).parent.parent / "data_proc"
    launches_path = data_dir / "launches.parquet"
    revenues_path = data_dir / "launch_revenues.parquet"
    
    if not launches_path.exists() or not revenues_path.exists():
        print("Data files not found. Run data collection first.")
        return
    
    # Load the data
    df_l = pd.read_parquet(launches_path)
    df_r = pd.read_parquet(revenues_path)
    
    print('📊 LAUNCHES DATASET:')
    print(f'Shape: {df_l.shape[0]} drugs, {df_l.shape[1]} columns')
    print(f'Therapeutic areas: {df_l["therapeutic_area"].value_counts().to_dict()}')
    print()
    
    print('🔍 SAMPLE LAUNCHES:')
    print(df_l[['drug_name', 'company', 'therapeutic_area', 'approval_date']].head(10))
    print()
    
    print('💰 REVENUES DATASET:')
    print(f'Shape: {df_r.shape[0]} revenue records, {df_r.shape[1]} columns')
    print(f'Years covered: {sorted(df_r["year_since_launch"].unique())}')
    print(f'Total revenue: ${df_r["revenue_usd"].sum():,.2f}')
    print(f'Average annual revenue per drug (Y0-Y5): ${df_r["revenue_usd"].mean():,.2f}')
    print(f'Median annual revenue per drug (Y0-Y5): ${df_r["revenue_usd"].median():,.2f}')
    print()
    
    print('📈 SAMPLE REVENUES:')
    print(df_r.head(10))
    print()
    
    print('🎯 TOP REVENUE DRUGS (Year 5):')
    year5_revenues = df_r[df_r['year_since_launch'] == 5].nlargest(10, 'revenue_usd')
    print(year5_revenues[['launch_id', 'revenue_usd']])
    print()
    
    print('🏆 TOP COMPANIES BY TOTAL REVENUE (Y0-Y5):')
    # Merge revenues with launches to get company names
    merged_df = pd.merge(df_r, df_l[['launch_id', 'company']], on='launch_id', how='left')
    company_revenues = merged_df.groupby('company')['revenue_usd'].sum().nlargest(5)
    print(company_revenues.apply(lambda x: f'${x:,.2f}'))
    print()


if __name__ == "__main__":
    # Run data exploration
    explore_data()
    
    # Run tests
    pytest.main([__file__, "-v"])
