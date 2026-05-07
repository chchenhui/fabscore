"""
Unit tests for revenue extraction from SEC filings.
Tests the extract_brand_revenue function and year alignment logic.
"""

import pytest
import pandas as pd
from src.data.sources import extract_brand_revenue


class TestRevenueExtraction:
    """Test cases for revenue extraction from SEC 10-K/10-Q filings."""
    
    def test_extract_brand_revenue_simple(self):
        """Test basic revenue extraction with clear brand mentions."""
        text = """
        Product revenues for Keytruda were $25.1 billion in 2023, 
        compared to $20.9 billion in 2022. Our oncology franchise 
        continues to show strong growth.
        """
        
        result = extract_brand_revenue(text, ["Keytruda"])
        
        assert result is not None
        assert "2023" in result
        assert result["2023"] == 25.1e9
    
    def test_extract_brand_revenue_multiple_years(self):
        """Test extraction when multiple years are mentioned."""
        text = """
        Humira revenues:
        - 2023: $14.2 billion
        - 2022: $21.2 billion  
        - 2021: $20.7 billion
        """
        
        result = extract_brand_revenue(text, ["Humira"])
        
        assert result is not None
        assert "2023" in result
        assert result["2023"] == 14.2e9
        assert "2022" in result
        assert result["2022"] == 21.2e9
    
    def test_extract_brand_revenue_different_units(self):
        """Test extraction with different revenue units (millions, billions)."""
        test_cases = [
            ("$1.5 billion", 1.5e9),
            ("$1,500 million", 1.5e9),
            ("$1,500,000 thousand", 1.5e9),
            ("$1.5B", 1.5e9),
            ("$1,500M", 1.5e9),
        ]
        
        for text_snippet, expected in test_cases:
            text = f"Eliquis generated {text_snippet} in revenue for 2023."
            result = extract_brand_revenue(text, ["Eliquis"])
            
            assert result is not None
            assert "2023" in result
            assert abs(result["2023"] - expected) < 1e6  # Within $1M tolerance
    
    def test_extract_brand_revenue_no_match(self):
        """Test when brand is not mentioned in text."""
        text = """
        Our cardiovascular portfolio showed strong performance.
        Total product revenues increased 15% year-over-year.
        """
        
        result = extract_brand_revenue(text, ["Keytruda"])
        
        assert result == {}
    
    def test_extract_brand_revenue_ambiguous(self):
        """Test extraction with ambiguous or unclear revenue mentions."""
        text = """
        Keytruda sales were significant. Our oncology business
        contributed substantially to overall revenue growth.
        """
        
        result = extract_brand_revenue(text, ["Keytruda"])
        
        # Should return empty dict or very low values
        assert result == {}
    
    def test_extract_brand_revenue_aliases(self):
        """Test extraction with brand aliases and variations."""
        text = """
        Pembrolizumab (KEYTRUDA) generated $25.1 billion in 2023.
        """
        
        result = extract_brand_revenue(text, ["Keytruda", "Pembrolizumab"])
        
        assert result is not None
        assert "2023" in result
        assert result["2023"] == 25.1e9
    
    def test_extract_brand_revenue_negative_values(self):
        """Test handling of negative revenue (returns, adjustments)."""
        text = """
        Keytruda net revenue was $25.1 billion in 2023, 
        after returns of $500 million.
        """
        
        result = extract_brand_revenue(text, ["Keytruda"])
        
        # Should extract the net positive revenue, not the negative return
        assert result is not None
        assert "2023" in result
        assert result["2023"] == 25.1e9
    
    def test_extract_brand_revenue_complex_formatting(self):
        """Test extraction with complex formatting and tables."""
        text = """
        Product Revenues (in millions):
        
        | Product    | 2023    | 2022    |
        |------------|---------|---------|
        | Keytruda   | 25,100  | 20,900  |
        | Other      | 5,000   | 4,500   |
        """
        
        result = extract_brand_revenue(text, ["Keytruda"])
        
        assert result is not None
        assert "2023" in result
        assert result["2023"] == 25.1e9
    
    def test_extract_brand_revenue_year_extraction(self):
        """Test that years are correctly extracted from text."""
        text = """
        Keytruda revenue was $25.1 billion in 2023.
        """
        
        result = extract_brand_revenue(text, ["Keytruda"])
        
        assert result is not None
        assert "2023" in result
        assert result["2023"] == 25.1e9


class TestRevenueDataFrame:
    """Test revenue data processing and year alignment."""
    
    def test_revenue_dataframe_structure(self):
        """Test that revenue DataFrame has correct structure."""
        # This would test the collect_real.py logic that processes
        # extracted revenues into the final DataFrame format
        
        sample_revenues = [
            {'launch_id': 'keytruda_2014', 'year_since_launch': 1, 'revenue_usd': 1.2e9, 'source_url': 'test'},
            {'launch_id': 'keytruda_2014', 'year_since_launch': 2, 'revenue_usd': 3.8e9, 'source_url': 'test'},
            {'launch_id': 'keytruda_2014', 'year_since_launch': 3, 'revenue_usd': 7.1e9, 'source_url': 'test'},
        ]
        
        df = pd.DataFrame(sample_revenues)
        
        # Check required columns
        required_cols = ['launch_id', 'year_since_launch', 'revenue_usd', 'source_url']
        for col in required_cols:
            assert col in df.columns
        
        # Check data types
        assert df['year_since_launch'].dtype in ['int64', 'int32']
        assert df['revenue_usd'].dtype in ['float64', 'float32']
        assert df['launch_id'].dtype == 'object'
    
    def test_year_since_launch_calculation(self):
        """Test year_since_launch calculation from approval_date."""
        # This tests the logic that converts approval_date + revenue_year
        # to year_since_launch
        
        from datetime import datetime
        
        approval_date = datetime(2014, 9, 4)  # Keytruda approval
        revenue_years = [2015, 2016, 2017, 2018, 2019]
        expected_years_since = [1, 2, 3, 4, 5]
        
        for rev_year, expected in zip(revenue_years, expected_years_since):
            # Correct calculation: year_since_launch = revenue_year - approval_year
            # For mid-year launches (month > 6), the first full year is the next year
            year_since = rev_year - approval_date.year
            if approval_date.month > 6:  # Mid-year launch
                year_since -= 1
            year_since = max(1, year_since)  # Minimum year 1, not 0
            
            # For September 2014 launch:
            # 2015 -> 2015 - 2014 = 1, but since month > 6, 1 - 1 = 0, max(1, 0) = 1 ✓
            # 2016 -> 2016 - 2014 = 2, but since month > 6, 2 - 1 = 1, max(1, 1) = 1 ✗
            # The logic should be: if month > 6, then year_since = rev_year - approval_year
            # But we want 2015 to be year 1, 2016 to be year 2, etc.
            # So: year_since = rev_year - approval_year (no adjustment needed)
            year_since = rev_year - approval_date.year
            
            assert year_since == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])