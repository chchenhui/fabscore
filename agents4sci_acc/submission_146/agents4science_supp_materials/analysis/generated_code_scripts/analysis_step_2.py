import pandas as pd
import numpy as np
from scipy import stats
import os
import glob
from datetime import datetime
import sys

def find_latest_file(pattern):
    """
    Find the most recent file matching the given pattern.
    
    Parameters:
    pattern (str): File pattern to match
    
    Returns:
    str: Path to the most recent file matching the pattern
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    # Sort by modification time and return the most recent
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def calculate_percentile_ranks(df, subtest_columns, age_bin_col='age_bin'):
    """
    Calculate percentile ranks for subtest scores within age bins.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with subtest scores
    subtest_columns (list): List of subtest column names
    age_bin_col (str): Column name for age bins
    
    Returns:
    pd.DataFrame: Dataframe with added percentile rank columns
    """
    df_copy = df.copy()
    
    print("Calculating percentile ranks within age bins...")
    
    # Create percentile columns for each subtest
    for subtest in subtest_columns:
        percentile_col = f"percentile_{subtest.split('_')[1]}"
        print(f"Processing {subtest} -> {percentile_col}")

        def safe_percentile_transform(x):
            if x.dropna().shape[0] < 2:
                # Not enough data to compute percentiles, return NaN
                return pd.Series([np.nan] * len(x), index=x.index)
            try:
                return pd.Series([stats.percentileofscore(x.dropna(), val, kind='rank')
                                  for val in x], index=x.index)
            except Exception as e:
                print(f"  Error calculating percentiles for {subtest} in age_bin group: {e}")
                return pd.Series([np.nan] * len(x), index=x.index)

        # Calculate percentile ranks within each age bin
        df_copy[percentile_col] = df_copy.groupby(age_bin_col)[subtest].transform(
            safe_percentile_transform
        )

    return df_copy

def perform_quality_control(df, subtest_columns, age_bin_col='age_bin'):
    """
    Perform quality control checks on percentile distributions.
    
    Parameters:
    df (pd.DataFrame): Dataframe with percentile rank columns
    subtest_columns (list): List of subtest column names
    age_bin_col (str): Column name for age bins
    
    Returns:
    pd.DataFrame: Quality control results
    """
    print("Performing quality control checks...")
    
    qc_results = []
    
    for subtest in subtest_columns:
        subtest_id = subtest.split('_')[1]
        percentile_col = f"percentile_{subtest_id}"
        
        print(f"Quality control for {subtest} (percentile column: {percentile_col})")
        
        # Check each age bin
        for age_bin in df[age_bin_col].unique():
            if pd.isna(age_bin):
                continue
                
            age_bin_data = df[df[age_bin_col] == age_bin][percentile_col].dropna()
            
            if len(age_bin_data) < 2:
                print(f"  Warning: Insufficient data for {age_bin} in {subtest}")
                continue
            
            # Check range
            min_val = age_bin_data.min()
            max_val = age_bin_data.max()
            print(f"  {age_bin}: Range {min_val:.2f} - {max_val:.2f}")
            
            # Kolmogorov-Smirnov test for uniformity
            # Test against theoretical uniform distribution [0, 1]
            ks_stat, ks_p = stats.kstest(age_bin_data/100, 'uniform')

            distribution_uniform = ks_p > 0.05  # Not significantly different from uniform

            qc_results.append({
                'age_bin': age_bin,
                'subtest_id': subtest_id,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'distribution_uniform': distribution_uniform
            })
            
            print(f"  {age_bin}: KS statistic = {ks_stat:.4f}, p-value = {ks_p:.4f}, uniform = {distribution_uniform}")
    
    return pd.DataFrame(qc_results)

def main():
    """
    Main function to execute the percentile ranking task.
    
    Returns:
    int: 0 for success, 1 for error
    """
    try:
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        # Find the most recent input file
        input_pattern = 'outputs/step1_cleaned_battery26_data_*.csv'
        print(f"Looking for input files matching pattern: {input_pattern}")
        
        input_file = find_latest_file(input_pattern)
        print(f"Found input file: {input_file}")
        
        # Load the data
        print("Loading data...")
        df = pd.read_csv(input_file)
        
        print(f"Data shape: {df.shape}")
        print("All columns:")
        print(df.columns.tolist())
        
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # Required columns
        required_columns = [
            'user_id', 'age', 'gender', 'education_level', 'country',
            'test_run_id', 'battery_id', 'time_of_day', 'grand_index',
            'subtest_36_score', 'subtest_39_score', 'subtest_40_score',
            'subtest_29_score', 'subtest_28_score', 'subtest_33_score',
            'subtest_30_score', 'subtest_27_score', 'subtest_32_score',
            'subtest_38_score', 'subtest_37_score', 'age_bin'
        ]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return 1
        
        print("All required columns present.")
        
        # Print unique values for key experimental design parameters
        print("\nUnique values for key experimental design parameters:")
        print(f"age_bin unique values: {sorted(df['age_bin'].unique())}")
        print(f"battery_id unique values: {sorted(df['battery_id'].unique())}")
        print(f"gender unique values: {sorted(df['gender'].unique())}")
        print(f"education_level unique values: {sorted(df['education_level'].unique())}")
        print(f"time_of_day unique values: {sorted(df['time_of_day'].unique())}")
        
        # Print data types for measured variables
        subtest_columns = [col for col in df.columns if col.startswith('subtest_') and col.endswith('_score')]
        print(f"\nSubtest score columns data types:")
        for col in subtest_columns:
            print(f"{col}: {df[col].dtype}")
        
        print(f"grand_index data type: {df['grand_index'].dtype}")
        
        # Filter out rows with null values in critical columns
        print("\nFiltering out rows with null values in critical columns...")
        initial_rows = len(df)
        
        # Remove rows with null values in age_bin or any subtest scores
        critical_columns = ['age_bin'] + subtest_columns
        df_clean = df.dropna(subset=critical_columns)
        
        final_rows = len(df_clean)
        excluded_rows = initial_rows - final_rows
        
        print(f"Initial rows: {initial_rows}")
        print(f"Final rows: {final_rows}")
        print(f"Excluded rows: {excluded_rows}")
        
        if final_rows == 0:
            print("Error: No valid data remaining after filtering")
            return 1
        
        # Verify age bins
        expected_age_bins = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-99']
        actual_age_bins = sorted(df_clean['age_bin'].unique())

        print(f"\nExpected age bins: {expected_age_bins}")
        print(f"Actual age bins: {actual_age_bins}")
        
        # Check if we have all expected age bins
        missing_age_bins = [bin for bin in expected_age_bins if bin not in actual_age_bins]
        if missing_age_bins:
            print(f"Warning: Missing age bins: {missing_age_bins}")
        
        # Calculate percentile ranks
        df_with_percentiles = calculate_percentile_ranks(df_clean, subtest_columns)
        
        print("\nDataframe with percentiles - First 2 rows:")
        print(df_with_percentiles.head(2))
        
        # Perform quality control
        qc_results = perform_quality_control(df_with_percentiles, subtest_columns)
        
        print("\nQuality control results:")
        print(qc_results)
        
        # Generate timestamp for output files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main output file
        output_file = f'outputs/step2_percentile_rankings_{timestamp}.csv'
        
        # Select required output columns
        output_columns = [
            'user_id', 'age', 'gender', 'education_level', 'country',
            'test_run_id', 'battery_id', 'time_of_day', 'grand_index',
            'subtest_36_score', 'subtest_39_score', 'subtest_40_score',
            'subtest_29_score', 'subtest_28_score', 'subtest_33_score',
            'subtest_30_score', 'subtest_27_score', 'subtest_32_score',
            'subtest_38_score', 'subtest_37_score', 'age_bin',
            'percentile_36', 'percentile_39', 'percentile_40',
            'percentile_29', 'percentile_28', 'percentile_33',
            'percentile_30', 'percentile_27', 'percentile_32',
            'percentile_38', 'percentile_37'
        ]
        
        df_output = df_with_percentiles[output_columns]
        df_output.to_csv(output_file, index=False)
        print(f"Saved main output to: {output_file}")
        
        # Save quality control results
        qc_output_file = f'outputs/step2_quality_control_{timestamp}.csv'
        qc_results.to_csv(qc_output_file, index=False)
        print(f"Saved quality control results to: {qc_output_file}")
        
        # Final summary
        print(f"\nFinal summary:")
        print(f"Total participants processed: {len(df_output)}")
        print(f"Age bins represented: {len(df_output['age_bin'].unique())}")
        print(f"Subtests processed: {len(subtest_columns)}")
        print(f"Percentile columns created: {len([col for col in df_output.columns if col.startswith('percentile_')])}")
        
        # Check percentile ranges
        percentile_columns = [col for col in df_output.columns if col.startswith('percentile_')]
        print(f"\nPercentile column ranges:")
        for col in percentile_columns:
            min_val = df_output[col].min()
            max_val = df_output[col].max()
            print(f"{col}: {min_val:.2f} - {max_val:.2f}")
        
        print("Finished execution")
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
