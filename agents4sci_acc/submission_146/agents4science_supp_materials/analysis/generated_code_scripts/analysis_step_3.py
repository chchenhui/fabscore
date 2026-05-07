import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import glob
from datetime import datetime

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

def load_and_validate_data(filepath):
    """
    Load and validate the input CSV file.
    
    Parameters:
    filepath (str): Path to the input CSV file
    
    Returns:
    pd.DataFrame: Validated DataFrame with required columns
    """
    print(f"Loading data from: {filepath}")
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Strip whitespace from column names for robustness
    df.columns = df.columns.str.strip()
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("First 3 rows:")
    print(df.head(3))
    
    # Required columns
    required_columns = [
        'user_id', 'age', 'gender', 'education_level', 'country', 'test_run_id',
        'battery_id', 'time_of_day', 'grand_index', 'subtest_36_score',
        'subtest_39_score', 'subtest_40_score', 'subtest_29_score',
        'subtest_28_score', 'subtest_33_score', 'subtest_30_score',
        'subtest_27_score', 'subtest_32_score', 'subtest_38_score',
        'subtest_37_score', 'age_bin', 'percentile_36', 'percentile_39',
        'percentile_40', 'percentile_29', 'percentile_28', 'percentile_33',
        'percentile_30', 'percentile_27', 'percentile_32', 'percentile_38',
        'percentile_37'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"All required columns present: {len(required_columns)} columns validated")
    
    # Print unique values for key experimental design parameters
    print("\nUnique values for key experimental design parameters:")
    categorical_columns = ['gender', 'education_level', 'country', 'battery_id', 'time_of_day', 'age_bin']
    for col in categorical_columns:
        if col in df.columns:
            unique_vals = df[col].unique()
            print(f"{col}: {unique_vals}")
    
    # Print data types for measured variables
    print("\nData types for measured variables:")
    measured_columns = ['age', 'grand_index'] + [col for col in df.columns if col.startswith('subtest_') or col.startswith('percentile_')]
    for col in measured_columns:
        if col in df.columns:
            print(f"{col}: {df[col].dtype}")
    
    return df

def clean_data(df):
    """
    Clean the data by removing rows with missing values in key columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    pd.DataFrame: Cleaned DataFrame
    """
    print("\nCleaning data...")
    
    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")
    
    # Define percentile columns for heterogeneity calculations
    percentile_columns = [
        'percentile_36', 'percentile_39', 'percentile_40', 'percentile_29',
        'percentile_28', 'percentile_33', 'percentile_30', 'percentile_27',
        'percentile_32', 'percentile_38', 'percentile_37'
    ]
    
    # Key columns that must not have missing values
    key_columns = ['user_id', 'grand_index'] + percentile_columns
    
    # Check for missing values in key columns
    missing_counts = {}
    for col in key_columns:
        missing_count = df[col].isna().sum()
        missing_counts[col] = missing_count
        if missing_count > 0:
            print(f"Missing values in {col}: {missing_count}")
    
    # Remove rows with missing values in key columns
    df_clean = df.dropna(subset=key_columns)
    
    final_rows = len(df_clean)
    excluded_rows = initial_rows - final_rows
    
    print(f"Final number of rows: {final_rows}")
    print(f"Excluded rows due to missing values: {excluded_rows}")
    
    if excluded_rows > 0:
        exclusion_summary = f"Exclusions summary:\n"
        exclusion_summary += f"Total excluded: {excluded_rows}\n"
        for col, count in missing_counts.items():
            if count > 0:
                exclusion_summary += f"  - {col}: {count} missing values\n"
        print(exclusion_summary)
    
    return df_clean

def calculate_heterogeneity_metrics(df):
    """
    Calculate percentile range and percentile IQR for each participant.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with percentile columns
    
    Returns:
    pd.DataFrame: DataFrame with added heterogeneity metrics
    """
    print("\nCalculating heterogeneity metrics...")
    
    # Define percentile columns
    percentile_columns = [
        'percentile_36', 'percentile_39', 'percentile_40', 'percentile_29',
        'percentile_28', 'percentile_33', 'percentile_30', 'percentile_27',
        'percentile_32', 'percentile_38', 'percentile_37'
    ]
    
    print(f"Using percentile columns: {percentile_columns}")
    
    # Extract percentile values for calculations
    percentile_data = df[percentile_columns].values
    
    # Calculate Percentile Range (max - min)
    percentile_range = np.max(percentile_data, axis=1) - np.min(percentile_data, axis=1)
    
    # Calculate Percentile IQR (75th percentile - 25th percentile) - more efficient
    percentiles_75_25 = np.percentile(percentile_data, [75, 25], axis=1, method='linear')
    percentile_iqr = percentiles_75_25[0] - percentiles_75_25[1]
    
    # Add metrics to DataFrame
    df['percentile_range'] = percentile_range
    df['percentile_iqr'] = percentile_iqr
    
    print(f"Calculated percentile_range for {len(df)} participants")
    print(f"Calculated percentile_iqr for {len(df)} participants")
    
    # Print summary statistics
    print("\nSummary statistics for heterogeneity metrics:")
    print("Percentile Range:")
    print(f"  Mean: {np.mean(percentile_range):.3f}")
    print(f"  Std: {np.std(percentile_range):.3f}")
    print(f"  Min: {np.min(percentile_range):.3f}")
    print(f"  Max: {np.max(percentile_range):.3f}")
    
    print("Percentile IQR:")
    print(f"  Mean: {np.mean(percentile_iqr):.3f}")
    print(f"  Std: {np.std(percentile_iqr):.3f}")
    print(f"  Min: {np.min(percentile_iqr):.3f}")
    print(f"  Max: {np.max(percentile_iqr):.3f}")
    
    return df

def validate_discriminant_validity(df):
    """
    Validate that heterogeneity metrics are independent of general ability.
    
    Parameters:
    df (pd.DataFrame): DataFrame with heterogeneity metrics and grand_index
    
    Returns:
    pd.DataFrame: Validation results DataFrame
    """
    print("\nValidating discriminant validity...")
    
    # Calculate correlations with grand_index
    range_corr, range_p = pearsonr(df['percentile_range'], df['grand_index'])
    iqr_corr, iqr_p = pearsonr(df['percentile_iqr'], df['grand_index'])
    
    print(f"Correlation between percentile_range and grand_index: r = {range_corr:.4f}, p = {range_p:.4f}")
    print(f"Correlation between percentile_iqr and grand_index: r = {iqr_corr:.4f}, p = {iqr_p:.4f}")
    
    # Check discriminant validity criteria (|r| < 0.20)
    range_valid = abs(range_corr) < 0.20
    iqr_valid = abs(iqr_corr) < 0.20
    
    print(f"Discriminant validity for percentile_range: {'Met' if range_valid else 'Not met'} (|r| = {abs(range_corr):.4f})")
    print(f"Discriminant validity for percentile_iqr: {'Met' if iqr_valid else 'Not met'} (|r| = {abs(iqr_corr):.4f})")
    
    # Create validation results DataFrame
    validation_results = pd.DataFrame({
        'metric_name': ['percentile_range', 'percentile_iqr'],
        'correlation_with_grand_index': [range_corr, iqr_corr],
        'correlation_p_value': [range_p, iqr_p],
        'discriminant_validity_met': [range_valid, iqr_valid]
    })
    
    print("\nValidation results:")
    print(validation_results)
    
    return validation_results

def save_outputs(df, validation_results):
    """
    Save output files with timestamp.
    
    Parameters:
    df (pd.DataFrame): Main DataFrame with heterogeneity metrics
    validation_results (pd.DataFrame): Validation results DataFrame
    """
    print("\nSaving output files...")
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define output columns for main file
    output_columns = [
        'user_id', 'age', 'gender', 'education_level', 'country', 'test_run_id',
        'battery_id', 'time_of_day', 'grand_index', 'subtest_36_score',
        'subtest_39_score', 'subtest_40_score', 'subtest_29_score',
        'subtest_28_score', 'subtest_33_score', 'subtest_30_score',
        'subtest_27_score', 'subtest_32_score', 'subtest_38_score',
        'subtest_37_score', 'age_bin', 'percentile_36', 'percentile_39',
        'percentile_40', 'percentile_29', 'percentile_28', 'percentile_33',
        'percentile_30', 'percentile_27', 'percentile_32', 'percentile_38',
        'percentile_37', 'percentile_range', 'percentile_iqr'
    ]
    
    # Save main output file
    main_output_file = f'outputs/step3_heterogeneity_metrics_{timestamp}.csv'
    df[output_columns].to_csv(main_output_file, index=False)
    print(f"Main output saved to: {main_output_file}")
    print(f"Main output shape: {df[output_columns].shape}")
    
    # Save validation results
    validation_output_file = f'outputs/step3_validation_results_{timestamp}.csv'
    validation_results.to_csv(validation_output_file, index=False)
    print(f"Validation results saved to: {validation_output_file}")
    print(f"Validation results shape: {validation_results.shape}")
    
    # Print first 2 rows of main output for verification
    print("\nFirst 2 rows of main output:")
    print("Columns:", list(df[output_columns].columns))
    print(df[output_columns].head(2))
    
    # Print validation results for verification
    print("\nValidation results:")
    print("Columns:", list(validation_results.columns))
    print(validation_results)

def main():
    """
    Main function to execute the heterogeneity metrics calculation.
    
    Returns:
    int: 0 for success, 1 for error
    """
    try:
        print("Starting Step 3: Heterogeneity Metrics Calculation")
        print("=" * 60)
        
        # Find the latest input file
        input_pattern = 'outputs/step2_percentile_rankings_*.csv'
        input_file = find_latest_file(input_pattern)
        
        # Load and validate data
        df = load_and_validate_data(input_file)
        
        # Clean data
        df_clean = clean_data(df)
        
        # Calculate heterogeneity metrics
        df_with_metrics = calculate_heterogeneity_metrics(df_clean)
        
        # Validate discriminant validity
        validation_results = validate_discriminant_validity(df_with_metrics)
        
        # Save outputs
        save_outputs(df_with_metrics, validation_results)
        
        print("\n" + "=" * 60)
        print("Step 3 completed successfully!")
        print("Finished execution")
        return 0
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
