import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

def load_and_validate_data(file_path):
    """
    Load the raw data from CSV file and validate required columns.
    
    Parameters:
    file_path (str): Path to the input CSV file
    
    Returns:
    pd.DataFrame: Loaded dataframe with validated columns
    """
    print(f"Loading data from: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("First 3 rows:")
    print(df.head(3))
    
    # Required columns
    required_columns = [
        'user_id', 'age', 'gender', 'education_level', 'country',
        'test_run_id', 'battery_id', 'specific_subtest_id', 'time_of_day',
        'raw_score', 'grand_index'
    ]
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print("All required columns present.")
    
    # Print unique values for key experimental design parameters
    print(f"Unique battery_id values: {sorted(df['battery_id'].unique())}")
    print(f"Unique specific_subtest_id values: {sorted(df['specific_subtest_id'].unique())}")
    print(f"Unique time_of_day values: {sorted(df['time_of_day'].unique())}")
    print(f"Unique gender values: {df['gender'].unique()}")
    print(f"Unique education_level values: {sorted(df['education_level'].unique())}")
    print(f"Age data type: {df['age'].dtype}")
    print(f"Raw_score data type: {df['raw_score'].dtype}")
    print(f"Grand_index data type: {df['grand_index'].dtype}")
    
    return df

def verify_subtests(df):
    """
    Verify that all expected subtests are present in the data.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    bool: True if all subtests are present
    """
    expected_subtests = [36, 39, 40, 29, 28, 33, 30, 27, 32, 38, 37]
    actual_subtests = df['specific_subtest_id'].unique()
    
    print(f"Expected subtests: {expected_subtests}")
    print(f"Actual subtests: {sorted(actual_subtests)}")
    
    missing_subtests = [st for st in expected_subtests if st not in actual_subtests]
    if missing_subtests:
        print(f"WARNING: Missing subtests: {missing_subtests}")
        return False
    
    print("All expected subtests are present.")
    return True

def reshape_to_wide_format(df):
    """
    Reshape data from long format to wide format with one row per participant.
    
    Parameters:
    df (pd.DataFrame): Input dataframe in long format
    
    Returns:
    pd.DataFrame: Reshaped dataframe in wide format
    """
    print("Reshaping data from long to wide format...")
    
    # Remove rows with null values in critical columns
    df_clean = df.dropna(subset=['user_id', 'specific_subtest_id', 'raw_score'])
    print(f"Removed {len(df) - len(df_clean)} rows with null values in critical columns")
    
    # Get demographic and test info (taking first occurrence for each user/test_run)
    demo_cols = ['user_id', 'age', 'gender', 'education_level', 'country', 
                 'test_run_id', 'battery_id', 'time_of_day', 'grand_index']
    demo_df = df_clean[demo_cols].drop_duplicates(subset=['user_id', 'test_run_id']).reset_index(drop=True)
    
    # Pivot the subtest scores
    score_pivot = df_clean.pivot_table(
        index=['user_id', 'test_run_id'], 
        columns='specific_subtest_id', 
        values='raw_score', 
        aggfunc='first'
    ).reset_index()
    
    # Rename columns to match required format
    score_pivot.columns = ['user_id', 'test_run_id'] + [f'subtest_{int(col)}_score' for col in score_pivot.columns[2:]]
    
    # Merge demographic data with scores
    wide_df = demo_df.merge(score_pivot, on=['user_id', 'test_run_id'], how='inner')
    
    print(f"Wide format data shape: {wide_df.shape}")
    print(f"Wide format columns: {list(wide_df.columns)}")
    print("First 5 rows of wide format data:")
    with pd.option_context('display.max_columns', None):
        print(wide_df.head(5))
    
    return wide_df

def apply_exclusion_criteria(df):
    """
    Apply all exclusion criteria and track exclusions.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    tuple: (cleaned_df, exclusion_log)
    """
    print("\n" + "="*60)
    print("Applying exclusion criteria...")
    print("="*60)
    
    exclusion_log = []
    initial_count = len(df)
    print(f"Initial participant count: {initial_count}")
    
    # 1. Timing-based exclusions
    print("\n=== TIMING-BASED EXCLUSIONS ===")
    before_count = len(df)
    trail_mask = ((df['subtest_39_score'] > 900) | (df['subtest_40_score'] > 900))
    df = df[~trail_mask]
    excluded_timing = before_count - len(df)
    exclusion_log.append(('Trail Making >15 minutes', excluded_timing))
    print(f"Excluded {excluded_timing} participants taking >15 minutes on Trail Making A or B")
    
    # 2. Completion-based exclusions
    print("\n=== COMPLETION-BASED EXCLUSIONS ===")
    
    # Exclude participants missing grand_index
    before_count = len(df)
    df = df.dropna(subset=['grand_index'])
    excluded_grand_index = before_count - len(df)
    exclusion_log.append(('Missing grand_index', excluded_grand_index))
    print(f"Excluded {excluded_grand_index} participants missing grand_index scores")
    
    # Exclude participants missing essential demographic data
    before_count = len(df)
    df = df.dropna(subset=['age', 'gender', 'education_level'])
    excluded_demo = before_count - len(df)
    exclusion_log.append(('Missing demographics', excluded_demo))
    print(f"Excluded {excluded_demo} participants missing essential demographic data")
    
    # Convert to integer types after removing missing values
    df['age'] = df['age'].astype(int)
    df['education_level'] = df['education_level'].astype(int)
    
    # Exclude participants with education_level = 99
    before_count = len(df)
    df = df[df['education_level'] != 99]
    excluded_edu99 = before_count - len(df)
    exclusion_log.append(('Education level 99', excluded_edu99))
    print(f"Excluded {excluded_edu99} participants with education_level = 99")
    
    # Exclude participants with missing data on >2 of the 11 subtests
    subtest_cols = [f'subtest_{st}_score' for st in [36, 39, 40, 29, 28, 33, 30, 27, 32, 38, 37]]
    before_count = len(df)
    missing_counts = df[subtest_cols].isnull().sum(axis=1)
    df = df[missing_counts <= 2]
    excluded_missing_subtests = before_count - len(df)
    exclusion_log.append(('Missing >2 subtests', excluded_missing_subtests))
    print(f"Excluded {excluded_missing_subtests} participants missing >2 subtests")
    
    # 3. Performance validity exclusions
    print("\n=== PERFORMANCE VALIDITY EXCLUSIONS ===")
    
    # Exclude participants with identical scores across ≥8 subtests
    before_count = len(df)
    subtest_data = df[subtest_cols].fillna(-999)  # Fill NaN with sentinel value
    identical_counts = []
    for idx, row in subtest_data.iterrows():
        # Count the most frequent value (mode) occurrence
        value_counts = pd.Series(row.values).value_counts()
        max_count = value_counts.max()
        identical_counts.append(max_count)
    
    df = df[np.array(identical_counts) < 8]
    excluded_identical = before_count - len(df)
    exclusion_log.append(('Identical scores ≥8 subtests', excluded_identical))
    print(f"Excluded {excluded_identical} participants with identical scores across ≥8 subtests")
    
    # Go/no-go accuracy exclusion (ID 32)
    # before_count = len(df)
    # gonogo_col = 'subtest_32_score'
    # if gonogo_col in df.columns:
    #     # Assume raw_score is already a proportion (0-1)
    #     df = df[(df[gonogo_col].isna()) | (df[gonogo_col] >= 0.5)]
    #     excluded_gonogo = before_count - len(df)
    #     exclusion_log.append(('Go/no-go accuracy <50%', excluded_gonogo))
    #     print(f"Excluded {excluded_gonogo} participants with Go/no-go accuracy <50%")
    
    return df, exclusion_log

def reverse_score_subtests(df):
    """
    Apply reverse scoring to time-based subtests within age bins.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Dataframe with reverse-scored subtests
    """
    print("\n=== REVERSE SCORING TIME-BASED SUBTESTS ===")
    
    # Create age bins
    df['age_bin'] = pd.cut(df['age'], 
                          bins=[18, 30, 40, 50, 60, 70, 100], 
                          labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-99'],
                          right=False)
    
    print(f"Age bin distribution:")
    print(df['age_bin'].value_counts().sort_index())
    
    # Reverse-scoring subtests (Go/no-go ID 32; Trail Making A/B IDs 39, 40)
    reverse_subtests = [32, 39, 40]
    
    for subtest_id in reverse_subtests:
        col_name = f'subtest_{subtest_id}_score'
        if col_name in df.columns:
            print(f"\nReverse scoring {col_name}...")
            
            for age_bin in df['age_bin'].cat.categories:
                mask = df['age_bin'] == age_bin
                age_data = df.loc[mask, col_name].dropna()
                
                if len(age_data) > 0:
                    max_score = age_data.max()
                    print(f"  Age bin {age_bin}: max score = {max_score}")
                    
                    # Reverse score: max + 1 - score
                    df.loc[mask, col_name] = max_score + 1 - df.loc[mask, col_name]
    
    print("\nFirst 5 rows after reverse scoring:")
    with pd.option_context('display.max_columns', None):
        print(df.head(5))
    return df

def apply_outlier_exclusions(df):
    """
    Apply statistical outlier exclusions within age bins.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    tuple: (cleaned_df, outlier_exclusions)
    """
    print("\n=== OUTLIER EXCLUSIONS ===")
    
    subtest_cols = [f'subtest_{st}_score' for st in [36, 39, 40, 29, 28, 33, 30, 27, 32, 38, 37]]
    
    outlier_flags = pd.DataFrame(index=df.index.copy(), columns=subtest_cols, data=False)
    
    for age_bin in df['age_bin'].cat.categories:
        print(f"\nProcessing age bin: {age_bin}")
        age_mask = df['age_bin'] == age_bin
        age_data = df[age_mask]
        
        if len(age_data) == 0:
            continue
            
        for col in subtest_cols:
            if col in df.columns:
                scores = age_data[col].dropna()
                if len(scores) > 0:
                    mean_score = scores.mean()
                    std_score = scores.std()
                    
                    if std_score > 0:
                        # Identify outliers (>3 SD from mean)
                        outliers = np.abs(scores - mean_score) > 3 * std_score
                        outlier_flags.loc[scores.index, col] = outliers
                        
                        print(f"  {col}: {outliers.sum()} outliers detected")
    
    # Count outlier flags per participant
    outlier_counts = outlier_flags.sum(axis=1)
    
    # Exclude participants with ≥2 outlier flags
    before_count = len(df)
    df = df[outlier_counts < 2]
    excluded_outliers = before_count - len(df)
    
    print(f"\nExcluded {excluded_outliers} participants flagged as outliers on ≥2 subtests")
    print("\nFirst 5 rows after outlier exclusion:")
    with pd.option_context('display.max_columns', None):
        print(df.head(5))
    
    return df, excluded_outliers

def create_final_dataset(df):
    """
    Create the final cleaned dataset with required columns.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Final cleaned dataset
    """
    print("\n=== CREATING FINAL DATASET ===")
    
    # Required output columns
    required_cols = [
        'user_id', 'test_run_id', 'age', 'gender', 'education_level', 'country',
        'battery_id', 'time_of_day', 'grand_index',
        'subtest_36_score', 'subtest_39_score', 'subtest_40_score',
        'subtest_29_score', 'subtest_28_score', 'subtest_33_score',
        'subtest_30_score', 'subtest_27_score', 'subtest_32_score',
        'subtest_38_score', 'subtest_37_score', 'age_bin'
    ]
    
    # Select only required columns
    final_df = df[required_cols].copy()
    
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Final dataset columns: {list(final_df.columns)}")
    print("First 5 rows of final dataset:")
    with pd.option_context('display.max_columns', None):
        print(final_df.head(5))
    
    return final_df

def save_outputs(final_df, exclusion_log):
    """
    Save the final dataset and exclusion log to CSV files.
    
    Parameters:
    final_df (pd.DataFrame): Final cleaned dataset
    exclusion_log (list): List of exclusion criteria and counts
    """
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60)
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        print("Created outputs directory")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save final dataset
    final_filename = f"outputs/step1_cleaned_battery26_data_{timestamp}.csv"
    final_df.to_csv(final_filename, index=False)
    print(f"Saved final dataset to: {final_filename}")
    
    # Save exclusion log
    exclusion_df = pd.DataFrame(exclusion_log, columns=['exclusion_criteria', 'total_excluded'])
    exclusion_filename = f"outputs/step1_exclusion_log_{timestamp}.csv"
    exclusion_df.to_csv(exclusion_filename, index=False)
    print(f"Saved exclusion log to: {exclusion_filename}")
    
    print("\nExclusion Summary:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(exclusion_df)
    
    print(f"\nFinal participant count: {len(final_df)}")

def print_final_summary(initial_count, exclusion_log, final_count, final_df):
    """
    Print a comprehensive summary of the data cleaning process.
    """
    print("\n" + "="*60)
    print("FINAL SUMMARY STATISTICS")
    print("="*60)
    print(f"Initial participant count: {initial_count}")
    print("\nExclusion breakdown:")
    exclusion_df = pd.DataFrame(exclusion_log, columns=['exclusion_criteria', 'total_excluded'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(exclusion_df)
    print(f"\nFinal participant count: {final_count}")
    print("\nFinal dataset info:")
    print(final_df.info())
    print("\nFirst 5 rows of final dataset:")
    with pd.option_context('display.max_columns', None):
        print(final_df.head(5))
    print("\nDescriptive statistics (age, education_level):")
    print(final_df[['age', 'education_level']].describe())
    print("\nGender distribution:")
    print(final_df['gender'].value_counts(dropna=False))
    print("\nCountry distribution (top 10):")
    print(final_df['country'].value_counts().head(10))
    print("\nAge bin distribution:")
    print(final_df['age_bin'].value_counts().sort_index())
    print("="*60)

def main():
    """
    Main function to execute the data processing pipeline.
    
    Returns:
    int: 0 for success, 1 for error
    """
    try:
        print("\n" + "="*60)
        print("Starting Battery 26 data processing pipeline...")
        print("="*60)
        
        # Load and validate data
        input_file = "raw_data/battery26_df.csv"
        df = load_and_validate_data(input_file)
        
        # Verify subtests
        verify_subtests(df)
        
        # Reshape to wide format
        wide_df = reshape_to_wide_format(df)
        
        # Apply exclusion criteria
        cleaned_df, exclusion_log = apply_exclusion_criteria(wide_df)
        
        # Apply reverse scoring
        reversed_df = reverse_score_subtests(cleaned_df)
        
        # Apply outlier exclusions
        final_df, outlier_exclusions = apply_outlier_exclusions(reversed_df)
        exclusion_log.append(('Statistical outliers ≥2 subtests', outlier_exclusions))
        
        # Create final dataset
        final_dataset = create_final_dataset(final_df)
        
        # Save outputs
        save_outputs(final_dataset, exclusion_log)
        
        # Print final summary statistics
        initial_count = len(wide_df)
        final_count = len(final_dataset)
        print_final_summary(initial_count, exclusion_log, final_count, final_dataset)
        
        print("\nFinished execution")
        return 0
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
