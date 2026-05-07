#!/usr/bin/env python3
"""
Sensitivity Analysis Script for Heterogeneity Metrics Study
Performs multiple sensitivity analyses to assess robustness of primary findings.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import scipy.stats as stats
from scipy.stats import pearsonr
import warnings
import re
warnings.filterwarnings('ignore')

def find_most_recent_file(pattern):
    """
    Find the most recent file matching a pattern.
    
    Parameters:
    pattern (str): File pattern to match
    
    Returns:
    str: Path to most recent file or None if no files found
    """
    files = glob.glob(pattern)
    if not files:
        return None
    # Use lambda to avoid comparison issues with os.path.getctime
    return max(files, key=lambda x: os.path.getctime(x))

def load_and_validate_data():
    """
    Load and validate input datasets.
    
    Returns:
    tuple: (heterogeneity_df, battery_df) or (None, None) if loading fails
    """
    print("Loading input datasets...")
    
    # Find most recent heterogeneity metrics file
    het_pattern = "outputs/step3_heterogeneity_metrics_*.csv"
    het_file = find_most_recent_file(het_pattern)
    
    if not het_file:
        print(f"ERROR: No heterogeneity metrics file found matching pattern: {het_pattern}")
        return None, None
    
    print(f"Loading heterogeneity metrics from: {het_file}")
    
    # Load battery data
    battery_file = "raw_data/battery26_df.csv"
    if not os.path.exists(battery_file):
        print(f"ERROR: Battery data file not found: {battery_file}")
        return None, None
    
    print(f"Loading battery data from: {battery_file}")
    
    try:
        # Load heterogeneity metrics
        het_df = pd.read_csv(het_file)
        print(f"Heterogeneity metrics shape: {het_df.shape}")
        print("Heterogeneity metrics columns:", list(het_df.columns))
        print("First 3 rows of heterogeneity metrics:")
        print(het_df.head(3))
        
        # Load battery data
        battery_df = pd.read_csv(battery_file)
        print(f"Battery data shape: {battery_df.shape}")
        print("Battery data columns:", list(battery_df.columns))
        print("First 3 rows of battery data:")
        print(battery_df.head(3))
        
        # Validate required columns for heterogeneity data
        het_required_cols = [
            'user_id', 'age', 'gender', 'education_level', 'country',
            'test_run_id', 'battery_id', 'time_of_day', 'grand_index',
            'subtest_36_score', 'subtest_39_score', 'subtest_40_score',
            'subtest_29_score', 'subtest_28_score', 'subtest_33_score',
            'subtest_30_score', 'subtest_27_score', 'subtest_32_score',
            'subtest_38_score', 'subtest_37_score', 'age_bin',
            'percentile_36', 'percentile_39', 'percentile_40',
            'percentile_29', 'percentile_28', 'percentile_33',
            'percentile_30', 'percentile_27', 'percentile_32',
            'percentile_38', 'percentile_37', 'percentile_range', 'percentile_iqr'
        ]
        
        missing_het_cols = [col for col in het_required_cols if col not in het_df.columns]
        if missing_het_cols:
            print(f"ERROR: Missing required columns in heterogeneity data: {missing_het_cols}")
            return None, None
        
        # Validate required columns for battery data
        battery_required_cols = [
            'user_id', 'age', 'gender', 'education_level', 'country',
            'test_run_id', 'battery_id', 'specific_subtest_id',
            'time_of_day', 'raw_score', 'grand_index'
        ]
        
        missing_battery_cols = [col for col in battery_required_cols if col not in battery_df.columns]
        if missing_battery_cols:
            print(f"ERROR: Missing required columns in battery data: {missing_battery_cols}")
            return None, None
        
        # Print unique values for key experimental design parameters
        print("\nUnique values for key experimental design parameters:")
        print(f"Education levels in heterogeneity data: {sorted([str(x) for x in het_df['education_level'].unique()])}")
        print(f"Gender values in heterogeneity data: {sorted([str(x) for x in het_df['gender'].unique()])}")
        print(f"Age bins in heterogeneity data: {sorted([str(x) for x in het_df['age_bin'].unique()])}")
        print(f"Countries in heterogeneity data: {sorted([str(x) for x in het_df['country'].unique()])}")
        print(f"Time of day values in heterogeneity data: {sorted([str(x) for x in het_df['time_of_day'].unique()])}")
        
        print(f"Education levels in battery data: {sorted([str(x) for x in battery_df['education_level'].unique()])}")
        print(f"Gender values in battery data: {sorted([str(x) for x in battery_df['gender'].unique()])}")
        print(f"Specific subtest IDs in battery data: {sorted([str(x) for x in battery_df['specific_subtest_id'].unique()])}")
        print(f"Countries in battery data: {sorted([str(x) for x in battery_df['country'].unique()])}")
        print(f"Time of day values in battery data: {sorted([str(x) for x in battery_df['time_of_day'].unique()])}")
        
        # Remove rows with null values in key columns
        print("\nFiltering out null values...")
        het_initial_count = len(het_df)
        het_df = het_df.dropna(subset=het_required_cols)
        het_final_count = len(het_df)
        print(f"Heterogeneity data: {het_initial_count} -> {het_final_count} rows after removing nulls")
        
        battery_initial_count = len(battery_df)
        battery_df = battery_df.dropna(subset=battery_required_cols)
        # Filter out education_level == 99.0 and NaN
        battery_df = battery_df[battery_df['education_level'].notna()]
        battery_df = battery_df[battery_df['education_level'] != 99.0]
        battery_final_count = len(battery_df)
        print(f"Battery data: {battery_initial_count} -> {battery_final_count} rows after removing nulls and invalid education_level")
        
        return het_df, battery_df
        
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None, None

def sensitivity_analysis_a(het_df):
    """
    Sensitivity Analysis A: Coefficient of Variation metric.
    
    Parameters:
    het_df (pd.DataFrame): Heterogeneity metrics dataframe
    
    Returns:
    pd.DataFrame: Data with coefficient of variation metric
    """
    print("\n=== SENSITIVITY ANALYSIS A: Coefficient of Variation ===")
    
    # Calculate coefficient of variation for each participant
    percentile_cols = [
        'percentile_36', 'percentile_39', 'percentile_40',
        'percentile_29', 'percentile_28', 'percentile_33',
        'percentile_30', 'percentile_27', 'percentile_32',
        'percentile_38', 'percentile_37'
    ]
    
    print(f"Calculating coefficient of variation using columns: {percentile_cols}")
    
    # Calculate coefficient of variation (std/mean) for each row
    percentile_values = het_df[percentile_cols].values
    means = np.mean(percentile_values, axis=1)
    stds = np.std(percentile_values, axis=1, ddof=1)
    
    # Handle division by zero
    coefficient_of_variation = np.where(means != 0, stds / means, 0)
    
    # Create output dataframe
    output_df = het_df[[
        'user_id', 'age', 'gender', 'education_level', 'country',
        'test_run_id', 'battery_id', 'time_of_day', 'grand_index',
        'age_bin', 'percentile_range', 'percentile_iqr'
    ]].copy()
    
    output_df['coefficient_of_variation'] = coefficient_of_variation
    
    print(f"Coefficient of variation statistics:")
    print(f"Mean: {np.mean(coefficient_of_variation):.4f}")
    print(f"Std: {np.std(coefficient_of_variation):.4f}")
    print(f"Min: {np.min(coefficient_of_variation):.4f}")
    print(f"Max: {np.max(coefficient_of_variation):.4f}")
    
    print("First 2 rows of coefficient of variation analysis:")
    print(output_df.head(2))
    
    return output_df

def sensitivity_analysis_b(het_df):
    """
    Sensitivity Analysis B: Collapsed 3-level education variable.
    
    Parameters:
    het_df (pd.DataFrame): Heterogeneity metrics dataframe
    
    Returns:
    pd.DataFrame: Data with collapsed education variable
    """
    print("\n=== SENSITIVITY ANALYSIS B: Collapsed Education Variable ===")
    
    # Create collapsed education variable
    def collapse_education(edu_level):
        if edu_level in [1, 2]:
            return 'Low'
        elif edu_level in [3, 4, 8]:
            return 'Medium'
        elif edu_level in [5, 6, 7]:
            return 'High'
        else:
            return 'Unknown'
    
    output_df = het_df[[
        'user_id', 'age', 'gender', 'education_level', 'country',
        'test_run_id', 'battery_id', 'time_of_day', 'grand_index',
        'age_bin', 'percentile_range', 'percentile_iqr'
    ]].copy()
    
    output_df['education_collapsed'] = output_df['education_level'].apply(collapse_education)
    
    print("Education level mapping:")
    print(output_df.groupby(['education_level', 'education_collapsed']).size().reset_index(name='count'))
    
    print("Distribution of collapsed education levels:")
    print(output_df['education_collapsed'].value_counts())
    
    print("First 2 rows of collapsed education analysis:")
    print(output_df.head(2))
    
    return output_df

def sensitivity_analysis_c(battery_df):
    """
    Sensitivity Analysis C: Alternative outlier cutoff (2.5 SD).
    
    Parameters:
    battery_df (pd.DataFrame): Raw battery data
    
    Returns:
    pd.DataFrame: Data processed with 2.5 SD outlier cutoff
    """
    print("\n=== SENSITIVITY ANALYSIS C: Alternative Outlier Cutoff (2.5 SD) ===")
    
    # Filter to subtests of interest
    subtests_of_interest = [36, 39, 40, 29, 28, 33, 30, 27, 32, 38, 37]
    battery_filtered = battery_df[battery_df['specific_subtest_id'].isin(subtests_of_interest)].copy()

    # Ensure proper data types
    battery_filtered['specific_subtest_id'] = battery_filtered['specific_subtest_id'].astype(int)
    battery_filtered['education_level'] = battery_filtered['education_level'].astype(int)

    print(f"Filtered to {len(battery_filtered)} rows with subtests of interest")
    
    # Apply 2.5 SD outlier removal based on age-bin means per subtest, and remove participants with outlying scores on 2+ subtests
    print("Applying 2.5 SD outlier removal per subtest and age bin, removing participants with outlying scores on 2+ subtests...")
    initial_count = len(battery_filtered)

    # Create age bins matching original analysis
    battery_filtered['age_bin'] = pd.cut(
        battery_filtered['age'],
        bins=[17, 29, 39, 49, 59, 69, 100],
        labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70-99']
    )

    # Identify outliers per subtest and age bin
    def flag_outliers(group):
        mean_score = group['raw_score'].mean()
        std_score = group['raw_score'].std()
        lower_bound = mean_score - 2.5 * std_score
        upper_bound = mean_score + 2.5 * std_score
        group = group.copy()
        group['is_outlier'] = ~group['raw_score'].between(lower_bound, upper_bound)
        return group

    battery_flagged = battery_filtered.groupby(['specific_subtest_id', 'age_bin'], group_keys=False).apply(flag_outliers)

    # Count number of subtests where each participant is an outlier
    outlier_counts = battery_flagged.groupby(['user_id', 'test_run_id', 'battery_id'])['is_outlier'].sum().reset_index()
    outlier_counts = outlier_counts.rename(columns={'is_outlier': 'n_outlier_subtests'})

    # Merge back to battery_flagged
    battery_flagged = battery_flagged.merge(outlier_counts, on=['user_id', 'test_run_id', 'battery_id'], how='left')

    # Remove participants with outlying scores on 2 or more subtests
    valid_participants = outlier_counts[outlier_counts['n_outlier_subtests'] < 2][['user_id', 'test_run_id', 'battery_id']]
    battery_clean = battery_flagged.merge(valid_participants, on=['user_id', 'test_run_id', 'battery_id'], how='inner')

    # Remove the outlier columns
    battery_clean = battery_clean.drop(columns=['is_outlier', 'n_outlier_subtests'])

    final_count = len(battery_clean)
    print(f"Outlier removal: {initial_count} -> {final_count} rows ({initial_count - final_count} outliers removed)")

    # Calculate age-stratified percentiles
    print("Calculating age-stratified percentiles...")

    # Calculate percentiles within age bins using percentileofscore
    def calculate_percentiles(group):
        # Use percentileofscore for each value in the group
        group = group.copy()
        scores = group['raw_score'].values
        percentiles = [stats.percentileofscore(scores, x, kind='rank') for x in scores]
        group['percentile_rank'] = percentiles
        return group

    battery_clean = battery_clean.groupby(['specific_subtest_id', 'age_bin']).apply(calculate_percentiles).reset_index(drop=True)
    
    # Pivot to get one row per participant
    print("Pivoting data to participant level...")

    # Step 1: Create unique participant info dataframe
    participant_info_cols = ['user_id', 'age', 'gender', 'education_level', 'country', 
                            'test_run_id', 'battery_id', 'time_of_day', 'grand_index', 'age_bin']
    participant_info_df = battery_clean[participant_info_cols].drop_duplicates(
        subset=['user_id', 'test_run_id', 'battery_id']
    ).copy()
    print(f"Created unique participant info dataframe with {len(participant_info_df)} participants")

    # Step 2: Pivot with minimal index
    percentile_pivot = battery_clean.pivot_table(
        index=['user_id', 'test_run_id', 'battery_id'],
        columns='specific_subtest_id',
        values='percentile_rank',
        aggfunc='mean'
    ).reset_index()

    # Step 3: Merge back participant info
    percentile_pivot = pd.merge(participant_info_df, percentile_pivot, 
                               on=['user_id', 'test_run_id', 'battery_id'], how='left')
    print(f"Merged participant info, final shape: {percentile_pivot.shape}")

    # Rename columns - pivot creates integer column names, we need to rename them
    percentile_cols = {}
    for subtest in subtests_of_interest:
        if subtest in percentile_pivot.columns:
            percentile_cols[subtest] = f'percentile_{subtest}'

    if percentile_cols:
        percentile_pivot = percentile_pivot.rename(columns=percentile_cols)
        print(f"Renamed {len(percentile_cols)} percentile columns")

    # Calculate heterogeneity metrics
    print("Calculating alternative heterogeneity metrics...")

    percentile_score_cols = [f'percentile_{subtest}' for subtest in subtests_of_interest 
                            if f'percentile_{subtest}' in percentile_pivot.columns]

    if len(percentile_score_cols) == 0:
        print("ERROR: No valid percentile columns found after pivot operation")
        return pd.DataFrame()

    print(f"Found {len(percentile_score_cols)} valid percentile columns: {percentile_score_cols}")

    percentile_values = percentile_pivot[percentile_score_cols].values

    # Calculate range and IQR
    percentile_range_alt = np.max(percentile_values, axis=1) - np.min(percentile_values, axis=1)
    percentile_iqr_alt = np.percentile(percentile_values, 75, axis=1) - np.percentile(percentile_values, 25, axis=1)

    percentile_pivot['percentile_range_alt'] = percentile_range_alt
    percentile_pivot['percentile_iqr_alt'] = percentile_iqr_alt

    print(f"Alternative range statistics:")
    print(f"Mean: {np.mean(percentile_range_alt):.4f}")
    print(f"Std: {np.std(percentile_range_alt):.4f}")

    print(f"Alternative IQR statistics:")
    print(f"Mean: {np.mean(percentile_iqr_alt):.4f}")
    print(f"Std: {np.std(percentile_iqr_alt):.4f}")

    # Select output columns
    output_cols = [
        'user_id', 'age', 'gender', 'education_level', 'country',
        'test_run_id', 'battery_id', 'time_of_day', 'grand_index',
        'age_bin', 'percentile_range_alt', 'percentile_iqr_alt'
    ]

    output_df = percentile_pivot[output_cols].copy()

    print("First 2 rows of alternative outlier analysis:")
    print(output_df.head(2))

    return output_df

def sensitivity_analysis_d(het_df):
    """
    Sensitivity Analysis D: Split-half reliability analysis.
    
    Parameters:
    het_df (pd.DataFrame): Heterogeneity metrics dataframe
    
    Returns:
    pd.DataFrame: Data with split-half reliability metrics
    """
    print("\n=== SENSITIVITY ANALYSIS D: Split-Half Reliability ===")
    
    # Define odd and even subtest groups
    odd_subtests = [36, 40, 28, 30, 32, 37]  # 6 subtests
    even_subtests = [39, 29, 33, 27, 38]     # 5 subtests
    
    print(f"Odd subtests: {odd_subtests}")
    print(f"Even subtests: {even_subtests}")
    
    # Get percentile columns for each half
    odd_percentile_cols = [f'percentile_{subtest}' for subtest in odd_subtests]
    even_percentile_cols = [f'percentile_{subtest}' for subtest in even_subtests]
    
    print(f"Odd percentile columns: {odd_percentile_cols}")
    print(f"Even percentile columns: {even_percentile_cols}")
    
    # Calculate heterogeneity metrics for each half
    odd_percentiles = het_df[odd_percentile_cols].values
    even_percentiles = het_df[even_percentile_cols].values
    
    # Calculate range and IQR for each half
    odd_half_range = np.max(odd_percentiles, axis=1) - np.min(odd_percentiles, axis=1)
    odd_half_iqr = np.percentile(odd_percentiles, 75, axis=1) - np.percentile(odd_percentiles, 25, axis=1)
    
    even_half_range = np.max(even_percentiles, axis=1) - np.min(even_percentiles, axis=1)
    even_half_iqr = np.percentile(even_percentiles, 75, axis=1) - np.percentile(even_percentiles, 25, axis=1)
    
    # Calculate correlations
    correlation_range, p_value_range = pearsonr(odd_half_range, even_half_range)
    correlation_iqr, p_value_iqr = pearsonr(odd_half_iqr, even_half_iqr)
    
    print(f"Split-half reliability for range: r = {correlation_range:.4f}, p = {p_value_range:.4f}")
    print(f"Split-half reliability for IQR: r = {correlation_iqr:.4f}, p = {p_value_iqr:.4f}")
    
    # Create output dataframe
    output_df = het_df[[
        'user_id', 'age', 'gender', 'education_level', 'country',
        'test_run_id', 'battery_id', 'time_of_day', 'grand_index', 'age_bin'
    ]].copy()
    
    output_df['odd_half_range'] = odd_half_range
    output_df['even_half_range'] = even_half_range
    output_df['odd_half_iqr'] = odd_half_iqr
    output_df['even_half_iqr'] = even_half_iqr
    output_df['odd_even_correlation_range'] = correlation_range
    output_df['odd_even_correlation_iqr'] = correlation_iqr
    
    print("First 2 rows of split-half reliability analysis:")
    print(output_df.head(2))
    
    return output_df

def run_regression_analysis(df, dependent_var, analysis_name):
    """
    Run regression analysis with comprehensive diagnostics.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    dependent_var (str): Name of dependent variable
    analysis_name (str): Name of analysis for labeling
    
    Returns:
    pd.DataFrame: Regression results
    """
    print(f"\n--- Running regression analysis for {analysis_name} ---")
    
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # Prepare data for regression
        reg_df = df.copy()
        
        # Remove any remaining null values
        reg_df = reg_df.dropna(subset=[dependent_var, 'education_level', 'age', 'gender', 'country', 'time_of_day'])
        
        print(f"Regression data shape: {reg_df.shape}")
        print(f"Dependent variable '{dependent_var}' statistics:")
        print(f"Mean: {reg_df[dependent_var].mean():.4f}")
        print(f"Std: {reg_df[dependent_var].std():.4f}")
        print(f"Min: {reg_df[dependent_var].min():.4f}")
        print(f"Max: {reg_df[dependent_var].max():.4f}")
        
        # Check for education_collapsed vs education_level
        if 'education_collapsed' in reg_df.columns:
            education_var = 'C(education_collapsed, Treatment("Low"))'
            print("Using collapsed education variable with 'Low' as reference")
        elif analysis_name == 'Coefficient_of_Variation':
            education_var = 'C(education_level, Treatment(1))'
            print("Using original education variable with level 1 as reference for CoV analysis")
        else:
            education_var = 'C(education_level, Treatment(4))'
            print("Using original education variable with level 4 as reference")
        
        # Define regression formula
        formula = f"{dependent_var} ~ {education_var} + age + C(gender) + C(country) + C(time_of_day)"
        print(f"Regression formula: {formula}")
        
        # Fit regression model
        model = smf.ols(formula, data=reg_df).fit()
        
        print("Regression summary:")
        print(model.summary())
        
        # Extract results
        results = []
        
        # Get parameter names and values
        params = model.params
        std_errors = model.bse
        t_stats = model.tvalues
        p_values = model.pvalues
        conf_int = model.conf_int()
        
        # Calculate standardized coefficients using design matrix
        X_design_matrix = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
        X_std = X_design_matrix.std()
        y_std = reg_df[dependent_var].std()

        standardized_betas = {}
        for param_name in params.index:
            if param_name == 'Intercept':
                standardized_betas[param_name] = 0
            else:
                standardized_betas[param_name] = params[param_name] * (X_std[param_name] / y_std)

        # Apply Bonferroni correction
        alpha_bonferroni = 0.025
        
        for param_name in params.index:
            if param_name != 'Intercept':
                results.append({
                    'analysis_type': analysis_name,
                    'model_name': f'{dependent_var}_regression',
                    'dependent_variable': dependent_var,
                    'predictor_variable': param_name,
                    'coefficient': params[param_name],
                    'std_error': std_errors[param_name],
                    'standardized_beta': standardized_betas[param_name],
                    't_statistic': t_stats[param_name],
                    'p_value': p_values[param_name],
                    'conf_int_lower': conf_int.iloc[list(params.index).index(param_name), 0],
                    'conf_int_upper': conf_int.iloc[list(params.index).index(param_name), 1],
                    'significant_bonferroni': p_values[param_name] < alpha_bonferroni
                })
        
        # Model diagnostics
        print("\n--- Model Diagnostics ---")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.4f}, p-value: {model.f_pvalue:.4f}")

        # VIF calculation for multicollinearity
        print("\n--- Multicollinearity Diagnostics (VIF) ---")
        try:
            # Exclude intercept column for VIF
            exog_names = model.model.exog_names
            if 'Intercept' in exog_names:
                exog_names = [name for name in exog_names if name != 'Intercept']
            exog_idx = [model.model.exog_names.index(name) for name in exog_names]
            exog_for_vif = model.model.exog[:, exog_idx]
            vif_data = []
            for i, name in enumerate(exog_names):
                vif = variance_inflation_factor(exog_for_vif, i)
                vif_data.append({'variable': name, 'VIF': vif})
            vif_df = pd.DataFrame(vif_data)
            print(vif_df)
        except Exception as e:
            print(f"Could not calculate VIF: {e}")

        # Normality test of residuals
        residuals = model.resid
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        print(f"Shapiro-Wilk normality test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")

        # Homoscedasticity test
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
            print(f"Breusch-Pagan heteroscedasticity test: LM = {bp_stat:.4f}, p = {bp_p:.4f}")
        except:
            print("Could not perform Breusch-Pagan test")

        return pd.DataFrame(results)

    except Exception as e:
        print(f"ERROR in regression analysis: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to execute all sensitivity analyses.
    
    Returns:
    int: 0 for success, 1 for failure
    """
    try:
        print("Starting sensitivity analyses...")
        
        # Create outputs directory
        os.makedirs('outputs', exist_ok=True)
        
        # Get timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load data
        het_df, battery_df = load_and_validate_data()
        if het_df is None or battery_df is None:
            print("ERROR: Failed to load required data")
            return 1
        
        # Store all regression results
        all_regression_results = []
        
        # Sensitivity Analysis A: Coefficient of Variation
        print("\n" + "="*60)
        coeff_var_df = sensitivity_analysis_a(het_df)
        if coeff_var_df is not None:
            output_file = f"outputs/step6_sensitivity_coefficient_variation_{timestamp}.csv"
            coeff_var_df.to_csv(output_file, index=False)
            print(f"Saved coefficient of variation analysis to: {output_file}")
            
            # Run regression analysis
            reg_results = run_regression_analysis(coeff_var_df, 'coefficient_of_variation', 'Coefficient_of_Variation')
            all_regression_results.append(reg_results)
        
        # Sensitivity Analysis B: Collapsed Education
        print("\n" + "="*60)
        collapsed_edu_df = sensitivity_analysis_b(het_df)
        if collapsed_edu_df is not None:
            output_file = f"outputs/step6_sensitivity_collapsed_education_{timestamp}.csv"
            collapsed_edu_df.to_csv(output_file, index=False)
            print(f"Saved collapsed education analysis to: {output_file}")
            
            # Run regression analyses for both heterogeneity metrics
            reg_results_range = run_regression_analysis(collapsed_edu_df, 'percentile_range', 'Collapsed_Education_Range')
            reg_results_iqr = run_regression_analysis(collapsed_edu_df, 'percentile_iqr', 'Collapsed_Education_IQR')
            all_regression_results.extend([reg_results_range, reg_results_iqr])
        
        # Sensitivity Analysis C: Alternative Outlier Cutoff
        print("\n" + "="*60)
        alt_outlier_df = sensitivity_analysis_c(battery_df)
        if alt_outlier_df is not None and len(alt_outlier_df) > 0:
            output_file = f"outputs/step6_sensitivity_alternative_outliers_{timestamp}.csv"
            alt_outlier_df.to_csv(output_file, index=False)
            print(f"Saved alternative outlier analysis to: {output_file}")
            
            # Run regression analyses for alternative metrics
            reg_results_range_alt = run_regression_analysis(alt_outlier_df, 'percentile_range_alt', 'Alternative_Outliers_Range')
            reg_results_iqr_alt = run_regression_analysis(alt_outlier_df, 'percentile_iqr_alt', 'Alternative_Outliers_IQR')
            all_regression_results.extend([reg_results_range_alt, reg_results_iqr_alt])
        
        # Sensitivity Analysis D: Split-Half Reliability
        print("\n" + "="*60)
        split_half_df = sensitivity_analysis_d(het_df)
        if split_half_df is not None:
            output_file = f"outputs/step6_split_half_reliability_{timestamp}.csv"
            split_half_df.to_csv(output_file, index=False)
            print(f"Saved split-half reliability analysis to: {output_file}")
            
            # Run regression analyses for split-half metrics
            reg_results_odd_range = run_regression_analysis(split_half_df, 'odd_half_range', 'Split_Half_Odd_Range')
            reg_results_even_range = run_regression_analysis(split_half_df, 'even_half_range', 'Split_Half_Even_Range')
            reg_results_odd_iqr = run_regression_analysis(split_half_df, 'odd_half_iqr', 'Split_Half_Odd_IQR')
            reg_results_even_iqr = run_regression_analysis(split_half_df, 'even_half_iqr', 'Split_Half_Even_IQR')
            all_regression_results.extend([reg_results_odd_range, reg_results_even_range, reg_results_odd_iqr, reg_results_even_iqr])
        
        # Combine all regression results
        print("\n" + "="*60)
        print("Combining all regression results...")
        
        combined_results = pd.concat([df for df in all_regression_results if not df.empty], ignore_index=True)
        
        if not combined_results.empty:
            output_file = f"outputs/step6_sensitivity_regression_results_{timestamp}.csv"
            combined_results.to_csv(output_file, index=False)
            print(f"Saved combined regression results to: {output_file}")
            
            print("Combined regression results columns:", list(combined_results.columns))
            print("First 2 rows of combined results:")
            print(combined_results.head(2))
        
        # Create robustness comparison (properly compare to primary results from Step 4)
        print("\n" + "="*60)
        print("Creating robustness comparison...")

        # Find most recent Step 4 regression results file
        step4_pattern = "outputs/step4_primary_regression_results_*.csv"
        step4_file = find_most_recent_file(step4_pattern)
        if not step4_file or not os.path.exists(step4_file):
            print(f"WARNING: Could not find primary regression results file for robustness comparison: {step4_pattern}")
            robustness_df = pd.DataFrame()
        else:
            print(f"Loading primary regression results from: {step4_file}")
            primary_results = pd.read_csv(step4_file)
            # Only keep education-related predictors
            primary_edu = primary_results[primary_results['predictor_variable'].str.contains('education', case=False, na=False)].copy()
            # Extract education level numbers from predictor variable strings using regex
            primary_edu['edu_level_num'] = primary_edu['predictor_variable'].str.extract(r'\[T\.(\d+\.?\d*)\]').astype(float)
            robustness_data = []
            for analysis in combined_results['analysis_type'].unique():
                analysis_data = combined_results[combined_results['analysis_type'] == analysis]
                edu_effects = analysis_data[analysis_data['predictor_variable'].str.contains('education', case=False, na=False)]
                for _, row in edu_effects.iterrows():
                    effect_size_primary = np.nan
                    p_value_primary = np.nan
                    conclusion_consistent = 'No_Match'
                    match = pd.DataFrame()
                    # Extract education level number from sensitivity predictor
                    if '[T.' in row['predictor_variable'] and ']' in row['predictor_variable']:
                        sensitivity_edu_match = re.search(r'\[T\.(\d+\.?\d*)\]', row['predictor_variable'])
                        if sensitivity_edu_match:
                            sensitivity_edu_level_num = float(sensitivity_edu_match.group(1))
                            # Find match based on education level number
                            match = primary_edu[primary_edu['edu_level_num'] == sensitivity_edu_level_num]
                    if match is not None and not match.empty:
                        match_row = match.iloc[0]
                        effect_size_primary = match_row['standardized_beta']
                        p_value_primary = match_row['p_value']
                        conclusion_consistent = (row['p_value'] < 0.05) == (p_value_primary < 0.05)
                    else:
                        # Handle cases where direct matching fails
                        if 'education_collapsed' in row['predictor_variable']:
                            # Collapsed education variables are not directly comparable to original levels
                            conclusion_consistent = 'Not_Directly_Comparable'
                        elif 'education_level' in row['predictor_variable']:
                            conclusion_consistent = 'No_Primary_Match'
                        else:
                            conclusion_consistent = 'No_Primary_Match'
                    robustness_data.append({
                        'analysis_type': analysis,
                        'education_level': row['predictor_variable'],
                        'effect_size_primary': effect_size_primary,
                        'effect_size_sensitivity': row['standardized_beta'],
                        'p_value_primary': p_value_primary,
                        'p_value_sensitivity': row['p_value'],
                        'conclusion_consistent': conclusion_consistent
                    })
            robustness_df = pd.DataFrame(robustness_data)

        if not robustness_df.empty:
            output_file = f"outputs/step6_robustness_comparison_{timestamp}.csv"
            robustness_df.to_csv(output_file, index=False)
            print(f"Saved robustness comparison to: {output_file}")

            print("Robustness comparison columns:", list(robustness_df.columns))
            print("First 2 rows of robustness comparison:")
            print(robustness_df.head(2))
        
        print("\n" + "="*60)
        print("SENSITIVITY ANALYSES COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("Finished execution")
        return 0
        
    except Exception as e:
        print(f"ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
