#!/usr/bin/env python3
"""
Step 5: Interaction Regression Analysis
Conducts multiple linear regression models with interaction terms to test whether
the relationship between education and heterogeneity differs across age groups.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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
    print(f"Found latest file: {latest_file}")
    return latest_file

def load_and_validate_data(filepath):
    """
    Load and validate the input dataset.
    
    Parameters:
    filepath (str): Path to the input CSV file
    
    Returns:
    pd.DataFrame: Validated dataset
    """
    print(f"Loading data from: {filepath}")
    
    # Required columns for this analysis
    required_columns = [
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
    
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data with shape: {df.shape}")
        
        # Print all columns
        print("All columns in dataset:")
        print(df.columns.tolist())
        
        # Print first 3 rows
        print("\nFirst 3 rows of data:")
        print(df.head(3))
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return None
        
        # Print unique values for key categorical variables
        print("\nUnique values for key categorical variables:")
        print(f"Gender: {df['gender'].unique()}")
        print(f"Education level: {df['education_level'].unique()}")
        print(f"Country: {df['country'].unique()}")
        print(f"Time of day: {df['time_of_day'].unique()}")
        print(f"Age bin: {df['age_bin'].unique()}")
        
        # Print data types for key variables
        print("\nData types for key variables:")
        print(f"Age: {df['age'].dtype}")
        print(f"Percentile range: {df['percentile_range'].dtype}")
        print(f"Percentile IQR: {df['percentile_iqr'].dtype}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_age_groups(df):
    """
    Create three-level categorical age_group variable.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with age column
    
    Returns:
    pd.DataFrame: Dataframe with added age_group column
    """
    print("Creating age groups...")
    
    # Create age_group based on age
    def categorize_age(age):
        if pd.isna(age):
            return np.nan
        elif age >= 18 and age <= 39:
            return 'Younger'
        elif age >= 40 and age <= 49:
            return 'Middle'
        elif age >= 50:
            return 'Older'
        else:
            return np.nan
    
    df['age_group'] = df['age'].apply(categorize_age)
    
    print(f"Age group distribution:")
    print(df['age_group'].value_counts())
    
    return df

def create_education_groups(df):
    """
    Create three-level categorical education_group variable.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with education_level column
    
    Returns:
    pd.DataFrame: Dataframe with added education_group column
    """
    print("Creating education groups...")
    
    def categorize_education(education_level):
        if pd.isna(education_level):
            return np.nan
        elif education_level in [1, 2]:  # Some high school, High school diploma/GED
            return 'Low'
        elif education_level in [3, 4, 8]:  # Some college, College degree, Associate's degree
            return 'Medium'
        elif education_level in [5, 6, 7]:  # Professional degree, Master's degree, Ph.D.
            return 'High'
        else:
            return np.nan
    
    df['education_group'] = df['education_level'].apply(categorize_education)
    
    print(f"Education group distribution:")
    print(df['education_group'].value_counts())
    
    return df

def clean_data_for_regression(df):
    """
    Clean and prepare data for regression analysis.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Cleaned dataframe ready for regression
    """
    print("Cleaning data for regression analysis...")
    
    # Initial data size
    initial_size = len(df)
    print(f"Initial dataset size: {initial_size}")
    
    # Remove rows with missing values in key variables
    key_vars = ['age', 'gender', 'education_level', 'country', 'time_of_day',
                'percentile_range', 'percentile_iqr', 'age_group', 'education_group']
    
    print("Checking for missing values in key variables:")
    for var in key_vars:
        missing_count = df[var].isna().sum()
        print(f"{var}: {missing_count} missing values")
    
    # Filter out rows with missing values
    df_clean = df.dropna(subset=key_vars)
    
    # Bin time_of_day into meaningful categories
    bins = [-1, 4, 11, 17, 23]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df_clean['time_of_day_binned'] = pd.cut(df_clean['time_of_day'], bins=bins, labels=labels, right=True)
    df_clean['time_of_day_binned'] = df_clean['time_of_day_binned'].astype('category')
    
    # Update key_vars to include binned time_of_day and check for missing values
    key_vars_binned = ['age', 'gender', 'education_level', 'country', 'time_of_day_binned',
                       'percentile_range', 'percentile_iqr', 'age_group', 'education_group']
    print("Checking for missing values in key variables (including binned time_of_day):")
    for var in key_vars_binned:
        missing_count = df_clean[var].isna().sum()
        print(f"{var}: {missing_count} missing values")
    df_clean = df_clean.dropna(subset=key_vars_binned)
    
    final_size = len(df_clean)
    excluded_count = initial_size - final_size
    print(f"Excluded {excluded_count} rows due to missing values")
    print(f"Final dataset size: {final_size}")
    
    # Convert categorical variables to proper format
    df_clean['gender'] = df_clean['gender'].astype('category')
    df_clean['country'] = df_clean['country'].astype('category')
    df_clean['age_group'] = df_clean['age_group'].astype('category')
    df_clean['education_level'] = df_clean['education_level'].astype('category')
    df_clean['education_group'] = df_clean['education_group'].astype('category')
    df_clean['time_of_day_binned'] = df_clean['time_of_day_binned'].astype('category')
    
    print("Data cleaning completed successfully")
    return df_clean

def run_interaction_regression(df, dependent_var, model_name):
    """
    Run multiple linear regression with interaction terms.
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    dependent_var (str): Name of dependent variable
    model_name (str): Name of the model for output
    
    Returns:
    tuple: (fitted_model, results_df)
    """
    print(f"Running {model_name} regression model...")
    
    # Define the formula with interaction terms
    formula = f"{dependent_var} ~ C(education_group, Treatment('Low')) * C(age_group, Treatment('Younger')) + C(gender) + C(country) + C(time_of_day_binned)"
    
    print(f"Model formula: {formula}")
    
    try:
        # Fit the model
        model = smf.ols(formula, data=df).fit()
        
        print(f"Model fitted successfully")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        
        # Calculate standardized betas post-hoc
        sd_y = df[dependent_var].std()
        sd_x = pd.Series(np.std(model.model.exog, axis=0), index=model.model.exog_names)
        standardized_betas = model.params * sd_x / sd_y
        
        # Extract results
        results_list = []
        
        # Get coefficient summary
        coef_summary = model.summary2().tables[1]
        
        # Apply Bonferroni correction (alpha = 0.025 for two models)
        bonferroni_alpha = 0.025
        
        for idx, row in coef_summary.iterrows():
            # Check if this is an interaction term
            is_interaction = ':' in str(idx) and 'education_group' in str(idx) and 'age_group' in str(idx)
            
            # Calculate standardized beta
            if str(idx) == 'Intercept':
                standardized_beta = np.nan
            else:
                standardized_beta = standardized_betas.get(str(idx), np.nan)
            
            result_row = {
                'model_name': model_name,
                'dependent_variable': dependent_var,
                'predictor_variable': str(idx),
                'coefficient': row['Coef.'],
                'std_error': row['Std.Err.'],
                'standardized_beta': standardized_beta,
                't_statistic': row['t'],
                'p_value': row['P>|t|'],
                'conf_int_lower': row['[0.025'],
                'conf_int_upper': row['0.975]'],
                'significant_bonferroni': row['P>|t|'] < bonferroni_alpha,
                'interaction_term': is_interaction
            }
            results_list.append(result_row)
        
        results_df = pd.DataFrame(results_list)
        
        print(f"Extracted {len(results_df)} coefficients")
        print(f"Interaction terms found: {results_df['interaction_term'].sum()}")
        
        return model, results_df
        
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None, None

def test_assumptions(model, model_name, df, dependent_var):
    """
    Test regression assumptions, including a two-step VIF procedure to distinguish
    between data-level and structural multicollinearity.
    
    Parameters:
    model: Fitted statsmodels regression model
    model_name (str): Name of the model
    df (pd.DataFrame): Dataframe used for modeling
    dependent_var (str): Name of dependent variable
    
    Returns:
    pd.DataFrame: Results of assumption tests
    """
    print(f"Testing assumptions for {model_name}...")
    
    assumption_results = []
    
    try:
        # Get residuals and fitted values
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # 1. Normality test (Shapiro-Wilk)
        print("Testing normality of residuals...")
        if len(residuals) > 5000:
            # Sample for large datasets
            sample_residuals = np.random.choice(residuals, 5000, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
            print(f"Shapiro-Wilk test (sampled): statistic={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        else:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        
        assumption_results.append({
            'model_name': model_name,
            'test_name': 'Shapiro-Wilk',
            'test_statistic': shapiro_stat,
            'p_value': shapiro_p,
            'assumption_met': shapiro_p > 0.05
        })
        
        # 2. Homoscedasticity test (Breusch-Pagan)
        print("Testing homoscedasticity...")
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
        print(f"Breusch-Pagan test: statistic={bp_stat:.4f}, p={bp_p:.4f}")
        
        assumption_results.append({
            'model_name': model_name,
            'test_name': 'Breusch-Pagan',
            'test_statistic': bp_stat,
            'p_value': bp_p,
            'assumption_met': bp_p > 0.05
        })
        
        # 3. Independence test (Durbin-Watson)
        print("Testing independence of residuals...")
        dw_stat = durbin_watson(residuals)
        print(f"Durbin-Watson test: statistic={dw_stat:.4f}")
        
        # DW statistic around 2 indicates no autocorrelation
        dw_assumption_met = abs(dw_stat - 2) < 0.5
        
        assumption_results.append({
            'model_name': model_name,
            'test_name': 'Durbin-Watson',
            'test_statistic': dw_stat,
            'p_value': np.nan,  # DW doesn't have a p-value
            'assumption_met': dw_assumption_met
        })
        
        # 4. Multicollinearity test (VIF) - Two-step approach
        print("Testing multicollinearity (two-step VIF)...")
        try:
            # Step 1: Main effects only (formal assumption test)
            main_effects_formula = (
                f"{dependent_var} ~ C(education_group, Treatment('Low')) + C(age_group, Treatment('Younger')) + "
                f"C(gender) + C(country) + C(time_of_day_binned)"
            )
            main_effects_model = smf.ols(main_effects_formula, data=df).fit()
            exog_main = main_effects_model.model.exog
            exog_names_main = main_effects_model.model.exog_names
            vif_data_main = pd.DataFrame()
            vif_data_main["Variable"] = [exog_names_main[i] for i in range(1, len(exog_names_main))]
            vif_data_main["VIF"] = [variance_inflation_factor(exog_main, i) for i in range(1, len(exog_names_main))]
            max_vif_main = vif_data_main["VIF"].max()
            print(f"Maximum VIF (Main Effects): {max_vif_main:.4f}")
            assumption_results.append({
                'model_name': model_name,
                'test_name': 'VIF (Main Effects)',
                'test_statistic': max_vif_main,
                'p_value': np.nan,
                'assumption_met': max_vif_main < 10  # Common threshold
            })
        except Exception as e:
            print(f"Error calculating VIF (Main Effects): {e}")
            assumption_results.append({
                'model_name': model_name,
                'test_name': 'VIF (Main Effects)',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'assumption_met': np.nan
            })
        try:
            # Step 2: Full model (diagnostic only)
            exog_full = model.model.exog
            exog_names_full = model.model.exog_names
            vif_data_full = pd.DataFrame()
            vif_data_full["Variable"] = [exog_names_full[i] for i in range(1, len(exog_names_full))]
            vif_data_full["VIF"] = [variance_inflation_factor(exog_full, i) for i in range(1, len(exog_names_full))]
            max_vif_full = vif_data_full["VIF"].max()
            print(f"Maximum VIF (Full Model): {max_vif_full:.4f}")
            assumption_results.append({
                'model_name': model_name,
                'test_name': 'VIF (Full Model)',
                'test_statistic': max_vif_full,
                'p_value': np.nan,
                'assumption_met': np.nan  # Diagnostic only
            })
        except Exception as e:
            print(f"Error calculating VIF (Full Model): {e}")
            assumption_results.append({
                'model_name': model_name,
                'test_name': 'VIF (Full Model)',
                'test_statistic': np.nan,
                'p_value': np.nan,
                'assumption_met': np.nan
            })
        
        print(f"Assumption testing completed for {model_name}")
        
    except Exception as e:
        print(f"Error in assumption testing: {e}")
    
    return pd.DataFrame(assumption_results)

def extract_model_fit_stats(model, model_name, dependent_var):
    """
    Extract model fit statistics.
    
    Parameters:
    model: Fitted statsmodels regression model
    model_name (str): Name of the model
    dependent_var (str): Name of dependent variable
    
    Returns:
    dict: Model fit statistics
    """
    print(f"Extracting model fit statistics for {model_name}...")
    
    try:
        fit_stats = {
            'model_name': model_name,
            'dependent_variable': dependent_var,
            'r_squared': model.rsquared,
            'adjusted_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_p_value': model.f_pvalue,
            'aic': model.aic,
            'bic': model.bic,
            'n_observations': int(model.nobs)
        }
        
        print(f"Model fit statistics extracted successfully")
        return fit_stats
        
    except Exception as e:
        print(f"Error extracting model fit statistics: {e}")
        return None

def save_results(regression_results, assumption_results, fit_results, age_group_data, timestamp):
    """
    Save all results to CSV files.
    
    Parameters:
    regression_results (pd.DataFrame): Regression coefficients results
    assumption_results (pd.DataFrame): Assumption test results
    fit_results (pd.DataFrame): Model fit statistics
    age_group_data (pd.DataFrame): Data with age groups
    timestamp (str): Timestamp for filenames
    """
    print("Saving results to CSV files...")
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save regression results
    regression_file = f'outputs/step5_interaction_regression_results_{timestamp}.csv'
    regression_results.to_csv(regression_file, index=False)
    print(f"Saved regression results to: {regression_file}")
    
    # Save assumption test results
    assumption_file = f'outputs/step5_assumption_tests_{timestamp}.csv'
    assumption_results.to_csv(assumption_file, index=False)
    print(f"Saved assumption test results to: {assumption_file}")
    
    # Save model fit results
    fit_file = f'outputs/step5_model_fit_{timestamp}.csv'
    fit_results.to_csv(fit_file, index=False)
    print(f"Saved model fit results to: {fit_file}")
    
    # Save age group data
    age_group_file = f'outputs/step5_age_group_data_{timestamp}.csv'
    
    # Select only required columns for age group data
    age_group_columns = [
        'user_id', 'age', 'gender', 'education_level', 'country',
        'test_run_id', 'battery_id', 'time_of_day', 'grand_index',
        'age_bin', 'age_group', 'education_group', 'percentile_range', 'percentile_iqr'
    ]
    
    age_group_output = age_group_data[age_group_columns]
    age_group_output.to_csv(age_group_file, index=False)
    print(f"Saved age group data to: {age_group_file}")
    
    print("All results saved successfully")

def main():
    """
    Main function to execute the interaction regression analysis.
    
    Returns:
    int: 0 for success, 1 for error
    """
    try:
        print("Starting Step 5: Interaction Regression Analysis")
        print("="*50)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Timestamp: {timestamp}")
        
        # Find and load the input file
        input_pattern = "outputs/step3_heterogeneity_metrics_*.csv"
        input_file = find_latest_file(input_pattern)
        
        df = load_and_validate_data(input_file)
        if df is None:
            print("Error: Could not load input data")
            return 1
        
        # Create age groups
        df = create_age_groups(df)
        # Create education groups
        df = create_education_groups(df)
        
        # Clean data for regression
        df_clean = clean_data_for_regression(df)
        
        if len(df_clean) == 0:
            print("Error: No data remaining after cleaning")
            return 1
        
        print(f"Proceeding with {len(df_clean)} observations")
        
        # Initialize results storage
        all_regression_results = []
        all_assumption_results = []
        all_fit_results = []
        
        # Model 1: percentile_range
        print("\n" + "="*50)
        print("MODEL 1: PERCENTILE RANGE")
        print("="*50)
        
        model1, results1 = run_interaction_regression(df_clean, 'percentile_range', 'Model_1_Range')
        
        if model1 is not None:
            # Test assumptions
            assumptions1 = test_assumptions(model1, 'Model_1_Range', df_clean, 'percentile_range')
            all_assumption_results.append(assumptions1)
            
            # Extract fit statistics
            fit_stats1 = extract_model_fit_stats(model1, 'Model_1_Range', 'percentile_range')
            all_fit_results.append(fit_stats1)
            
            # Store results
            all_regression_results.append(results1)
            
            print("Model 1 completed successfully")
        else:
            print("Model 1 failed")
        
        # Model 2: percentile_iqr
        print("\n" + "="*50)
        print("MODEL 2: PERCENTILE IQR")
        print("="*50)
        
        model2, results2 = run_interaction_regression(df_clean, 'percentile_iqr', 'Model_2_IQR')
        
        if model2 is not None:
            # Test assumptions
            assumptions2 = test_assumptions(model2, 'Model_2_IQR', df_clean, 'percentile_iqr')
            all_assumption_results.append(assumptions2)
            
            # Extract fit statistics
            fit_stats2 = extract_model_fit_stats(model2, 'Model_2_IQR', 'percentile_iqr')
            all_fit_results.append(fit_stats2)
            
            # Store results
            all_regression_results.append(results2)
            
            print("Model 2 completed successfully")
        else:
            print("Model 2 failed")
        
        # Combine all results
        if all_regression_results:
            final_regression_results = pd.concat(all_regression_results, ignore_index=True)
            print(f"Combined regression results: {len(final_regression_results)} rows")
            
            # Print interaction results
            interaction_results = final_regression_results[final_regression_results['interaction_term'] == True]
            print(f"Interaction terms found: {len(interaction_results)}")
            
            if len(interaction_results) > 0:
                print("\nInteraction term results:")
                print(interaction_results[['model_name', 'predictor_variable', 'coefficient', 'p_value', 'significant_bonferroni']])
        else:
            print("No regression results to combine")
            return 1
        
        if all_assumption_results:
            final_assumption_results = pd.concat(all_assumption_results, ignore_index=True)
            print(f"Combined assumption test results: {len(final_assumption_results)} rows")
        else:
            print("No assumption test results to combine")
            return 1
        
        if all_fit_results:
            final_fit_results = pd.DataFrame(all_fit_results)
            print(f"Combined model fit results: {len(final_fit_results)} rows")
        else:
            print("No model fit results to combine")
            return 1
        
        # Save all results
        save_results(final_regression_results, final_assumption_results, 
                    final_fit_results, df_clean, timestamp)
        
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        # Print summary of significant interactions
        significant_interactions = final_regression_results[
            (final_regression_results['interaction_term'] == True) & 
            (final_regression_results['significant_bonferroni'] == True)
        ]
        
        if len(significant_interactions) > 0:
            print(f"Found {len(significant_interactions)} significant interaction terms after Bonferroni correction:")
            for _, row in significant_interactions.iterrows():
                print(f"- {row['model_name']}: {row['predictor_variable']} (p={row['p_value']:.4f})")
        else:
            print("No significant interaction terms found after Bonferroni correction")
        
        # Print assumption test summary
        print("\nAssumption test summary:")
        assumption_summary = final_assumption_results.groupby(['model_name', 'test_name'])['assumption_met'].first()
        for (model, test), met in assumption_summary.items():
            status = "MET" if met else "VIOLATED"
            print(f"- {model} {test}: {status}")
        
        print("\nInteraction regression analysis completed successfully")
        print("Finished execution")
        return 0
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
