import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
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
        return None
    return max(files, key=os.path.getctime)

def load_and_validate_data(file_path):
    """
    Load and validate the input dataset.
    
    Parameters:
    file_path (str): Path to the input CSV file
    
    Returns:
    pd.DataFrame: Validated dataset
    """
    print(f"Loading data from: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"All columns: {list(df.columns)}")
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
        'percentile_37', 'percentile_range', 'percentile_iqr'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return None
    
    print("All required columns present")
    
    # Print unique values for key experimental design parameters
    print(f"Unique education_level values: {sorted(df['education_level'].unique())}")
    print(f"Unique gender values: {df['gender'].unique()}")
    print(f"Unique country values: {df['country'].unique()}")
    print(f"Unique time_of_day values: {sorted(df['time_of_day'].unique())}")
    print(f"Unique age_bin values: {df['age_bin'].unique()}")
    print(f"Unique battery_id values: {sorted(df['battery_id'].unique())}")
    
    # Print data types for measured variables
    print(f"percentile_range data type: {df['percentile_range'].dtype}")
    print(f"percentile_iqr data type: {df['percentile_iqr'].dtype}")
    print(f"age data type: {df['age'].dtype}")
    print(f"grand_index data type: {df['grand_index'].dtype}")
    
    return df

def clean_data(df):
    """
    Clean the dataset by removing rows with missing values in key columns.
    
    Parameters:
    df (pd.DataFrame): Input dataset
    
    Returns:
    pd.DataFrame: Cleaned dataset
    """
    print("Cleaning data...")
    
    initial_rows = len(df)
    print(f"Initial number of rows: {initial_rows}")
    
    # Key columns for analysis
    key_columns = ['education_level', 'age', 'gender', 'country', 'time_of_day', 
                   'percentile_range', 'percentile_iqr']
    
    # Check for missing values in key columns
    for col in key_columns:
        missing_count = df[col].isnull().sum()
        print(f"Missing values in {col}: {missing_count}")
    
    # Remove rows with missing values in key columns
    df_clean = df.dropna(subset=key_columns)
    
    final_rows = len(df_clean)
    excluded_rows = initial_rows - final_rows
    
    print(f"Final number of rows: {final_rows}")
    print(f"Excluded rows due to missing values: {excluded_rows}")
    
    return df_clean

def prepare_regression_data(df):
    """
    Prepare data for regression analysis by creating dummy variables and formatting categorical variables.
    
    Parameters:
    df (pd.DataFrame): Input dataset
    
    Returns:
    pd.DataFrame: Prepared dataset for regression
    """
    print("Preparing data for regression analysis...")
    
    # Create a copy for regression
    reg_df = df.copy()
    
    # Ensure categorical variables are properly formatted
    reg_df['gender'] = reg_df['gender'].astype(str)
    reg_df['country'] = reg_df['country'].astype(str)
    reg_df['time_of_day'] = reg_df['time_of_day'].astype(str)
    reg_df['education_level'] = reg_df['education_level'].astype(int)
    
    print(f"Education level distribution:")
    print(reg_df['education_level'].value_counts().sort_index())
    
    print(f"Gender distribution:")
    print(reg_df['gender'].value_counts())
    
    print(f"Country distribution:")
    print(reg_df['country'].value_counts())
    
    print(f"Time of day distribution:")
    print(reg_df['time_of_day'].value_counts())
    
    # Binarize country variable: US vs Other using vectorized operation
    reg_df['country'] = np.where(reg_df['country'] == 'US', 'US', 'Other')

    # Bin time_of_day into four categories using pd.cut
    reg_df['time_of_day_binned'] = pd.cut(
        reg_df['time_of_day'].astype(int), 
        bins=[-1, 4, 11, 17, 23], 
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        include_lowest=True
    )

    # Print distributions of new variables
    print(f"Country distribution (binarized):")
    print(reg_df['country'].value_counts())

    print(f"Time of day binned distribution:")
    print(reg_df['time_of_day_binned'].value_counts())
    
    return reg_df

def run_regression_model(df, dependent_var, model_name):
    """
    Run a multiple linear regression model with assumption testing.
    
    Parameters:
    df (pd.DataFrame): Dataset for regression
    dependent_var (str): Name of the dependent variable
    model_name (str): Name identifier for the model
    
    Returns:
    tuple: (model_results, assumption_results)
    """
    print(f"\nRunning regression model: {model_name}")
    print(f"Dependent variable: {dependent_var}")
    
    # Define the formula with new binned variables and reference categories
    formula = f"{dependent_var} ~ C(education_level, Treatment(4)) + age + C(gender) + C(country, Treatment('US')) + C(time_of_day_binned, Treatment('Morning'))"
    
    print(f"Formula: {formula}")
    
    # Fit the model
    model = ols(formula, data=df).fit()
    
    print(f"Model summary:")
    print(model.summary())
    
    # Extract model results
    results = []
    
    # Get coefficients and statistics
    coef_df = pd.DataFrame({
        'coefficient': model.params,
        'std_error': model.bse,
        't_statistic': model.tvalues,
        'p_value': model.pvalues,
        'conf_int_lower': model.conf_int()[0],
        'conf_int_upper': model.conf_int()[1]
    })
    
    # Calculate standardized coefficients
    X = model.model.exog
    y = model.model.endog

    # Standardize continuous variables for beta calculation
    X_std = X.copy()
    y_std = (y - np.mean(y)) / np.std(y)

    # For continuous variables, standardize
    for i in range(X.shape[1]):
        if np.var(X[:, i]) > 0:  # Only standardize if there's variance
            X_std[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

    # Calculate standardized coefficients
    try:
        beta_model = sm.OLS(y_std, X_std).fit()
        # Convert numpy array to pandas Series with proper parameter names
        standardized_betas = pd.Series(beta_model.params, index=model.model.exog_names)
        print(f"Standardized betas calculated successfully")
    except Exception as e:
        # If standardization fails, create a Series with original coefficients
        print(f"Standardized beta calculation failed: {e}")
        standardized_betas = pd.Series(model.params, index=model.params.index)
    
    # Apply Bonferroni correction (alpha = 0.025 for two tests)
    bonferroni_alpha = 0.025
    
    for param_name, row in coef_df.iterrows():
        results.append({
            'model_name': model_name,
            'dependent_variable': dependent_var,
            'predictor_variable': param_name,
            'coefficient': row['coefficient'],
            'std_error': row['std_error'],
            'standardized_beta': standardized_betas.get(param_name, row['coefficient']),
            't_statistic': row['t_statistic'],
            'p_value': row['p_value'],
            'conf_int_lower': row['conf_int_lower'],
            'conf_int_upper': row['conf_int_upper'],
            'significant_bonferroni': row['p_value'] < bonferroni_alpha
        })
    
    # Model fit statistics
    model_fit = {
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
    
    # Assumption testing
    assumption_results = []
    
    # 1. Normality test (Shapiro-Wilk)
    try:
        shapiro_stat, shapiro_p = stats.shapiro(model.resid)
        assumption_results.append({
            'model_name': model_name,
            'test_name': 'Shapiro-Wilk',
            'test_statistic': shapiro_stat,
            'p_value': shapiro_p,
            'assumption_met': shapiro_p > 0.05
        })
        print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
    except Exception as e:
        print(f"Shapiro-Wilk test failed: {e}")
    
    # 2. Homoscedasticity test (Breusch-Pagan)
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
        assumption_results.append({
            'model_name': model_name,
            'test_name': 'Breusch-Pagan',
            'test_statistic': bp_stat,
            'p_value': bp_p,
            'assumption_met': bp_p > 0.05
        })
        print(f"Breusch-Pagan test: statistic={bp_stat:.4f}, p-value={bp_p:.4f}")
    except Exception as e:
        print(f"Breusch-Pagan test failed: {e}")
    
    # 3. Multicollinearity (VIF) - exclude intercept
    try:
        vif_data = pd.DataFrame()
        # Exclude intercept (index 0) from VIF calculation
        vif_data["Variable"] = [model.model.exog_names[i] for i in range(1, model.model.exog.shape[1])]
        vif_data["VIF"] = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
        
        max_vif = vif_data["VIF"].max()
        assumption_results.append({
            'model_name': model_name,
            'test_name': 'VIF',
            'test_statistic': max_vif,
            'p_value': np.nan,
            'assumption_met': max_vif < 10
        })
        print(f"VIF test: max VIF={max_vif:.4f}")
        print("VIF values:")
        print(vif_data)
    except Exception as e:
        print(f"VIF calculation failed: {e}")
    
    # 4. Independence test (Durbin-Watson)
    try:
        dw_stat = durbin_watson(model.resid)
        assumption_results.append({
            'model_name': model_name,
            'test_name': 'Durbin-Watson',
            'test_statistic': dw_stat,
            'p_value': np.nan,
            'assumption_met': 1.5 < dw_stat < 2.5
        })
        print(f"Durbin-Watson test: statistic={dw_stat:.4f}")
    except Exception as e:
        print(f"Durbin-Watson test failed: {e}")
    
    return results, assumption_results, model_fit

def save_results(regression_results, assumption_results, model_fit_results, timestamp):
    """
    Save all results to CSV files.
    
    Parameters:
    regression_results (list): List of regression results
    assumption_results (list): List of assumption test results
    model_fit_results (list): List of model fit statistics
    timestamp (str): Timestamp for filename
    """
    print("Saving results to CSV files...")
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save regression results
    reg_df = pd.DataFrame(regression_results)
    reg_filename = f'outputs/step4_primary_regression_results_{timestamp}.csv'
    reg_df.to_csv(reg_filename, index=False)
    print(f"Regression results saved to: {reg_filename}")
    print("Regression results columns:", list(reg_df.columns))
    print("First 2 rows of regression results:")
    print(reg_df.head(2))
    
    # Save assumption test results
    assumption_df = pd.DataFrame(assumption_results)
    assumption_filename = f'outputs/step4_assumption_tests_{timestamp}.csv'
    assumption_df.to_csv(assumption_filename, index=False)
    print(f"Assumption test results saved to: {assumption_filename}")
    print("Assumption test results columns:", list(assumption_df.columns))
    print("First 2 rows of assumption test results:")
    print(assumption_df.head(2))
    
    # Save model fit results
    fit_df = pd.DataFrame(model_fit_results)
    fit_filename = f'outputs/step4_model_fit_{timestamp}.csv'
    fit_df.to_csv(fit_filename, index=False)
    print(f"Model fit results saved to: {fit_filename}")
    print("Model fit results columns:", list(fit_df.columns))
    print("First 2 rows of model fit results:")
    print(fit_df.head(2))

def main():
    """
    Main function to execute the regression analysis pipeline.
    
    Returns:
    int: 0 for success, 1 for error
    """
    try:
        print("Starting primary regression analysis...")
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Timestamp: {timestamp}")
        
        # Find the most recent input file
        input_pattern = "outputs/step3_heterogeneity_metrics_*.csv"
        input_file = find_latest_file(input_pattern)
        
        if not input_file:
            print(f"No input file found matching pattern: {input_pattern}")
            return 1
        
        print(f"Using input file: {input_file}")
        
        # Load and validate data
        df = load_and_validate_data(input_file)
        if df is None:
            print("Failed to load and validate data")
            return 1
        
        # Clean data
        df_clean = clean_data(df)
        if len(df_clean) == 0:
            print("No valid data remaining after cleaning")
            return 1
        
        # Prepare data for regression
        reg_df = prepare_regression_data(df_clean)
        
        # Initialize results lists
        all_regression_results = []
        all_assumption_results = []
        all_model_fit_results = []
        
        # Model 1: percentile_range
        print("\n" + "="*60)
        print("MODEL 1: PERCENTILE RANGE")
        print("="*60)
        
        reg_results_1, assumption_results_1, model_fit_1 = run_regression_model(
            reg_df, 'percentile_range', 'Model_1_percentile_range'
        )
        
        all_regression_results.extend(reg_results_1)
        all_assumption_results.extend(assumption_results_1)
        all_model_fit_results.append(model_fit_1)
        
        # Model 2: percentile_iqr
        print("\n" + "="*60)
        print("MODEL 2: PERCENTILE IQR")
        print("="*60)
        
        reg_results_2, assumption_results_2, model_fit_2 = run_regression_model(
            reg_df, 'percentile_iqr', 'Model_2_percentile_iqr'
        )
        
        all_regression_results.extend(reg_results_2)
        all_assumption_results.extend(assumption_results_2)
        all_model_fit_results.append(model_fit_2)
        
        # Save all results
        save_results(all_regression_results, all_assumption_results, all_model_fit_results, timestamp)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total regression coefficients analyzed: {len(all_regression_results)}")
        print(f"Total assumption tests performed: {len(all_assumption_results)}")
        print(f"Total models fitted: {len(all_model_fit_results)}")
        
        # Print significant results with Bonferroni correction
        significant_results = [r for r in all_regression_results if r['significant_bonferroni']]
        print(f"\nSignificant results (Bonferroni corrected, α = 0.025): {len(significant_results)}")
        
        for result in significant_results:
            print(f"  {result['model_name']}: {result['predictor_variable']} "
                  f"(β = {result['coefficient']:.4f}, p = {result['p_value']:.4f})")
        
        # Print assumption test summary
        print(f"\nAssumption test summary:")
        for result in all_assumption_results:
            if not pd.isna(result['p_value']):
                print(f"  {result['model_name']} - {result['test_name']}: "
                      f"{'PASS' if result['assumption_met'] else 'FAIL'} "
                      f"(p = {result['p_value']:.4f})")
            else:
                print(f"  {result['model_name']} - {result['test_name']}: "
                      f"{'PASS' if result['assumption_met'] else 'FAIL'} "
                      f"(statistic = {result['test_statistic']:.4f})")
        
        print("Finished execution")
        return 0
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
