import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def analyze_building_energy(filepath):
    """
    Parses a single iNSPiRe Excel file to analyze retrofit energy savings.

    This function reads three specific sheets, links the 'before' and 'after'
    retrofit data, calculates the savings percentage, and compares the
    results to the building stock's average consumption.

    Args:
        filepath (str): The full path to the .xlsx file.

    Returns:
        pandas.DataFrame: A DataFrame with the combined analysis, or None if
                          the file cannot be processed.
    """
    filename = os.path.basename(filepath)
    print(f"--- Analyzing File: {filename} ---")
    
    try:
        # --- STEP 1: Precisely read the three required sheets ---
        print("   -> Reading required sheets...")
        
        # Table 1: Building stock statistics (Header is on the 1st row)
        df_stock = pd.read_excel(filepath, sheet_name='Building stock statistics', header=0).dropna(how='all')
        
        # Before Retrofit: Reference buildings (Header is on the 2nd row)
        df_ref = pd.read_excel(filepath, sheet_name='Reference buildings simulations', header=0).dropna(how='all')
        
        # After Retrofit: Target buildings (Data header is on the 4th row)
        df_target = pd.read_excel(filepath, sheet_name='Target buildings simulations', header=1).dropna(how='all')
        
        print("   ✅ Sheets read successfully.")

    except Exception as e:
        print(f"  [ERROR] Could not read required sheets from {filename}. Error: {e}")
        return None

    # --- STEP 2: Link 'before' and 'after' data and calculate savings ---
    print("   -> Linking 'before' and 'after' simulation data...")
    
    # Prepare 'before' data
    df_ref_slim = df_ref[['Climate', 'Type of building', 'Age of construction', 'Consumption (kWh/m2y)']].copy()
    df_ref_slim.rename(columns={'Consumption (kWh/m2y)': 'consumption_before'}, inplace=True)

    # Prepare 'after' data
    df_target.rename(columns={'Building average yearly': 'consumption_after', 'Building type': 'Type of building'}, inplace=True)
    
    # Link the tables
    merge_keys = ['Climate', 'Type of building', 'Age of construction']
    df_simulations = pd.merge(df_target, df_ref_slim, on=merge_keys, how='inner')
    
    if df_simulations.empty:
        print("  [Warning] No matching records found between 'before' and 'after' data.")
        return None
    
    # Calculate savings percentage
    df_simulations['consumption_before'] = pd.to_numeric(df_simulations['consumption_before'], errors='coerce')
    df_simulations = df_simulations[df_simulations['consumption_before'] > 0]
    df_simulations['savings_percentage'] = (df_simulations['consumption_before'] - df_simulations['consumption_after']) / df_simulations['consumption_before']
    print("   ✅ Savings percentage calculated.")
    
    # --- STEP 3: Link with stock average consumption for comparison ---
    print("   -> Linking with building stock average for comparison...")
    
    # Get total average consumption from the stock data
    df_stock_slim = df_stock[['Climate', 'Type of building', 'Average consumption (kWh/m2y)']].copy()
    df_stock_avg = df_stock_slim.groupby(['Climate', 'Type of building'])['Average consumption (kWh/m2y)'].sum().reset_index()
    df_stock_avg.rename(columns={'Average consumption (kWh/m2y)': 'stock_average_consumption'}, inplace=True)

    # Merge the stock average into our main dataset
    df_final = pd.merge(df_simulations, df_stock_avg, on=['Climate', 'Type of building'], how='left')
    print("   ✅ Stock average data linked successfully.")

    return df_final

# --- Example of how to use this script ---
if __name__ == "__main__":
    
    # 1. DEFINE the path to the Excel file you want to analyze
    example_filepath = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\iNSPiRe FP7 Retrofit Database\SFH_I_AWHP_CEI.xlsx'
    
    # 2. DEFINE where you want to save the output CSV
    output_csv_path = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\full_retrofit_analysis.csv'

    print("--- 🚀 Starting Single File Analysis ---")
    
    if os.path.exists(example_filepath):
        # Call the main function to process the file
        analysis_data = analyze_building_energy(example_filepath)

        if analysis_data is not None and not analysis_data.empty:
            print("\n✅ File analysis successful!")
            
            # Save the resulting DataFrame to a CSV file
            analysis_data.to_csv(output_csv_path, index=False, encoding='utf-8')
            print(f"   Full analysis data saved to: {output_csv_path}")

            # Display a preview of the key columns
            preview_cols = [
                'Type of building',
                'consumption_before', 
                'consumption_after', 
                'savings_percentage',
                'stock_average_consumption' # The key comparison column
            ]
            print("\n--- Data Preview (Top 5 Rows) ---")
            print(analysis_data[preview_cols].head())
        else:
            print("\n❌ Could not extract valid analysis data from the file.")
    else:
        print(f"\n❌ ERROR: The example file was not found at the specified path.")
        print(f"   Please make sure the path is correct: {example_filepath}")