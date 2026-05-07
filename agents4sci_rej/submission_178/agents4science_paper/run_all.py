import pandas as pd
import os
import numpy as np

def find_header_row(df, key_columns):
    """Finds the row index of a table's header by searching for key column names."""
    for idx, row in df.iterrows():
        row_str = ' '.join(str(cell) for cell in row if pd.notna(cell))
        # Check if all key columns are present in the row's text
        if all(key_col in row_str for key_col in key_columns):
            return idx
    return None

def extract_data_below_header(df, header_row_idx):
    """Extracts all data rows below a found header row until an empty row is encountered."""
    # The actual data starts on the next row
    data_start_row = header_row_idx + 1
    
    # Find where the data ends (the first row that is entirely empty)
    end_row = data_start_row
    while end_row < len(df) and not df.iloc[end_row].isnull().all():
        end_row += 1
        
    if data_start_row >= end_row:
        return None # No data found below header
        
    # Extract the data
    table_df = df.iloc[data_start_row:end_row].copy()
    # Set the column names from the header row
    table_df.columns = df.iloc[header_row_idx]
    
    return table_df

def parse_inspire_excel_robust(filepath):
    """
    Parses a single iNSPiRe Excel file by finding tables based on their
    unique column headers, and combines them with file-level metadata.
    """
    filename = os.path.basename(filepath)
    print(f"--- Processing: {filename} ---")
    
    df_full = None
    try:
        xls = pd.ExcelFile(filepath)
        sheet_name = 'Demand & Consumption' if 'Demand & Consumption' in xls.sheet_names else xls.sheet_names[0]
        df_full = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"  [Warning] Could not read file. Skipping. Error: {e}")
        return None

    # 1. Extract file-level metadata
    metadata = {
        'btype_general': df_full.iloc[2, 1],
        'period': df_full.iloc[3, 1],
        'system': df_full.iloc[4, 1],
        'terminal': df_full.iloc[5, 1],
        'source_file': filename
    }

    # 2. Define tables by their title and a few unique, key columns
    tables_to_find = {
        "Building stock statistics": ['Inhabitants/employs (Mil.)', 'Average demand (kWh/m2y)'],
        "Reference buildings simulations": ['S/V - External surface over Volume ratio', 'Share per Type'],
        "Target buildings simulations": ['Retrofit energy level heating demand', 'Solar thermal coll. area']
    }
    
    all_file_data = []

    # 3. Find and extract each table by its key columns
    for title, key_cols in tables_to_find.items():
        header_row = find_header_row(df_full, key_cols)
        
        if header_row is not None:
            table_df = extract_data_below_header(df_full, header_row)

            if table_df is not None:
                table_df['data_source_table'] = title
                for key, value in metadata.items():
                    table_df[key] = value
                
                # Drop columns that are completely empty
                table_df.dropna(axis=1, how='all', inplace=True)
                all_file_data.append(table_df)
                print(f"  -> Found and extracted '{title}' ({len(table_df)} rows)")

    if not all_file_data:
        print("  [Warning] No matching data tables found in this file.")
        return None
    
    return pd.concat(all_file_data, ignore_index=True)

# --- Main Program ---
def main():
    root_dir = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\iNSPiRe FP7 Retrofit Database'
    output_path = r'C:\Users\liang\Desktop\Agent4Science2025\dataset\train_final_consolidated_v2.csv'
    
    print("--- 🚀 Starting Automated Data Consolidation (Robust Version) ---")
    
    all_dfs = [parse_inspire_excel_robust(os.path.join(root_dir, f)) 
               for f in os.listdir(root_dir) if f.endswith('.xlsx') and not f.startswith('~')]
    
    valid_dfs = [df for df in all_dfs if df is not None]
    
    if not valid_dfs:
        print("\n[Error] No data could be extracted from any files. Process halted.")
        return

    final_dataset = pd.concat(valid_dfs, ignore_index=True)
    # A final clean step to remove rows that might be all empty
    final_dataset.dropna(how='all', subset=final_dataset.columns.difference(['source_file']), inplace=True)
    
    final_dataset.to_csv(output_path, index=False, encoding='utf-8')
    
    print("\n--- ✨ Task Complete! ---")
    print(f"✅ Processed {len(valid_dfs)} files successfully.")
    print(f"✅ Final consolidated dataset has {len(final_dataset)} rows.")
    print(f"✅ Data saved to: {output_path}")

if __name__ == "__main__":
    main()