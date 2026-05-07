import pandas as pd
import os
import argparse

def aggregate_csv_files(directory):
    # List all files in the directory
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Filter only .csv files
    csv_files = [f for f in all_files if f.endswith('.csv')]
    print(f"Found {len(csv_files)} csv files")

    if not csv_files:
        print("No CSV files found in the provided directory.")
        return

    # Load all CSV files into a list of dataframes
    dfs = [pd.read_csv(os.path.join(directory, csv_file)) for csv_file in csv_files]

    # Concatenate all dataframes into a single dataframe
    concatenated_df = pd.concat(dfs, ignore_index=True)
    # print(concatenated_df.columns)
    # for i, r in concatenated_df.iterrows():
    #     if r['smooth_iters'] != 2 or r['smoothing'] != 1000: continue
    #     print(r['correct'], r['answers'].split("\n")[0], "|", r['true_answer'])
    concatenated_df['count'] = len(csv_files)

    # Group by columns 'A' and 'B', then calculate mean
    groupby_cols = ['smoothing']
    if 'smooth_iters' in concatenated_df.columns:
        groupby_cols.append("smooth_iters")
    aggregated_df = concatenated_df.groupby(groupby_cols).mean(["correct"]).reset_index().sort_values(["smooth_iters","smoothing"])

    # Save the result to a new CSV file
    print(aggregated_df)
    #aggregated_df.to_csv(os.path.join(directory, 'aggregated_output.csv'), index=False)
    #print("Aggregation complete. Results saved to 'aggregated_output.csv'.")

def main():
    parser = argparse.ArgumentParser(description="Aggregate data from CSV files in a given directory.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing the CSV files")

    args = parser.parse_args()

    if os.path.exists(args.directory_path) and os.path.isdir(args.directory_path):
        aggregate_csv_files(args.directory_path)
    else:
        print("Invalid directory path provided.")

if __name__ == "__main__":
    main()
