#!/usr/bin/env python3
"""
phase4_reranker.py

Performs Phase 4 candidate selection (reranking) based on the aggregated
results from Phase 3. It applies different selection strategies and compares
their outcomes.

The script reads a summary CSV, selects the best candidate for each task and
scale according to three rules (Energy-Guided, Top-1, Best-Time), and then
computes metrics to evaluate the trade-offs between these strategies, providing
an overall, per-task, and per-task-per-scale summary.

Usage:
  python phase4_reranker.py \
    --input_csv ./final_summary.csv \
    --output_dir ./phase4_analysis
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_metrics(df, group_name="Overall"):
    """Calculates comparison metrics for a given dataframe slice."""
    num_cases = len(df)
    if num_cases == 0:
        return None

    # A. Divergence from baselines
    diff_vs_top1 = (df['candidate_id_energy'] != df['candidate_id_top1']).sum()
    diff_vs_best_time = (df['candidate_id_energy'] != df['candidate_id_best_time']).sum()

    # B. Relative energy savings
    # Replace 0s in denominator with NaN to handle division by zero safely; .mean() ignores NaNs.
    energy_savings_vs_top1 = (
        (df['energy_median_kwh_top1'] - df['energy_median_kwh_energy']) /
        df['energy_median_kwh_top1'].replace(0, np.nan)
    ).mean()
    
    energy_savings_vs_best_time = (
        (df['energy_median_kwh_best_time'] - df['energy_median_kwh_energy']) /
        df['energy_median_kwh_best_time'].replace(0, np.nan)
    ).mean()

    # C. Relative runtime penalty
    runtime_penalty_vs_top1 = (
        (df['runtime_median_energy'] - df['runtime_median_top1']) /
        df['runtime_median_top1'].replace(0, np.nan)
    ).mean()

    runtime_penalty_vs_best_time = (
        (df['runtime_median_energy'] - df['runtime_median_best_time']) /
        df['runtime_median_best_time'].replace(0, np.nan)
    ).mean()

    return {
        "group": group_name,
        "num_cases": num_cases,
        "divergence_vs_top1_frac": diff_vs_top1 / num_cases,
        "divergence_vs_best_time_frac": diff_vs_best_time / num_cases,
        "energy_savings_vs_top1_avg_rel": energy_savings_vs_top1,
        "energy_savings_vs_best_time_avg_rel": energy_savings_vs_best_time,
        "runtime_penalty_vs_top1_avg_rel": runtime_penalty_vs_top1,
        "runtime_penalty_vs_best_time_avg_rel": runtime_penalty_vs_best_time,
    }

def analyze_and_rerank(input_csv, output_dir):
    """
    Loads aggregated data, applies selection rules, and calculates comparison metrics.
    """
    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Reading aggregated results from: {input_path}")
    df = pd.read_csv(input_path)

    # Step 1: Filter for only fully correct candidates
    correct_df = df[df['correct_fraction'] == 1.0].copy()
    if correct_df.empty:
        print("Error: No fully correct candidates found in the input file. Cannot proceed.")
        return
    print(f"Found {len(correct_df)} entries from fully correct candidates.")

    # Step 2: Apply the three selection rules
    top1_selections = correct_df.sort_values('candidate_id').groupby(['task', 'scale']).first()
    best_time_selections = correct_df.sort_values(['runtime_median', 'energy_median_kwh']).groupby(['task', 'scale']).first()
    energy_guided_selections = correct_df.sort_values(['energy_median_kwh', 'runtime_median']).groupby(['task', 'scale']).first()

    # Step 3: Merge selections and save detailed comparison files
    merged_df = energy_guided_selections.add_suffix('_energy') \
        .join(top1_selections.add_suffix('_top1')) \
        .join(best_time_selections.add_suffix('_best_time'))

    # Save the detailed comparison table
    merged_df.to_csv(output_path / 'detailed_selections.csv')
    print(f"\nDetailed selection data saved to: {output_path / 'detailed_selections.csv'}")

    # Save a simpler summary of just the candidate choices
    selections_summary = merged_df[['candidate_id_energy', 'candidate_id_top1', 'candidate_id_best_time']].rename(
        columns={
            'candidate_id_energy': 'energy_guided_selection',
            'candidate_id_top1': 'top1_selection',
            'candidate_id_best_time': 'best_time_selection'
        }
    )
    selections_summary.to_csv(output_path / 'selections_summary.csv')
    print(f"Candidate selection summary saved to: {output_path / 'selections_summary.csv'}")

    # Step 3b: Calculate and save per-task, per-scale analysis
    print(f"Calculating per-task, per-scale metrics...")
    per_scale_df = merged_df.copy()

    # Calculate divergence (boolean: 1 if different, 0 if same)
    per_scale_df['divergence_vs_top1'] = (per_scale_df['candidate_id_energy'] != per_scale_df['candidate_id_top1']).astype(int)
    per_scale_df['divergence_vs_best_time'] = (per_scale_df['candidate_id_energy'] != per_scale_df['candidate_id_best_time']).astype(int)

    # Calculate relative energy savings
    per_scale_df['energy_savings_vs_top1_rel'] = (
        (per_scale_df['energy_median_kwh_top1'] - per_scale_df['energy_median_kwh_energy']) /
        per_scale_df['energy_median_kwh_top1'].replace(0, np.nan)
    )
    per_scale_df['energy_savings_vs_best_time_rel'] = (
        (per_scale_df['energy_median_kwh_best_time'] - per_scale_df['energy_median_kwh_energy']) /
        per_scale_df['energy_median_kwh_best_time'].replace(0, np.nan)
    )

    # Calculate relative runtime penalty
    per_scale_df['runtime_penalty_vs_top1_rel'] = (
        (per_scale_df['runtime_median_energy'] - per_scale_df['runtime_median_top1']) /
        per_scale_df['runtime_median_top1'].replace(0, np.nan)
    )
    per_scale_df['runtime_penalty_vs_best_time_rel'] = (
        (per_scale_df['runtime_median_energy'] - per_scale_df['runtime_median_best_time']) /
        per_scale_df['runtime_median_best_time'].replace(0, np.nan)
    )

    # Select and rename columns for the final report
    columns_to_keep = {
        'candidate_id_energy': 'energy_guided_selection',
        'candidate_id_top1': 'top1_selection',
        'candidate_id_best_time': 'best_time_selection',
        'divergence_vs_top1': 'divergence_vs_top1',
        'divergence_vs_best_time': 'divergence_vs_best_time',
        'energy_savings_vs_top1_rel': 'energy_savings_vs_top1_rel',
        'energy_savings_vs_best_time_rel': 'energy_savings_vs_best_time_rel',
        'runtime_penalty_vs_top1_rel': 'runtime_penalty_vs_top1_rel',
        'runtime_penalty_vs_best_time_rel': 'runtime_penalty_vs_best_time_rel',
    }

    per_scale_summary = per_scale_df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)

    per_scale_summary.to_csv(output_path / 'per_task_per_scale_summary.csv')
    print(f"Per-task, per-scale analysis saved to: {output_path / 'per_task_per_scale_summary.csv'}")

    # Step 4: Calculate, print, and save the OVERALL metrics report
    overall_report = calculate_metrics(merged_df, "Overall")
    
    report_lines = []
    report_lines.append("--- Phase 4: Overall Analysis Report ---")
    
    if overall_report:
        report_lines.append(f"Analyzed {overall_report['num_cases']} unique (task, scale) combinations.\n")
        report_lines.append("1. Divergence from Baselines:")
        report_lines.append(f"   - Energy-Guided differs from Top-1 in {overall_report['divergence_vs_top1_frac']:.2%} of cases")
        report_lines.append(f"   - Energy-Guided differs from Best-Time in {overall_report['divergence_vs_best_time_frac']:.2%} of cases\n")
        report_lines.append("2. Energy Savings (vs. Baselines):")
        report_lines.append(f"   - Avg. relative energy savings vs. Top-1: {overall_report['energy_savings_vs_top1_avg_rel']:.2%}")
        report_lines.append(f"   - Avg. relative energy savings vs. Best-Time: {overall_report['energy_savings_vs_best_time_avg_rel']:.2%}\n")
        report_lines.append("3. Runtime Penalty (vs. Baselines):")
        report_lines.append(f"   - Avg. relative runtime change vs. Top-1: {overall_report['runtime_penalty_vs_top1_avg_rel']:+.2%}")
        report_lines.append(f"   - Avg. relative runtime change vs. Best-Time: {overall_report['runtime_penalty_vs_best_time_avg_rel']:+.2%}")
    
    report_lines.append("--- End of Overall Report ---")

    # Join lines, print to console, and write to file
    report_str = "\n".join(report_lines)
    print("\n" + report_str + "\n")

    report_txt_path = output_path / 'overall_analysis_report.txt'
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write(report_str)
    print(f"Overall analysis report saved to: {report_txt_path}")

    # Also save the raw numbers to a CSV for machine readability if the report was generated
    if overall_report:
        overall_df = pd.DataFrame([overall_report]).set_index('group')
        overall_csv_path = output_path / 'overall_summary.csv'
        overall_df.to_csv(overall_csv_path)
        print(f"Overall summary data saved to: {overall_csv_path}")

    # Step 5: Calculate and report PER-TASK metrics
    per_task_reports = []
    tasks = merged_df.index.get_level_values('task').unique()
    
    for task in sorted(tasks):
        task_df = merged_df.loc[pd.IndexSlice[task, :], :]
        task_report = calculate_metrics(task_df, group_name=task)
        if task_report:
            per_task_reports.append(task_report)

    if per_task_reports:
        per_task_df = pd.DataFrame(per_task_reports).set_index('group')
        per_task_df.to_csv(output_path / 'per_task_summary.csv')
        print(f"\nPer-task analysis summary saved to: {output_path / 'per_task_summary.csv'}")

        print("\n--- Phase 4: Per-Task Analysis Report ---")
        with pd.option_context('display.max_rows', None, 'display.width', 120):
             # Formatting for cleaner console output
            formatted_df = per_task_df.copy()
            for col in formatted_df.columns:
                if 'frac' in col or 'avg_rel' in col:
                     # Use + for penalty to show sign, regular for others
                    sign = '+' if 'penalty' in col else ''
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:{sign}.2%}" if pd.notna(x) else "N/A")
            print(formatted_df)
        print("--- End of Per-Task Report ---")


def main():
    parser = argparse.ArgumentParser(
        description="Perform Phase 4 reranking and analysis on aggregated results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_csv", 
        required=True,
        help="Path to the aggregated summary CSV from Phase 3 (e.g., './final_summary.csv')."
    )
    parser.add_argument(
        "--output_dir", 
        required=True,
        help="Directory to save the detailed analysis files."
    )
    args = parser.parse_args()
    
    analyze_and_rerank(args.input_csv, args.output_dir)

if __name__ == "__main__":
    main()