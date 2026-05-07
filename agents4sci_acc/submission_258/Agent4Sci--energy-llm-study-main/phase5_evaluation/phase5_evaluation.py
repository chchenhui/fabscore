#!/usr/bin/env python3
"""
phase5_evaluation.py

Performs the final Phase 5 evaluation of the experiment. This script generates
the main analytical results, including task-level case studies, aggregate
statistical analysis, visualizations, and sensitivity analysis of the measurements.

It reads the outputs from Phase 3 (raw runs) and Phase 4 (reranked selections)
to produce a comprehensive report on the effectiveness of energy-guided selection.

Usage:
  python phase5_evaluation.py \
    --phase3_dirs ./outputs/phase3 ./outputs/phase3_rerun \
    --phase4_dir ./phase4_analysis \
    --candidates_dir ./filtered_candidates-codellama:70b-instruct-v5 \
    --output_dir ./phase5_evaluation
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import warnings

# Suppress warnings from matplotlib about fonts
warnings.filterwarnings("ignore", category=UserWarning)


def analyze_code_structure(file_path):
    """
    A simple heuristic to guess the algorithmic structure of a candidate.
    Checks for keywords like 'numpy' or common list comprehension patterns.
    """
    if not file_path or not Path(file_path).exists():
        return "Source Not Found"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'import numpy' in content or 'np.' in content:
                return "Vectorized (NumPy)"
            if '[' in content and ' for ' in content and ' in ' in content:
                return "Loop-based (Comprehension)"
            if 'for ' in content and ' in ' in content:
                 return "Loop-based (Standard)"
            return "Unknown"
    except Exception:
        return "Analysis Error"


def run_evaluation(phase3_dirs, phase4_dir, candidates_dir, output_dir):
    """Main function to orchestrate the Phase 5 evaluation."""
    p3_paths = [Path(d) for d in phase3_dirs]
    p4_path = Path(phase4_dir)
    cand_path = Path(candidates_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_path = out_path / 'evaluation_summary_report.txt'

    with open(report_path, 'w', encoding='utf-8') as report_file:
        def log(message):
            """Helper to print to console and write to the report file."""
            print(message)
            report_file.write(message + '\n')

        log("--- Starting Phase 5: Evaluation ---")

        # --- Load Data ---
        # NOTE: The guide refers to candidate_selection.csv, we use a more detailed version
        # from our phase 4 script which is functionally equivalent for this analysis.
        selections_file = p4_path / 'detailed_selections.csv' 
        if not selections_file.exists():
            log(f"Error: Phase 4 output file '{selections_file.name}' not found in {p4_path}. Please run Phase 4 first.")
            return
        
        selections_df = pd.read_csv(selections_file)
        log(f"Loaded selection data from {selections_file}")

        # --- 1. Task-Level Analysis ---
        log("\n--- 1. Task-Level Analysis ---")
        
        # Calculate savings and penalties as defined in the guide
        selections_df['savings_vs_top1'] = (selections_df['energy_median_kwh_top1'] - selections_df['energy_median_kwh_energy']) / selections_df['energy_median_kwh_top1']
        selections_df['savings_vs_besttime'] = (selections_df['energy_median_kwh_best_time'] - selections_df['energy_median_kwh_energy']) / selections_df['energy_median_kwh_best_time']
        selections_df['runtime_penalty_vs_besttime'] = selections_df['runtime_median_energy'] - selections_df['runtime_median_best_time']
        
        case_studies_df = selections_df[selections_df['candidate_id_energy'] != selections_df['candidate_id_best_time']].copy()
        
        log(f"\nFound {len(case_studies_df)} case studies where Energy-Guided and Best-Time diverge.")
        
        # Augment case studies with algorithmic structure info
        if not case_studies_df.empty:
            log("Analyzing algorithmic structure for case studies...")
            meta_files = []
            for p3_path in p3_paths:
                meta_files.extend(list(p3_path.glob('metadata/*_metadata.csv')))

            if meta_files:
                all_meta = pd.concat([pd.read_csv(f) for f in meta_files]).drop_duplicates(subset=['task', 'candidate_id'], keep='last')
                
                # Create a map for easy lookup
                path_map = all_meta.set_index(['task', 'candidate_id'])['filepath'].to_dict()

                def get_structure(row, cand_type):
                    key = (row['task'], row[cand_type])
                    return analyze_code_structure(path_map.get(key))

                case_studies_df['structure_energy'] = case_studies_df.apply(get_structure, cand_type='candidate_id_energy', axis=1)
                case_studies_df['structure_best_time'] = case_studies_df.apply(get_structure, cand_type='candidate_id_best_time', axis=1)

                case_studies_path = out_path / 'case_studies.csv'
                case_studies_df.to_csv(case_studies_path, index=False)
                log(f"Saved {len(case_studies_df)} detailed case studies to {case_studies_path}")

        # Generate per-task report table
        task_level_report = selections_df[[
            'task', 'scale', 'candidate_id_energy', 'candidate_id_best_time', 'candidate_id_top1',
            'energy_median_kwh_energy', 'runtime_median_energy', 'savings_vs_top1', 'savings_vs_besttime'
        ]].rename(columns={
            'task': 'Task', 'scale': 'Scale', 'candidate_id_energy': 'Energy-Guided ID',
            'candidate_id_best_time': 'Best-Time ID', 'candidate_id_top1': 'Top-1 ID',
            'energy_median_kwh_energy': 'Energy (kWh)', 'runtime_median_energy': 'Runtime (s)',
            'savings_vs_top1': 'Savings vs Top-1 (%)', 'savings_vs_besttime': 'Savings vs Best-Time (%)'
        })
        task_level_report_path = out_path / 'task_level_summary.csv'
        task_level_report.to_csv(task_level_report_path, index=False, float_format='%.6f')
        log(f"Saved task-level summary table to {task_level_report_path}")

        # --- 2. Aggregate Analysis ---
        log("\n\n--- 2. Aggregate Analysis ---")
        
        # Energy efficiency improvement
        agg_stats = {
            'avg_saving_vs_top1': selections_df['savings_vs_top1'].mean(),
            'median_saving_vs_top1': selections_df['savings_vs_top1'].median(),
            'avg_saving_vs_besttime': selections_df['savings_vs_besttime'].mean(),
            'median_saving_vs_besttime': selections_df['savings_vs_besttime'].median(),
            'avg_runtime_penalty': selections_df['runtime_penalty_vs_besttime'].mean(),
            'median_runtime_penalty': selections_df['runtime_penalty_vs_besttime'].median()
        }
        log("\nEnergy Efficiency Improvement (vs. Baselines):")
        log(f"  - Average Savings vs. Top-1:      {agg_stats['avg_saving_vs_top1']:.2%}")
        log(f"  - Average Savings vs. Best-Time:  {agg_stats['avg_saving_vs_besttime']:.2%}")
        log("\nRuntime Penalty (vs. Best-Time):")
        log(f"  - Average Penalty: {agg_stats['avg_runtime_penalty']:.4f} seconds")
        log(f"  - Median Penalty:  {agg_stats['median_runtime_penalty']:.4f} seconds")

        pd.DataFrame([agg_stats]).to_csv(out_path / 'aggregate_summary_stats.csv', index=False)

        # Distribution plots
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(selections_df['savings_vs_besttime'].dropna() * 100, bins=20, kde=True, ax=axes[0])
        axes[0].set_title('Distribution of Energy Savings vs. Best-Time')
        axes[0].set_xlabel('Relative Energy Saving (%)')
        sns.boxplot(x=selections_df['savings_vs_besttime'].dropna() * 100, ax=axes[1])
        axes[1].set_title('Box Plot of Energy Savings')
        plot_path = out_path / 'energy_savings_distribution.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        log(f"\nGenerated plot of energy savings distribution at: {plot_path}")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(selections_df['runtime_penalty_vs_besttime'].dropna(), bins=20, kde=True, ax=axes[0])
        axes[0].set_title('Distribution of Runtime Penalty vs. Best-Time')
        axes[0].set_xlabel('Runtime Penalty (seconds)')
        sns.boxplot(x=selections_df['runtime_penalty_vs_besttime'].dropna(), ax=axes[1])
        axes[1].set_title('Box Plot of Runtime Penalty')
        plot_path_penalty = out_path / 'runtime_penalty_distribution.png'
        plt.tight_layout()
        plt.savefig(plot_path_penalty)
        log(f"Generated plot of runtime penalty distribution at: {plot_path_penalty}")
        plt.close()

        # Statistical significance
        log("\nStatistical Significance (Wilcoxon signed-rank test):")
        for baseline in ['top1', 'best_time']:
            energy_guided = selections_df['energy_median_kwh_energy']
            energy_baseline = selections_df[f'energy_median_kwh_{baseline}']
            diff_mask = (energy_guided != energy_baseline).dropna()
            if diff_mask.sum() > 0:
                stat, p_value = wilcoxon(energy_guided[diff_mask], energy_baseline[diff_mask], alternative='less')
                log(f"\n  Comparison vs. {baseline.replace('_', '-').title()}:")
                log(f"    - p-value: {p_value:.5f}")
                log(f"    - Result: {'Statistically significant (p < 0.05)' if p_value < 0.05 else 'Not statistically significant (p >= 0.05)'}")
            else:
                log(f"\n  Comparison vs. {baseline.replace('_', '-').title()}: Skipped, no energy differences found.")

        # --- 3. Sensitivity Analysis ---
        log("\n\n--- 3. Sensitivity Analysis ---")
        raw_results_files = []
        for p3_path in p3_paths:
            if (p3_path / 'results').exists():
                raw_results_files.extend(list((p3_path / 'results').glob('*_results.csv')))
        
        if raw_results_files:
            latest_files_map = {}
            for f in raw_results_files:
                task_name = f.name.replace('_results.csv', '')
                mod_time = f.stat().st_mtime
                if task_name not in latest_files_map or mod_time > latest_files_map[task_name][1]:
                    latest_files_map[task_name] = (f, mod_time)
            
            final_files_to_load = [f[0] for f in latest_files_map.values()]
            raw_df = pd.concat([pd.read_csv(f) for f in final_files_to_load])
            
            raw_df['runtime_s'] = pd.to_numeric(raw_df['runtime_s'], errors='coerce')
            raw_df['energy_kwh'] = pd.to_numeric(raw_df['energy_kwh'], errors='coerce')

            cv_df = raw_df.groupby(['task', 'candidate_id', 'scale']).agg(
                runtime_cv=('runtime_s', lambda x: x.std() / x.mean()),
                energy_cv=('energy_kwh', lambda x: x.std() / x.mean())
            ).dropna()
            
            log("\nRobustness of measurements (Coefficient of Variation across runs):")
            log(f"  - Average CV for Runtime: {cv_df['runtime_cv'].mean():.2%}")
            log(f"  - Average CV for Energy:  {cv_df['energy_cv'].mean():.2%}")

        # Input scale effect
        ranking_stability = selections_df.groupby('task')['candidate_id_energy'].nunique()
        stable_tasks = (ranking_stability == 1).sum()
        total_tasks = len(ranking_stability)
        stability_frac = stable_tasks / total_tasks if total_tasks > 0 else 0
        
        log("\nRanking stability (Effect of input scale):")
        log(f"  - {stable_tasks} out of {total_tasks} tasks ({stability_frac:.2%}) had a consistent energy-guided candidate across all scales.")

        log("\n\n--- Phase 5 Evaluation Complete ---")
        log(f"Full report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Perform Phase 5 final evaluation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--phase3_dirs", nargs='+', required=True, help="One or more paths to Phase 3 output directories.")
    parser.add_argument("--phase4_dir", required=True, help="Path to the Phase 4 analysis directory.")
    parser.add_argument("--candidates_dir", required=True, help="Path to the root directory of candidate source files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the evaluation plots and reports.")
    
    args = parser.parse_args()
    run_evaluation(args.phase3_dirs, args.phase4_dir, args.candidates_dir, args.output_dir)

if __name__ == "__main__":
    main()

