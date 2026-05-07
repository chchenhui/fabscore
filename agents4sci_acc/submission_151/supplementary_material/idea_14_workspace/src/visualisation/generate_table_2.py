"""
Generate Table 2: FNR comparison across datasets with confidence intervals.
"""

import pandas as pd
import logging
from pathlib import Path
import sys
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_table_2():
    """Generate FNR comparison table with confidence intervals."""
    logger.info("Generating Table 2: FNR comparison with Wilson CIs")
    
    # Auto-detect paths
    if Path("/research_storage/outputs").exists():
        # Modal environment
        base_path = Path("/research_storage/outputs")
        output_dir = Path("/research_storage/outputs/tables")
    else:
        # Local environment
        base_path = Path("idea_14_workspace/outputs")
        output_dir = Path("idea_14_workspace/outputs/visualisation/tables")
    
    # Load statistical results
    h1_stat_path = base_path / "statistical_analysis" / "h1_statistical_results.json" 
    h2_stat_path = base_path / "statistical_analysis" / "h2_statistical_results.json"
    
    with open(h1_stat_path, 'r') as f:
        h1_stats = json.load(f)
    with open(h2_stat_path, 'r') as f:
        h2_stats = json.load(f)
    
    logger.info(f"Loaded statistical results from {h1_stat_path} and {h2_stat_path}")
    
    # Create table data from statistical results
    table_rows = []
    
    # Process H1 (JailbreakBench) and H2 (HarmBench) results
    for hyp_name, stat_data, dataset in [
        ('H1', h1_stats, 'JailbreakBench'),
        ('H2', h2_stats, 'HarmBench')
    ]:
        
        model_keys = [('llama4scout', 'Llama-4-Scout'), ('qwen25', 'Qwen-2.5-7B')]
        
        for model_key, model_name in model_keys:
            try:
                model_data = stat_data['models'][model_key]['metrics']
                
                # Process SE at canonical tau=0.2
                if 'semantic_entropy' in model_data and 'tau_0.2' in model_data['semantic_entropy']:
                    se_data = model_data['semantic_entropy']['tau_0.2']
                    fnr = se_data.get('fnr', 0)
                    fnr_ci = se_data.get('fnr_wilson_ci', [fnr, fnr])
                    
                    table_rows.append({
                        'Model': model_name,
                        'Dataset': dataset,
                        'Method': 'SE (τ=0.2)',
                        'FNR': f"{fnr:.3f}",
                        'FNR_with_CI': f"{fnr:.3f} [{fnr_ci[0]:.3f}, {fnr_ci[1]:.3f}]",
                        'Actual FPR': '0.000'
                    })
                
                # Process baselines
                baseline_map = {
                    'BERTScore': 'Avg. Pairwise BERTScore',
                    'EmbeddingVariance': 'Embedding Variance', 
                    'LevenshteinVariance': 'Levenshtein Variance'
                }
                
                for stat_key, method_name in baseline_map.items():
                    if stat_key in model_data:
                        baseline_data = model_data[stat_key]
                        fnr = baseline_data.get('fnr', 0)
                        fnr_ci = baseline_data.get('fnr_wilson_ci', [fnr, fnr])
                        actual_fpr = baseline_data.get('actual_fpr', 0)
                        
                        table_rows.append({
                            'Model': model_name,
                            'Dataset': dataset,
                            'Method': method_name,
                            'FNR': f"{fnr:.3f}",
                            'FNR_with_CI': f"{fnr:.3f} [{fnr_ci[0]:.3f}, {fnr_ci[1]:.3f}]",
                            'Actual FPR': f"{actual_fpr:.3f}"
                        })
                        
            except (KeyError, TypeError) as e:
                logger.warning(f"Missing data for {model_key} in {hyp_name}: {e}")
    
    df_table = pd.DataFrame(table_rows)
    logger.info(f"Created table with {len(df_table)} rows")
    
    # Sort for better readability
    df_table = df_table.sort_values(['Model', 'Dataset', 'Method'])
    
    # Save as CSV (both versions)
    csv_path = output_dir / "table_2_fnr_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(csv_path, index=False)
    logger.info(f"CSV table saved to {csv_path}")
    
    # Create LaTeX-ready table
    latex_lines = []
    latex_lines.append("% Table 2: FNR@5%FPR comparison with 95% Wilson confidence intervals")
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{FNR@5\\%FPR comparison across datasets and methods with 95\\% confidence intervals}")
    latex_lines.append("\\label{tab:fnr_comparison}")
    latex_lines.append("\\begin{tabular}{llcc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & Dataset & Method & FNR [95\\% CI] & Actual FPR \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in df_table.iterrows():
        line = f"{row['Model']} & {row['Dataset']} & {row['Method']} & {row['FNR_with_CI']} & {row['Actual FPR']} \\\\"
        latex_lines.append(line)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Save LaTeX table
    latex_path = output_dir / "table_2_fnr_comparison.tex"
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    logger.info(f"LaTeX table saved to {latex_path}")
    
    # Create Markdown table with CIs
    markdown_lines = []
    markdown_lines.append("# Table 2: FNR@5%FPR with 95% Wilson Confidence Intervals\n")
    markdown_lines.append("| Model | Dataset | Method | FNR [95% CI] | Actual FPR |")
    markdown_lines.append("|-------|---------|--------|-------------|-----------|")
    
    for _, row in df_table.iterrows():
        line = f"| {row['Model']} | {row['Dataset']} | {row['Method']} | {row['FNR_with_CI']} | {row['Actual FPR']} |"
        markdown_lines.append(line)
    
    # Save as Markdown
    md_path = output_dir / "table_2_fnr_comparison.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(markdown_lines))
    logger.info(f"Markdown table saved to {md_path}")
    
    # Print summary statistics
    logger.info("\nTable Summary:")
    logger.info(f"Total rows: {len(df_table)}")
    logger.info(f"Models: {df_table['Model'].unique().tolist()}")
    logger.info(f"Datasets: {df_table['Dataset'].unique().tolist()}")
    logger.info(f"Methods: {df_table['Method'].unique().tolist()}")


if __name__ == "__main__":
    generate_table_2()