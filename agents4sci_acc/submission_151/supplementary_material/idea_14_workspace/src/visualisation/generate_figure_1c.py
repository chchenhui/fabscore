"""
Generate Figure 1C: Comprehensive AUROC across both datasets with confidence intervals.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
import sys
import json

# Import plot utils
from visualisation.plot_utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_figure_1c():
    """Generate comprehensive AUROC comparison across both models and datasets with CIs."""
    logger.info("Generating Figure 1C: Comprehensive AUROC across both datasets with error bars")
    
    # Auto-detect paths
    if Path("/research_storage/outputs").exists():
        # Modal environment
        base_path = Path("/research_storage/outputs")
        output_path = Path("/research_storage/outputs/figures/figure_1c_auroc_comprehensive.png")
    else:
        # Local environment
        base_path = Path("idea_14_workspace/outputs")
        output_path = Path("idea_14_workspace/outputs/visualisation/figures/figure_1c_auroc_comprehensive.png")
    
    # Load statistical results for both hypotheses
    h1_stat_path = base_path / "statistical_analysis" / "h1_statistical_results.json"
    h2_stat_path = base_path / "statistical_analysis" / "h2_statistical_results.json"
    
    with open(h1_stat_path, 'r') as f:
        h1_stats = json.load(f)
    with open(h2_stat_path, 'r') as f:
        h2_stats = json.load(f)
    
    logger.info(f"Loaded statistical results from {h1_stat_path} and {h2_stat_path}")
    
    # Create figure with subplots for each dataset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = ['JailbreakBench', 'HarmBench']
    stats_data = [h1_stats, h2_stats]
    axes = [ax1, ax2]
    
    for dataset, stat_data, ax in zip(datasets, stats_data, axes):
        # Extract data from statistical results
        model_keys = [('llama4scout', 'Llama-4-Scout'), ('qwen25', 'Qwen-2.5-7B')]
        method_keys = [
            ('semantic_entropy', 'Semantic Entropy', 'semantic_entropy'),
            ('BERTScore', 'BERTScore', 'bertscore'),
            ('EmbeddingVariance', 'Embedding Variance', 'embedding_variance'),
            ('LevenshteinVariance', 'Levenshtein Variance', 'levenshtein')
        ]
        
        x = np.arange(len(model_keys))
        width = 0.2
        
        # Process each method
        for i, (method_key, method_label, color_key) in enumerate(method_keys):
            values = []
            errors_lower = []
            errors_upper = []
            tau_labels = []
            
            for model_key, model_name in model_keys:
                try:
                    model_data = stat_data['models'][model_key]['metrics']
                    
                    if method_key == 'semantic_entropy':
                        # Find best tau for SE
                        best_auroc = 0
                        best_ci = None
                        best_tau = None
                        
                        for tau_key in ['tau_0.1', 'tau_0.2', 'tau_0.3', 'tau_0.4']:
                            if tau_key in model_data['semantic_entropy']:
                                tau_data = model_data['semantic_entropy'][tau_key]
                                if 'auroc' in tau_data and tau_data['auroc'] > best_auroc:
                                    best_auroc = tau_data['auroc']
                                    best_ci = tau_data.get('delong_ci', [best_auroc, best_auroc])
                                    best_tau = tau_key.replace('tau_', '')
                        
                        values.append(best_auroc)
                        if best_ci and len(best_ci) == 2:
                            errors_lower.append(best_auroc - best_ci[0])
                            errors_upper.append(best_ci[1] - best_auroc)
                        else:
                            errors_lower.append(0)
                            errors_upper.append(0)
                        tau_labels.append(f"τ={best_tau}" if best_tau else "")
                    else:
                        # Regular metrics
                        if method_key in model_data:
                            auroc = model_data[method_key].get('auroc', 0)
                            ci = model_data[method_key].get('delong_ci', [auroc, auroc])
                            values.append(auroc)
                            if ci and len(ci) == 2:
                                errors_lower.append(auroc - ci[0])
                                errors_upper.append(ci[1] - auroc)
                            else:
                                errors_lower.append(0)
                                errors_upper.append(0)
                        else:
                            values.append(0)
                            errors_lower.append(0)
                            errors_upper.append(0)
                        tau_labels.append("")
                        
                except (KeyError, TypeError) as e:
                    logger.warning(f"Missing data for {model_key} - {method_key}: {e}")
                    values.append(0)
                    errors_lower.append(0)
                    errors_upper.append(0)
                    tau_labels.append("")
            
            # Plot bars with error bars
            color = get_color(color_key)
            bars = ax.bar(x + i*width, values, width, 
                          yerr=[errors_lower, errors_upper],
                          capsize=3,
                          label=method_label if dataset == datasets[0] else "", 
                          color=color, ecolor='black', alpha=0.8)
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[j] + 0.01,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=9)
            
            # Add tau annotations for Semantic Entropy
            if method_key == 'semantic_entropy':
                for j, (bar, tau_label) in enumerate(zip(bars, tau_labels)):
                    if tau_label and bar.get_height() > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., 
                               bar.get_height() + errors_upper[j] + 0.035,
                               tau_label, ha='center', va='bottom', 
                               fontsize=8, fontweight='bold', color='red')
        
        # Customize subplot
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([name for _, name in model_keys], rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
    
    # Add overall title and legend
    fig.suptitle('AUROC Comparison Across Models and Datasets (SE at Best τ)', fontsize=14, y=1.02)
    ax1.legend(loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 1C generation complete")


if __name__ == "__main__":
    generate_figure_1c()