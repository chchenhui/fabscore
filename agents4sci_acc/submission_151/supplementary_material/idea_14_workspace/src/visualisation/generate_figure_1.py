"""
Generate Figure 1: AUROC Comparison on JailbreakBench.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
import sys

# Import plot utils
from visualisation.plot_utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_figure_1():
    """Generate AUROC comparison bar chart."""
    logger.info("Generating Figure 1: AUROC Comparison on JailbreakBench")
    
    # Auto-detect paths
    if Path("/research_storage/outputs/visualisation/temp/f1_data.csv").exists():
        # Modal environment
        data_path = Path("/research_storage/outputs/visualisation/temp/f1_data.csv")
        output_path = Path("/research_storage/outputs/figures/figure_1_auroc_comparison.png")
    else:
        # Local environment
        data_path = Path("idea_14_workspace/outputs/visualisation/temp/f1_data.csv")
        output_path = Path("idea_14_workspace/outputs/figures/figure_1_auroc_comparison.png")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Filter for methods we want to show
    methods_to_show = ['semantic_entropy', 'avg_pairwise_bertscore', 'embedding_variance']
    df_filtered = df[df['Method'].isin(methods_to_show)].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for grouped bar plot
    models = ['Llama-4-Scout', 'Qwen-2.5-7B']
    x = np.arange(len(models))
    width = 0.25
    
    # Get values for each method
    se_values = []
    se_tau_values = []
    bert_values = []
    emb_values = []
    
    for model in models:
        model_data = df_filtered[df_filtered['Model'] == model]
        
        se_data = model_data[model_data['Method'] == 'semantic_entropy']
        if len(se_data) > 0:
            se_values.append(se_data['Value'].values[0])
            tau_val = se_data['tau'].values[0]
            se_tau_values.append(f"τ={tau_val}" if not pd.isna(tau_val) else "")
        else:
            se_values.append(0)
            se_tau_values.append("")
        
        bert_val = model_data[model_data['Method'] == 'avg_pairwise_bertscore']['Value'].values
        bert_values.append(bert_val[0] if len(bert_val) > 0 else 0)
        
        emb_val = model_data[model_data['Method'] == 'embedding_variance']['Value'].values
        emb_values.append(emb_val[0] if len(emb_val) > 0 else 0)
    
    # Create bars
    bars1 = ax.bar(x - width, se_values, width, label='Semantic Entropy', 
                   color=get_color('semantic_entropy'))
    bars2 = ax.bar(x, bert_values, width, label='Avg. Pairwise BERTScore',
                   color=get_color('avg_pairwise_bertscore'))
    bars3 = ax.bar(x + width, emb_values, width, label='Embedding Variance',
                   color=get_color('embedding_variance'))
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10)
    
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Add tau annotations for SE bars
    for i, (bar, tau_label) in enumerate(zip(bars1, se_tau_values)):
        if tau_label and bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., 
                   bar.get_height() + 0.02,
                   tau_label, ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Baseline Methods Outperform Semantic Entropy on JailbreakBench (SE at Best τ)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.set_ylim([0.5, 0.85])
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 1 generation complete")


if __name__ == "__main__":
    generate_figure_1()