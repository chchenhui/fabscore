"""
Generate Figure 1H: AUROC Comparison on HarmBench.
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


def generate_figure_1h():
    """Generate AUROC comparison bar chart for HarmBench."""
    logger.info("Generating Figure 1H: AUROC Comparison on HarmBench")
    
    # Auto-detect paths
    if Path("/research_storage/outputs/visualisation/temp/f1h_data.csv").exists():
        # Modal environment
        data_path = Path("/research_storage/outputs/visualisation/temp/f1h_data.csv")
        output_path = Path("/research_storage/outputs/figures/figure_1h_auroc_harmbench.png")
    else:
        # Local environment
        data_path = Path("idea_14_workspace/outputs/visualisation/temp/f1h_data.csv")
        output_path = Path("idea_14_workspace/outputs/figures/figure_1h_auroc_harmbench.png")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    models = df['Model'].unique()
    methods = df['Method'].unique()
    
    x = np.arange(len(models))
    width = 0.2
    
    # Create bars for each method
    for i, method in enumerate(methods):
        method_data = df[df['Method'] == method]
        values = []
        tau_labels = []
        
        for model in models:
            model_method_data = method_data[method_data['Model'] == model]
            if len(model_method_data) > 0:
                values.append(model_method_data['Value'].iloc[0])
                tau_val = model_method_data['tau'].iloc[0]
                if pd.notna(tau_val):
                    tau_labels.append(f"τ={tau_val}")
                else:
                    tau_labels.append("")
            else:
                values.append(0)
                tau_labels.append("")
        
        # Get method color and label
        if method == 'semantic_entropy':
            color = get_color('semantic_entropy')
            label = f'Semantic Entropy (best τ)'
        elif method == 'avg_pairwise_bertscore':
            color = get_color('bertscore')
            label = 'BERTScore'
        elif method == 'embedding_variance':
            color = get_color('embedding_variance')
            label = 'Embedding Variance'
        elif method == 'levenshtein_variance':
            color = get_color('levenshtein')
            label = 'Levenshtein Variance'
        else:
            color = get_color('default')
            label = method
        
        bars = ax.bar(x + i*width, values, width, label=label, color=color)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Add tau annotations for Semantic Entropy
        if method == 'semantic_entropy':
            for j, (bar, tau_label) in enumerate(zip(bars, tau_labels)):
                if tau_label and bar.get_height() > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           bar.get_height() + 0.02,
                           tau_label, ha='center', va='bottom', 
                           fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('AUROC Comparison on HarmBench (SE at Best τ)', fontsize=14)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 1H generation complete")


if __name__ == "__main__":
    generate_figure_1h()