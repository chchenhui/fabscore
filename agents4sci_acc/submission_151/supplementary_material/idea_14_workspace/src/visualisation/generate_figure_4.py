"""
Generate Figure 4: Breakdown of False Negative Causes.
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


def generate_figure_4():
    """Generate stacked bar chart for false negative breakdown."""
    logger.info("Generating Figure 4: Breakdown of False Negative Causes")
    
    # Auto-detect paths
    if Path("/research_storage/outputs/visualisation/temp/f4_data.csv").exists():
        # Modal environment
        data_path = Path("/research_storage/outputs/visualisation/temp/f4_data.csv")
        output_path = Path("/research_storage/outputs/figures/figure_4_fn_breakdown.png")
    else:
        # Local environment
        data_path = Path("idea_14_workspace/outputs/visualisation/temp/f4_data.csv")
        output_path = Path("idea_14_workspace/outputs/figures/figure_4_fn_breakdown.png")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Pivot data for stacked bar chart
    df_pivot = df.pivot(index='Experiment', columns='Cause', values='Count')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked bar chart
    experiments = df_pivot.index
    x = np.arange(len(experiments))
    width = 0.5
    
    # Get counts for each cause
    confound_counts = df_pivot['Consistency Confound'].values
    other_counts = df_pivot['Other'].values
    
    # Create cleaner x-axis labels
    clean_labels = []
    for exp in experiments:
        if 'Llama' in exp:
            clean_labels.append('Llama\n(JailbreakBench)')
        elif 'Qwen' in exp:
            clean_labels.append('Qwen\n(HarmBench)')
        else:
            clean_labels.append(exp)
    
    # Create bars
    bars1 = ax.bar(x, confound_counts, width, label='Consistency Confound',
                   color=get_color('consistency_confound'))
    bars2 = ax.bar(x, other_counts, width, bottom=confound_counts,
                   label='Other', color=get_color('other'))
    
    # Add annotations
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Confound annotation
        total = confound_counts[i] + other_counts[i]
        if confound_counts[i] > 0:
            percentage = (confound_counts[i] / total) * 100
            ax.text(bar1.get_x() + bar1.get_width()/2., 
                   bar1.get_height()/2.,
                   f'{int(confound_counts[i])}\n({percentage:.1f}%)',
                   ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Other annotation
        if other_counts[i] > 0:
            percentage = (other_counts[i] / total) * 100
            ax.text(bar2.get_x() + bar2.get_width()/2.,
                   confound_counts[i] + other_counts[i]/2.,
                   f'{int(other_counts[i])}\n({percentage:.1f}%)',
                   ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Model and Dataset', fontsize=12)
    ax.set_ylabel('Count of False Negatives', fontsize=12)
    ax.set_title('Consistency Confound Accounts for Majority of False Negatives (τ=0.2)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(clean_labels)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 4 generation complete")


if __name__ == "__main__":
    generate_figure_4()