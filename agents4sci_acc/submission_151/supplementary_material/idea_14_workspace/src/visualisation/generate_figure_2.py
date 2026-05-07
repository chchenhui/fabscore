"""
Generate Figure 2: SE vs. Response Length scatter plot.
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


def generate_figure_2():
    """Generate SE vs. Response Length scatter plot."""
    logger.info("Generating Figure 2: SE vs. Response Length")
    
    # Auto-detect paths
    if Path("/workspace/outputs/h3/per_prompt_analysis/llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl").exists():
        # Modal environment
        data_path = Path("/workspace/outputs/h3/per_prompt_analysis/llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl")
        output_path = Path("/research_storage/outputs/figures/figure_2_se_vs_length.png")
    else:
        # Local environment
        data_path = Path("idea_14_workspace/outputs/h3/per_prompt_analysis/llama-4-scout-17b-16e-instruct_H2_h3_prompt_analysis.jsonl")
        output_path = Path("idea_14_workspace/outputs/figures/figure_2_se_vs_length.png")
    
    # Read JSONL file
    import json
    records = []
    with open(data_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Map labels to readable names
    df['Label'] = df['label'].map({0: 'Benign', 1: 'Harmful'})
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with different colors for labels
    for label, color in [('Benign', get_color('benign')), ('Harmful', get_color('harmful'))]:
        mask = df['Label'] == label
        ax.scatter(df[mask]['log_length'], 
                  df[mask]['original_se_tau_0.1'],
                  c=color, label=label, alpha=0.6, s=50)
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['log_length'], df['original_se_tau_0.1'])
    line_x = np.array([df['log_length'].min(), df['log_length'].max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'k--', alpha=0.5, linewidth=2, 
           label=f'Regression (R²={r_value**2:.3f})')
    
    # Customize plot
    ax.set_xlabel('log(Median Response Length)', fontsize=12)
    ax.set_ylabel('SE Score (τ=0.1)', fontsize=12)
    ax.set_title('Semantic Entropy Shows Weak Correlation with Response Length (Llama on HarmBench)', 
                fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 2 generation complete")


if __name__ == "__main__":
    generate_figure_2()