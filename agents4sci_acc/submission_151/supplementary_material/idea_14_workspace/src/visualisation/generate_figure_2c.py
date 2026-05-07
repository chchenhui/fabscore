"""
Generate Figure 2C: 2x2 grid of SE vs Response Length for both models on HarmBench.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from pathlib import Path
import sys
from scipy import stats

# Import plot utils
from visualisation.plot_utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_figure_2c():
    """Generate 2x2 grid showing SE vs Response Length correlations."""
    logger.info("Generating Figure 2C: 2x2 SE vs Response Length Grid")
    
    # Auto-detect paths
    if Path("/research_storage/outputs/visualisation/temp/f2c_data.csv").exists():
        # Modal environment
        data_path = Path("/research_storage/outputs/visualisation/temp/f2c_data.csv")
        output_path = Path("/research_storage/outputs/figures/figure_2c_se_length_grid.png")
    else:
        # Local environment
        data_path = Path("idea_14_workspace/outputs/visualisation/temp/f2c_data.csv")
        output_path = Path("idea_14_workspace/outputs/figures/figure_2c_se_length_grid.png")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Map labels to readable names
    df['Label'] = df['label'].map({0: 'Benign', 1: 'Harmful'})
    
    # Debug: Check data contents
    logger.info(f"Unique Models in data: {df['Model'].unique()}")
    logger.info(f"Unique tau values in data: {df['tau'].unique()}")
    logger.info(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    
    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models = ['Llama-4-Scout', 'Qwen-2.5-7B']
    taus = [0.1, 0.2]  # Use numeric values instead of strings
    
    for i, model in enumerate(models):
        for j, tau in enumerate(taus):
            ax = axes[i, j]
            
            # Filter data for this subplot - convert tau to float for comparison
            subplot_data = df[(df['Model'] == model) & (df['tau'] == tau)]
            logger.info(f"Subplot {model}, τ={tau}: {len(subplot_data)} points")
            
            # Create scatter plot with different colors for labels
            for label, color in [('Benign', get_color('benign')), ('Harmful', get_color('harmful'))]:
                mask = subplot_data['Label'] == label
                subset = subplot_data[mask]
                if len(subset) > 0:
                    logger.debug(f"  Plotting {len(subset)} {label} points")
                    ax.scatter(subset['log_length'], 
                              subset['se_score'],
                              c=color, label=label, alpha=0.6, s=30)
                else:
                    logger.warning(f"  No {label} points to plot for {model}, τ={tau}")
            
            # Add regression line
            if len(subplot_data) > 1:  # Need at least 2 points for regression
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        subplot_data['log_length'], 
                        subplot_data['se_score']
                    )
                    line_x = np.array([subplot_data['log_length'].min(), subplot_data['log_length'].max()])
                    line_y = slope * line_x + intercept
                    ax.plot(line_x, line_y, 'k--', alpha=0.5, linewidth=1.5, 
                           label=f'R²={r_value**2:.3f}')
                    logger.debug(f"  Added regression line with R²={r_value**2:.3f}")
                except Exception as e:
                    logger.warning(f"  Could not add regression line: {e}")
            elif len(subplot_data) == 1:
                logger.warning(f"  Only 1 data point - skipping regression line")
            
            # Customize subplot
            ax.set_xlabel('log(Median Response Length)', fontsize=10)
            ax.set_ylabel(f'SE Score (τ={tau})', fontsize=10)
            ax.set_title(f'{model}, τ={tau}', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Set consistent axis limits for comparison
            ax.set_xlim([3.5, 8.8])
            ax.set_ylim([-0.1, 2.1])
    
    # Add overall title
    fig.suptitle('Semantic Entropy vs Response Length on HarmBench: Weak Correlation Across Models and τ Values', 
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 2C generation complete")


if __name__ == "__main__":
    generate_figure_2c()