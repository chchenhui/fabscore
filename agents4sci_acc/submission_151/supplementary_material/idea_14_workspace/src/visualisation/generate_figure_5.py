"""
Generate Figure 5: Paraphrase Impact Comparison (JBB vs JBB Paraphrase).
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


def generate_figure_5():
    """Generate paraphrase impact comparison showing FNR change."""
    logger.info("Generating Figure 5: Paraphrase Impact Comparison")
    
    # Auto-detect paths
    if Path("/research_storage/outputs/visualisation/temp/f5_data.csv").exists():
        # Modal environment
        data_path = Path("/research_storage/outputs/visualisation/temp/f5_data.csv")
        output_path = Path("/research_storage/outputs/figures/figure_5_paraphrase_impact.png")
    else:
        # Local environment
        data_path = Path("idea_14_workspace/outputs/visualisation/temp/f5_data.csv")
        output_path = Path("idea_14_workspace/outputs/figures/figure_5_paraphrase_impact.png")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path}")
    
    # Create figure with subplots for each model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = df['Model'].unique()
    axes = [ax1, ax2]
    
    for model, ax in zip(models, axes):
        model_data = df[df['Model'] == model]
        
        # Prepare data for bar chart
        methods = model_data['Method'].tolist()
        fnr_changes = model_data['FNR_Change_pp'].tolist()
        
        # Create color map for methods
        colors = []
        for method in methods:
            if 'Semantic Entropy' in method:
                if 'τ=0.1' in method:
                    colors.append(get_color('semantic_entropy'))
                else:
                    colors.append(get_color('semantic_entropy_alt'))
            elif 'BERTScore' in method:
                colors.append(get_color('bertscore'))
            elif 'Embedding Variance' in method:
                colors.append(get_color('embedding_variance'))
            elif 'Levenshtein' in method:
                colors.append(get_color('levenshtein'))
            else:
                colors.append(get_color('default'))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(methods))
        bars = ax.barh(y_pos, fnr_changes, color=colors)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, fnr_changes)):
            # Position label at end of bar
            x_pos = value + (0.5 if value >= 0 else -0.5)
            ax.text(x_pos, bar.get_y() + bar.get_height()/2.,
                   f'{value:.1f}pp',
                   ha='left' if value >= 0 else 'right', 
                   va='center', fontsize=10, fontweight='bold')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
        
        # Customize subplot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel('FNR Change (percentage points)', fontsize=11)
        ax.set_title(f'{model}', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set reasonable x-axis limits
        max_abs_change = max(abs(min(fnr_changes)), abs(max(fnr_changes)))
        ax.set_xlim([-max_abs_change * 1.3, max_abs_change * 1.3])
    
    # Add overall title
    fig.suptitle('Paraphrase Impact on Detection Methods: FNR Change (JBB → JBB Paraphrase)', 
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 5 generation complete")


if __name__ == "__main__":
    generate_figure_5()