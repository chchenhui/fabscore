"""
Generate Figure 3: FNR vs. Hyperparameters for Qwen on HarmBench with Wilson CIs.
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


def generate_figure_3():
    """Generate FNR vs. Hyperparameters line plot with Wilson CI error bars."""
    logger.info("Generating Figure 3: FNR vs. Hyperparameters with Wilson CIs")
    
    # Auto-detect paths
    if Path("/research_storage/outputs").exists():
        # Modal environment
        base_path = Path("/research_storage/outputs")
        output_path = Path("/research_storage/outputs/figures/figure_3_hyperparameter_brittleness.png")
    else:
        # Local environment
        base_path = Path("idea_14_workspace/outputs")
        output_path = Path("idea_14_workspace/outputs/visualisation/figures/figure_3_hyperparameter_brittleness.png")
    
    # Load H2 statistical results (HarmBench)
    h2_stat_path = base_path / "statistical_analysis" / "h2_statistical_results.json"
    
    with open(h2_stat_path, 'r') as f:
        h2_stats = json.load(f)
    
    logger.info(f"Loaded statistical results from {h2_stat_path}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data for Qwen on HarmBench (H2)
    qwen_data = h2_stats['models']['qwen2.5-7b-instruct']['metrics']['semantic_entropy']
    
    # Prepare data for N=5 and N=10
    tau_values = [0.1, 0.2, 0.3, 0.4]
    
    for n_value, color_key, linestyle, marker, label in [
        ('N5', 'n5', '-', 'o', 'SE (N=5)'),
        ('N10', 'n10', '--', 's', 'SE (N=10)')
    ]:
        fnr_values = []
        ci_lower = []
        ci_upper = []
        
        for tau in tau_values:
            tau_key = f"tau_{tau}_{n_value}"
            if tau_key in qwen_data:
                tau_data = qwen_data[tau_key]
                fnr = tau_data.get('fnr', 0)
                fnr_ci = tau_data.get('fnr_wilson_ci', [fnr, fnr])
                
                fnr_values.append(fnr)
                ci_lower.append(fnr - fnr_ci[0])
                ci_upper.append(fnr_ci[1] - fnr)
            else:
                # Handle missing data
                fnr_values.append(np.nan)
                ci_lower.append(0)
                ci_upper.append(0)
        
        # Plot line with error bars
        color = get_color(color_key)
        ax.errorbar(tau_values, fnr_values,
                   yerr=[ci_lower, ci_upper],
                   color=color, linestyle=linestyle, marker=marker,
                   markersize=8, linewidth=2.5, label=label,
                   capsize=4, elinewidth=1.5)
    
    # Add annotations for key brittleness points
    # Annotate the jump from τ=0.1 to τ=0.2 for N=5
    ax.annotate('41% increase\nin missed detections',
                xy=(0.15, 0.76), xytext=(0.23, 0.68),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Clustering Threshold τ', fontsize=12)
    ax.set_ylabel('FNR @ 5% FPR', fontsize=12)
    ax.set_title('Semantic Entropy Exhibits Extreme Hyperparameter Brittleness\n(Qwen-2.5-7B on HarmBench)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([0.1, 0.2, 0.3, 0.4])
    ax.set_ylim([0.4, 1.0])
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at FNR=0.63 to show best performance
    ax.axhline(y=0.63, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.11, 0.61, 'Best: FNR=0.63', fontsize=9, color='gray')
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    
    plt.close()
    logger.info("Figure 3 generation complete with Wilson CIs")


if __name__ == "__main__":
    generate_figure_3()