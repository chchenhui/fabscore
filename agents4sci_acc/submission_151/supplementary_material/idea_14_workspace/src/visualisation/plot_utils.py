"""
Plot utilities module for consistent styling across all visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set consistent style for LaTeX template compliance
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Font settings to match LaTeX template (10pt base, Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 10

# Color palette
COLOR_PALETTE = {
    'llama': '#3498DB',  # Blue
    'qwen': '#E74C3C',   # Red
    'semantic_entropy': '#2ECC71',  # Green
    'baselines': '#7F7F7F',  # Grey
    'avg_pairwise_bertscore': '#95A5A6',  # Light grey
    'embedding_variance': '#34495E',  # Dark grey
    'levenshtein_variance': '#7F8C8D',  # Medium grey
    'consistency_confound': '#E74C3C',  # Red
    'other': '#BDC3C7',  # Light grey
    'benign': '#3498DB',  # Blue
    'harmful': '#E74C3C',  # Red
    'n5': '#2ECC71',  # Green
    'n10': '#F39C12',  # Orange
}

# Method name mapping
METHOD_NAMES = {
    'semantic_entropy': 'Semantic Entropy',
    'avg_pairwise_bertscore': 'Avg. Pairwise BERTScore',
    'embedding_variance': 'Embedding Variance',
    'levenshtein_variance': 'Levenshtein Variance',
    'semantic_entropy (τ=0.2)': 'SE (τ=0.2)',
    'semantic_entropy (best τ=0.1)': 'SE (best τ=0.1)',
    'semantic_entropy (best τ=0.3)': 'SE (best τ=0.3)',
    'semantic_entropy (best τ=0.4)': 'SE (best τ=0.4)',
}

# Model name mapping
MODEL_NAMES = {
    'llama': 'Llama-4-Scout',
    'qwen': 'Qwen-2.5-7B',
    'Llama-4-Scout': 'Llama-4-Scout',
    'Qwen-2.5-7B': 'Qwen-2.5-7B',
}

def get_color(key):
    """Get color for a given key."""
    # First check direct mapping
    if key in COLOR_PALETTE:
        return COLOR_PALETTE[key]
    
    # Check if it's a model
    if key.lower() in ['llama', 'llama-4-scout']:
        return COLOR_PALETTE['llama']
    elif key.lower() in ['qwen', 'qwen-2.5-7b']:
        return COLOR_PALETTE['qwen']
    
    # Check if it's a method
    if 'semantic_entropy' in key.lower():
        return COLOR_PALETTE['semantic_entropy']
    elif 'bertscore' in key.lower():
        return COLOR_PALETTE['avg_pairwise_bertscore']
    elif 'embedding' in key.lower():
        return COLOR_PALETTE['embedding_variance']
    elif 'levenshtein' in key.lower():
        return COLOR_PALETTE['levenshtein_variance']
    
    # Default to grey
    return COLOR_PALETTE['baselines']

def get_method_name(method):
    """Get human-readable method name."""
    if method in METHOD_NAMES:
        return METHOD_NAMES[method]
    return method

def get_model_name(model):
    """Get human-readable model name."""
    if model in MODEL_NAMES:
        return MODEL_NAMES[model]
    return model

def save_figure(fig, filename, bbox_inches='tight'):
    """Save figure with consistent settings."""
    fig.savefig(filename, bbox_inches=bbox_inches, dpi=300)
    logger.info(f"Figure saved to {filename}")

def create_figure(figsize=(12, 8)):
    """Create a new figure with default settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def format_percentage(value, decimals=1):
    """Format a value as a percentage."""
    return f"{value * 100:.{decimals}f}%"

def format_value(value, decimals=3):
    """Format a numeric value."""
    return f"{value:.{decimals}f}"