import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import glob
import os
import datetime
import sys

# --- Global Settings & Constants ---
# Consistent color palette for high-impact journal figures
COLOR_PALETTE_CATEGORICAL = ['#2b7dd2', '#ff9233', '#00c2bc', '#c2c2c2', '#97dba1', '#d62728', '#9467bd', '#8c564b']
PRIMARY_COLOR = COLOR_PALETTE_CATEGORICAL[0] # Blue for primary scatter plots
FONT_PROPS = {'family': 'Arial', 'weight': 'bold'}
plt.rcParams['font.family'] = 'Arial'

def find_latest_file(pattern):
    """Finds the most recently created file matching a glob pattern. Handles missing files gracefully."""
    files = glob.glob(pattern)
    if not files:
        print(f"Warning: No files found for pattern: {pattern}", file=sys.stderr)
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"Using data file: {latest_file}")
    return latest_file

def setup_axis_style(ax):
    """Applies consistent styling to an axis object."""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 0))
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(axis='both', which='major', labelsize=9, width=1.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), **FONT_PROPS)
    plt.setp(ax.get_yticklabels(), **FONT_PROPS)

def create_panel_a(ax, file_path):
    """Generates Panel A: Coefficient of Variation vs. Education Level."""
    df = pd.read_csv(file_path)
    required_cols = ['education_level', 'coefficient_of_variation']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Panel A data missing required columns: {required_cols}")
    # Use original data as-is (no dropna)
    if df.empty:
        raise ValueError("Panel A data is empty.")

    education_levels = sorted(df['education_level'].unique())
    data = [df[df['education_level'] == level]['coefficient_of_variation'].values for level in education_levels]
    
    boxes = ax.boxplot(data, notch=True, patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.5},
                       whiskerprops={'linewidth': 1}, capprops={'linewidth': 1},
                       positions=range(1, len(education_levels) + 1), showfliers=False)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, c=PRIMARY_COLOR, alpha=0.3, s=10, zorder=2, edgecolor='black', linewidth=0.5)

    for i, box in enumerate(boxes['boxes']):
        box.set_facecolor(PRIMARY_COLOR)
        box.set_alpha(0.8)
        box.set_edgecolor('black')
        box.set_linewidth(1)

    setup_axis_style(ax)
    ax.set_xlabel('Education Level', fontsize=10, labelpad=2, **FONT_PROPS)
    ax.set_ylabel('Coefficient of Variation', fontsize=10, labelpad=4, **FONT_PROPS)
    ax.set_xticks(range(1, len(education_levels) + 1))
    ax.set_xticklabels([str(int(level)) for level in education_levels])
    ax.set_ylim(0, 1) # As per specification
    
    print(f"Panel A: Plotted {len(df)} data points across {len(education_levels)} groups. Styling: boxplot with notch, patch_artist=True, median line (black, 1.5pt), whiskers/caps (1pt), scatter overlay (blue, alpha=0.3, s=10, black edge, 0.5pt), font Arial bold, axis grid dashed, white background.")
    return True

def create_panel_b(ax, file_path):
    """Generates Panel B: Percentile Range vs. Collapsed Education Groups."""
    df = pd.read_csv(file_path)
    required_cols = ['education_collapsed', 'percentile_range']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Panel B data missing required columns: {required_cols}")
    # Use original data as-is (no dropna)
    if df.empty:
        raise ValueError("Panel B data is empty.")

    categories = ['Low', 'Medium', 'High']
    df['education_collapsed'] = pd.Categorical(df['education_collapsed'], categories=categories, ordered=True)
    df = df.sort_values('education_collapsed')
    
    data = [df[df['education_collapsed'] == cat]['percentile_range'].values for cat in categories]

    boxes = ax.boxplot(data, notch=True, patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.5},
                       whiskerprops={'linewidth': 1}, capprops={'linewidth': 1},
                       positions=range(1, len(categories) + 1), showfliers=False)

    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, c=PRIMARY_COLOR, alpha=0.3, s=10, zorder=2, edgecolor='black', linewidth=0.5)

    for i, box in enumerate(boxes['boxes']):
        box.set_facecolor(PRIMARY_COLOR)
        box.set_alpha(0.8)
        box.set_edgecolor('black')
        box.set_linewidth(1)

    setup_axis_style(ax)
    ax.set_xlabel('Collapsed Education Groups', fontsize=10, labelpad=2, **FONT_PROPS)
    ax.set_ylabel('Percentile Range', fontsize=10, labelpad=4, **FONT_PROPS)
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100) # As per specification

    print(f"Panel B: Plotted {len(df)} data points across {len(categories)} groups. Styling: boxplot with notch, patch_artist=True, median line (black, 1.5pt), whiskers/caps (1pt), scatter overlay (blue, alpha=0.3, s=10, black edge, 0.5pt), font Arial bold, axis grid dashed, white background.")
    return True

def create_panel_c(ax, df):
    """Generates Panel C: Split-Half Reliability for Range metric."""
    required_cols = ['odd_half_range', 'even_half_range']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Panel C data missing required columns: {required_cols}")
    
    x_data = df['odd_half_range'].values
    y_data = df['even_half_range'].values
    
    correlation, p_value = stats.pearsonr(x_data, y_data)
    print(f"Panel C (Range): r = {correlation:.3f}, p = {p_value:.3f}. Styling: scatter (blue, alpha=0.6, s=30, black edge, 0.5pt), regression line (blue dashed, 2pt), identity line (gray, 1pt), font Arial bold, axis grid dashed, white background.")

    ax.scatter(x_data, y_data, c=PRIMARY_COLOR, alpha=0.6, s=30, zorder=2, edgecolor='black', linewidth=0.5)
    
    slope, intercept, _, _, _ = stats.linregress(x_data, y_data)
    line_x = np.array([np.min(x_data), np.max(x_data)])
    ax.plot(line_x, slope * line_x + intercept, color=PRIMARY_COLOR, linestyle='--', alpha=0.8, linewidth=2, zorder=1)
    
    min_val = min(np.min(x_data), np.min(y_data))
    max_val = max(np.max(x_data), np.max(y_data))
    ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='-', alpha=0.5, linewidth=1, zorder=0)
    
    setup_axis_style(ax)
    ax.set_xlabel('Odd Half Range', fontsize=10, labelpad=2, **FONT_PROPS)
    ax.set_ylabel('Even Half Range', fontsize=10, labelpad=4, **FONT_PROPS)
    # Set axis limits based on data range for panel C
    padding = (max_val - min_val) * 0.05
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    return True

def create_panel_d(ax, df):
    """Generates Panel D: Split-Half Reliability for IQR metric."""
    required_cols = ['odd_half_iqr', 'even_half_iqr']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Panel D data missing required columns: {required_cols}")
        
    x_data = df['odd_half_iqr'].values
    y_data = df['even_half_iqr'].values
    
    correlation, p_value = stats.pearsonr(x_data, y_data)
    print(f"Panel D (IQR): r = {correlation:.3f}, p = {p_value:.3f}. Styling: scatter (blue, alpha=0.6, s=30, black edge, 0.5pt), identity line (gray, 1pt), font Arial bold, axis grid dashed, white background.")

    ax.scatter(x_data, y_data, c=PRIMARY_COLOR, alpha=0.6, s=30, zorder=2, edgecolor='black', linewidth=0.5)
    
    min_val = min(np.min(x_data), np.min(y_data))
    max_val = max(np.max(x_data), np.max(y_data))
    ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='-', alpha=0.5, linewidth=1, zorder=0)
    
    setup_axis_style(ax)
    ax.set_xlabel('Odd Half IQR', fontsize=10, labelpad=2, **FONT_PROPS)
    ax.set_ylabel('Even Half IQR', fontsize=10, labelpad=4, **FONT_PROPS)
    # Set axis limits based on data range for panel D
    padding = (max_val - min_val) * 0.05
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    return True

def main():
    """Main function to generate the complete 2x2 figure."""
    output_dir = "figure_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # --- Find Data Files ---
    file_a = find_latest_file('analysis_outputs/step6_sensitivity_coefficient_variation_*.csv')
    file_b = find_latest_file('analysis_outputs/step6_sensitivity_collapsed_education_*.csv')
    file_c_d = find_latest_file('analysis_outputs/step6_split_half_reliability_*.csv')
    
    if not all([file_a, file_b, file_c_d]):
        raise FileNotFoundError("One or more required data files could not be found.")

    # --- Create Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 8.27), dpi=300, facecolor='white')
    
    # --- Generate Panels ---
    create_panel_a(axes[0, 0], file_a)
    create_panel_b(axes[0, 1], file_b)
    
    # Load data for C and D once
    df_c_d = pd.read_csv(file_c_d)
    if df_c_d.empty:
        raise ValueError("Split-half reliability data is empty.")

    create_panel_c(axes[1, 0], df_c_d)
    create_panel_d(axes[1, 1], df_c_d)

    # --- Consistency and Final Touches ---
    # Add panel labels (A, B, C, D) using position-based approach
    panel_labels = ['A', 'B', 'C', 'D']
    for i, (ax, label) in enumerate(zip(axes.flatten(), panel_labels)):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.05, pos.y1 + 0.05, label, 
                 fontsize=16, fontweight='bold', va='top', ha='right')

    # Adjust layout
    fig.tight_layout(pad=3.0)
    fig.subplots_adjust(wspace=0.35, hspace=0.3)

    # --- Save Figure ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"figure_4_{timestamp}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\nFigure saved successfully to: {output_filename}")
    return 0

if __name__ == "__main__":
    try:
        return_code = main()
        if return_code == 0:
            print("Finished execution")
        sys.exit(return_code)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)
