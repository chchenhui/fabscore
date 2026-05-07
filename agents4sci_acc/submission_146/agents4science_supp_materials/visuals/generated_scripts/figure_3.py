import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import glob
import os
import datetime
import sys

def find_latest_file(pattern):
    """Finds the most recently modified file matching a pattern."""
    files = glob.glob(pattern)
    if not files:
        print(f"Error: No files found matching pattern: {pattern}", file=sys.stderr)
        return None
    latest_file = max(files, key=os.path.getctime)
    print(f"Using latest data file: {latest_file}")
    return latest_file

def plot_panel(ax, data, color, title, x_col, y_col):
    """
    Generates a single scatter plot panel with a regression line.
    This function encapsulates the core logic from the original panel scripts.
    """
    if data.empty:
        print(f"Warning: No data to plot for '{title}'.")
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return

    # --- Data Preparation ---
    x_data = data[x_col]
    y_data = data[y_col]
    n_samples = len(data)
    print(f"Plotting panel '{title}' with {n_samples} data points.")

    # --- Plotting ---
    # Add jitter to x-coordinates to prevent overplotting
    jitter_amount = 0.1  # Changed from 0.15 to match original
    x_jittered = x_data + np.random.uniform(-jitter_amount, jitter_amount, len(x_data))

    # Create scatter plot
    ax.scatter(x_jittered, y_data, c=color, alpha=0.6, s=30, zorder=2,
               edgecolor='black', linewidth=0.5)
    print(f"Plotted {n_samples} individual data points with horizontal jitter.")

    # Calculate and plot trend line
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        line_x = np.array([x_data.min(), x_data.max()])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color=color, linestyle='--', alpha=0.9, zorder=3, linewidth=2)
        print(f"Regression for '{title}': R² = {r_value**2:.3f}, p-value = {p_value:.3f}")
        print(f"Added dashed regression line.")
    except Exception as e:
        print(f"Error in regression calculation for '{title}': {str(e)}", file=sys.stderr)

    # --- Styling ---
    ax.set_title(title, fontsize=11, fontweight='bold', fontfamily='Arial')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 5))
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=9, width=1.5, pad=5)
    plt.setp(ax.get_xticklabels(), fontweight='bold', fontfamily='Arial')
    plt.setp(ax.get_yticklabels(), fontweight='bold', fontfamily='Arial')

def main():
    """
    Main function to generate the complete 1x2 figure.
    """
    # --- Configuration ---
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    colors = {'Younger': '#2b7dd2', 'Older': '#ff9233'} # Azure blue, Warm amber

    # --- Data Loading ---
    age_group_file = find_latest_file('analysis_outputs/step5_age_group_data_*.csv')
    if age_group_file is None:
        return 1
    try:
        full_data = pd.read_csv(age_group_file)
        required_cols = ['age_group', 'education_level', 'percentile_range']
        if not all(col in full_data.columns for col in required_cols):
            print(f"Error: CSV is missing one or more required columns: {required_cols}", file=sys.stderr)
            return 1
        print("Age group data loaded successfully.")
        print(f"Shape: {full_data.shape}")
    except Exception as e:
        print(f"Error reading CSV file '{age_group_file}': {str(e)}", file=sys.stderr)
        return 1

    # --- Data Preparation ---
    data = full_data[required_cols].dropna().copy()
    younger_data = data[data['age_group'] == 'Younger']
    older_data = data[data['age_group'] == 'Older']

    # --- Data Validation ---
    print(f"Data validation - Total rows: {len(data)}")
    print(f"Younger adults: {len(younger_data)} rows")
    print(f"Older adults: {len(older_data)} rows")
    print(f"Expected approximately: Total ~1083, Younger ~530, Older ~383")

    if younger_data.empty or older_data.empty:
        print("Error: Data for one or both age groups is empty after cleaning.", file=sys.stderr)
        return 1

    # --- Figure Setup ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(8.27, 4.135),
        dpi=300,
        facecolor='white',
        sharey=True  # Share Y-axis for direct comparison
    )
    fig.set_facecolor('white')

    # --- Plot Panels ---
    plot_panel(ax1, younger_data, colors['Younger'], f"Younger Adults (N={len(younger_data)})", 'education_level', 'percentile_range')
    plot_panel(ax2, older_data, colors['Older'], f"Older Adults (N={len(older_data)})", 'education_level', 'percentile_range')

    # --- Final Figure-Level Adjustments ---
    # Set consistent axis limits for direct comparison
    x_min = data['education_level'].min() - 0.5
    x_max = data['education_level'].max() + 0.5
    y_min = 0
    y_max = 100
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    print(f"Applied consistent axis limits: X=({x_min:.1f}, {x_max:.1f}), Y=({y_min}, {y_max})")

    # Set shared axis labels
    fig.text(0.5, 0.02, 'Education Level', ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax1.set_ylabel('Percentile Range', fontsize=12, fontweight='bold', fontfamily='Arial', labelpad=10)
    ax2.set_ylabel('') # Remove redundant label

    # Adjust layout and add panel identifiers
    fig.tight_layout()
    plt.subplots_adjust(left=0.1, bottom=0.15, wspace=0.25) # Adjust for shared labels and spacing

    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    fig.text(pos1.x0 - 0.10, pos1.y1 + 0.08, 'A', fontsize=16, fontweight='bold', fontfamily='Arial', ha='left', va='top')
    fig.text(pos2.x0 - 0.08, pos2.y1 + 0.08, 'B', fontsize=16, fontweight='bold', fontfamily='Arial', ha='left', va='top')
    print("Added panel identifiers 'A' and 'B'.")

    # --- Save Figure ---
    output_dir = "figure_outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"figure_3_{timestamp}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Figure saved to: {output_filename}")
    print("Finished execution")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
