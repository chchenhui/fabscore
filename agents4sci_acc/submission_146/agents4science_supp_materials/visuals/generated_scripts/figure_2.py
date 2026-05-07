import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib import patheffects as path_effects
from scipy import stats
import glob
import os
import datetime
import sys
import traceback

def find_latest_file(base_path, pattern):
    """Finds the most recent file matching a pattern in a directory."""
    files = glob.glob(os.path.join(base_path, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)

def create_panel_a(ax, heterogeneity_data, validation_data, colors, metric, color, y_label):
    """
    Creates a Panel A scatter plot for Grand Index vs a specified heterogeneity metric.
    """
    print(f"\n--- Creating Panel A: Grand Index vs {metric} ---")
    # Extract required columns and remove NaN values
    print("Extracting and cleaning data for Panel A...")
    required_het_columns = ['grand_index', metric]
    if not all(col in heterogeneity_data.columns for col in required_het_columns):
        print(f"Missing required columns in heterogeneity data for Panel A ({metric}).")
        ax.text(0.5, 0.5, 'Data missing', ha='center', va='center', transform=ax.transAxes)
        return

    valid_indices = heterogeneity_data[required_het_columns].dropna().index
    x_data = heterogeneity_data.loc[valid_indices, 'grand_index']
    y_data = heterogeneity_data.loc[valid_indices, metric]

    print(f"Valid data points: {len(x_data)}")
    print(f"X data range: {x_data.min():.3f} to {x_data.max():.3f}")
    print(f"Y data range: {y_data.min():.3f} to {y_data.max():.3f}")

    # Get correlation statistics from validation data
    correlation_coeff = np.nan
    correlation_p = np.nan
    if 'metric_name' in validation_data.columns:
        metric_stats = validation_data[validation_data['metric_name'] == metric]
        if not metric_stats.empty:
            correlation_coeff = metric_stats['correlation_with_grand_index'].iloc[0]
            correlation_p = metric_stats['correlation_p_value'].iloc[0]
            print(f"Correlation coefficient: {correlation_coeff}")
            print(f"Correlation p-value: {correlation_p}")
        else:
            print(f"No {metric} statistics found in validation data")
    else:
        print("metric_name column not found in validation data")

    # Create scatter plot
    scatter = ax.scatter(x_data, y_data, c=color, alpha=0.6, s=30, zorder=2,
               edgecolor='black', linewidth=0.5)
    print(f"Plotted {len(x_data)} data points.")

    # Calculate and plot trend line if correlation is significant
    if not np.isnan(correlation_p) and correlation_p < 0.05:
        print("Adding trend line (significant correlation)")
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        line_x = np.array([min(x_data), max(x_data)])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color=color, linestyle='--', alpha=0.7, zorder=1, linewidth=1.5)
    else:
        print("No trend line added (correlation not significant or missing)")

    # Set axis limits with 10% padding around data points
    y_range = max(y_data) - min(y_data)
    y_padding = 0.1 * y_range
    ax.set_ylim(min(y_data) - y_padding, max(y_data) + y_padding)
    
    x_range = max(x_data) - min(x_data)
    x_padding = 0.1 * x_range
    ax.set_xlim(min(x_data) - x_padding, max(x_data) + x_padding)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 0))
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Customize axis labels - no x-label for individual panels
    ax.set_ylabel(y_label, fontsize=13, fontfamily='Arial', labelpad=4, fontweight='bold')

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=9, width=1.5)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Add subtle grid only for y-axis
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    print("--- Panel A Complete ---")

def create_panel_b(ax, heterogeneity_data, validation_data, colors, metric, color, y_label):
    """
    Creates a Panel B scatter plot for Grand Index vs a specified heterogeneity metric.
    """
    print(f"\n--- Creating Panel B: Grand Index vs {metric} ---")
    # Extract required columns and remove NaN values
    print("Extracting and cleaning data for Panel B...")
    required_het_columns = ['grand_index', metric]
    if not all(col in heterogeneity_data.columns for col in required_het_columns):
        print(f"Missing required columns in heterogeneity data for Panel B ({metric}).")
        ax.text(0.5, 0.5, 'Data missing', ha='center', va='center', transform=ax.transAxes)
        return

    valid_indices = heterogeneity_data[required_het_columns].dropna().index
    x_data = heterogeneity_data.loc[valid_indices, 'grand_index']
    y_data = heterogeneity_data.loc[valid_indices, metric]

    print(f"Valid data points: {len(x_data)}")
    print(f"X data range: {x_data.min():.3f} to {x_data.max():.3f}")
    print(f"Y data range: {y_data.min():.3f} to {y_data.max():.3f}")

    # Get correlation statistics from validation data
    correlation_coeff = np.nan
    correlation_p = np.nan
    if 'metric_name' in validation_data.columns:
        metric_stats = validation_data[validation_data['metric_name'] == metric]
        if not metric_stats.empty:
            correlation_coeff = metric_stats['correlation_with_grand_index'].iloc[0]
            correlation_p = metric_stats['correlation_p_value'].iloc[0]
            print(f"Correlation coefficient: {correlation_coeff}")
            print(f"Correlation p-value: {correlation_p}")
        else:
            print(f"No {metric} statistics found in validation data")
    else:
        print("metric_name column not found in validation data")

    # Create scatter plot
    scatter = ax.scatter(x_data, y_data, c=color, alpha=0.6, s=30, zorder=2,
               edgecolor='black', linewidth=0.5)
    print(f"Plotted {len(x_data)} data points.")

    # Calculate and plot trend line if correlation is significant
    if not np.isnan(correlation_p) and correlation_p < 0.05:
        print("Adding trend line (significant correlation)")
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        line_x = np.array([min(x_data), max(x_data)])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, color=color, linestyle='--', alpha=0.7, zorder=1, linewidth=1.5)
    else:
        print("No trend line added (correlation not significant or missing)")

    # Set axis limits with 10% padding around data points
    y_range = max(y_data) - min(y_data)
    y_padding = 0.1 * y_range
    ax.set_ylim(min(y_data) - y_padding, max(y_data) + y_padding)
    
    x_range = max(x_data) - min(x_data)
    x_padding = 0.1 * x_range
    ax.set_xlim(min(x_data) - x_padding, max(x_data) + x_padding)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 0))
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Customize axis labels - no x-label for individual panels
    ax.set_ylabel(y_label, fontsize=13, fontfamily='Arial', labelpad=4, fontweight='bold')

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=9, width=1.5)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Add subtle grid only for y-axis
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    print("--- Panel B Complete ---")

def create_panel_c(ax, hetero_data, colors):
    """
    Creates Panel C histogram on a given matplotlib axis object.
    """
    print("\n--- Creating Panel C ---")
    # Extract the required columns and remove NaN values
    print("Processing heterogeneity metrics for Panel C...")
    required_columns = ['percentile_range', 'percentile_iqr']
    if not all(col in hetero_data.columns for col in required_columns):
        print("Missing required columns in heterogeneity data for Panel C.")
        ax.text(0.5, 0.5, 'Data missing', ha='center', va='center', transform=ax.transAxes)
        return

    percentile_range = hetero_data['percentile_range'].dropna()
    percentile_iqr = hetero_data['percentile_iqr'].dropna()

    print(f"Percentile range data points: {len(percentile_range)}")
    print(f"Percentile IQR data points: {len(percentile_iqr)}")

    # Get descriptive statistics
    range_mean, range_std = percentile_range.mean(), percentile_range.std()
    iqr_mean, iqr_std = percentile_iqr.mean(), percentile_iqr.std()
    print(f"Percentile range - Mean: {range_mean:.2f}, SD: {range_std:.2f}")
    print(f"Percentile IQR - Mean: {iqr_mean:.2f}, SD: {iqr_std:.2f}")

    # Create side-by-side bar histograms to avoid overlap
    print("Plotting side-by-side bar histograms for Percentile Range and Percentile IQR...")
    bins = np.linspace(0, 100, 25)
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + bin_width / 2

    hist_range, _ = np.histogram(percentile_range, bins=bins)
    hist_iqr, _ = np.histogram(percentile_iqr, bins=bins)

    bar_width = bin_width * 0.4
    offset = bar_width / 2

    ax.bar(bin_centers - offset, hist_range, width=bar_width, color=colors[0], alpha=0.8, 
           label='Percentile Range', edgecolor='black', linewidth=0.7, zorder=2)
    ax.bar(bin_centers + offset, hist_iqr, width=bar_width, color=colors[1], alpha=0.8, 
           label='Percentile IQR', edgecolor='black', linewidth=0.7, zorder=2)

    # Set axis limits for consistency
    ax.set_xlim(0, 100)
    max_freq = max(np.max(hist_range), np.max(hist_iqr))
    ax.set_ylim(0, max_freq * 1.2)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 0))
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Customize axis labels
    ax.set_xlabel('Heterogeneity Metric Value', fontsize=12, fontfamily='Arial',
                  labelpad=2, fontweight='bold')
    ax.set_ylabel('Number of Participants', fontsize=13, fontfamily='Arial',
                  labelpad=4, fontweight='bold')

    # Customize tick labels
    ax.tick_params(axis='both', which='major', labelsize=9, width=1.5)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # Add subtle grid only for y-axis
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Add legend with best positioning
    ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=False, 
              framealpha=0.8, edgecolor='gray')

    print("--- Panel C Complete ---")

def main():
    """
    Main function to generate the complete Figure 2.
    """
    try:
        print("Starting Figure 2 generation...")
        base_path = 'analysis_outputs/'
        print(f"Looking for data files in: {base_path}")

        # Find latest data files
        heterogeneity_file = find_latest_file(base_path, 'step3_heterogeneity_metrics_*.csv')
        validation_file = find_latest_file(base_path, 'step3_validation_results_*.csv')

        if not heterogeneity_file or not validation_file:
            print("Error: Missing one or more required data files.")
            print(f"Heterogeneity file found: {heterogeneity_file}")
            print(f"Validation file found: {validation_file}")
            return 1

        print(f"Found heterogeneity metrics file: {heterogeneity_file}")
        print(f"Found validation results file: {validation_file}")

        # Load data
        heterogeneity_data = pd.read_csv(heterogeneity_file)
        validation_data = pd.read_csv(validation_file)
        print("\nHeterogeneity data loaded successfully")
        print("Heterogeneity data shape:", heterogeneity_data.shape)
        print("\nValidation data loaded successfully")
        print("Validation data shape:", validation_data.shape)

        # --- Figure Setup ---
        plt.style.use('default')
        plt.rcParams['font.family'] = 'Arial'
        colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']  # blue, red, green, purple, brown
        fig, axes = plt.subplots(1, 3, figsize=(8.27, 4.135), dpi=300, facecolor='white')
        fig.set_facecolor('white')
        ax_a = axes[0]
        ax_b = axes[1]
        ax_c = axes[2]

        # --- Create Panels ---
        # Panel A: Grand Index vs Percentile Range
        create_panel_a(
            ax_a,
            heterogeneity_data,
            validation_data,
            colors,
            metric='percentile_range',
            color=colors[0],
            y_label='Percentile Range'
        )
        
        # Panel B: Grand Index vs Percentile IQR
        create_panel_b(
            ax_b,
            heterogeneity_data,
            validation_data,
            colors,
            metric='percentile_iqr',
            color=colors[1],
            y_label='Percentile IQR'
        )
        
        # Panel C: Histogram
        create_panel_c(ax_c, heterogeneity_data, colors)

        # --- Final Touches and Layout ---
        # Adjust layout to prevent overlap and add spacing
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.subplots_adjust(wspace=0.32, hspace=0.1)

        # Add shared x-axis label for panels A and B
        fig.text(0.35, 0.02, 'Grand Index (Composite Score)', fontsize=12, fontfamily='Arial',
                fontweight='bold', ha='center', va='bottom')

        # Add panel identifiers slightly above and to the left of the top of the y-axis
        for i, ax in enumerate([ax_a, ax_b, ax_c]):
            pos = ax.get_position()
            # Position labels slightly above and to the left of the y-axis top
            label_x = pos.x0 - 0.02
            label_y = pos.y1 + 0.02
            fig.text(label_x, label_y, chr(65 + i), fontsize=14, fontweight='bold', 
                    ha='right', va='bottom')

        # --- Save Figure ---
        output_dir = "figure_outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_dir, f"figure_2_{timestamp}.png")

        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"\nFigure saved successfully to: {output_filename}")
        print("Finished execution")
        return 0

    except Exception as e:
        print(f"An error occurred during figure generation: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())