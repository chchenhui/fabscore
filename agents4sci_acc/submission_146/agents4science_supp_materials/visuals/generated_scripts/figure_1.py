import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import datetime
import sys

def find_latest_file(pattern):
    """Finds the most recently modified file matching a pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def plot_panel(ax, df, y_col, panel_label, colors, education_order, education_labels):
    """
    Generates a single panel (box plot with scatter overlay) for the figure.
    This function is a modular version of the provided individual panel scripts.
    """
    # 1. Prepare data for the plot
    # Only include levels present in the data, maintaining the specified order
    education_levels_ordered = [level for level in education_order if level in df['education_level'].unique()]
    
    data = []
    sample_sizes = []
    
    for level in education_levels_ordered:
        level_data = df[df['education_level'] == level][y_col].dropna().values
        data.append(level_data)
        sample_sizes.append(len(level_data))
        print(f"Panel {panel_label}: Education level {level} (n={len(level_data)})")

    categories = [education_labels.get(level, f'Level {level}') for level in education_levels_ordered]

    # Assess signal-to-noise ratio for individual datapoints overlay
    if len(data) > 0:
        # Calculate means for each category
        means = [np.mean(d) for d in data if len(d) > 0]
        if len(means) > 1:
            mean_range = max(means) - min(means)
            # Calculate overall data range
            all_data_flat = np.concatenate([d for d in data if len(d) > 0])
            if len(all_data_flat) > 0:
                data_range = np.max(all_data_flat) - np.min(all_data_flat)
                signal_to_noise_ratio = mean_range / data_range if data_range > 0 else 0
                show_individual_points = signal_to_noise_ratio >= 0.05  # 1:20 threshold
            else:
                show_individual_points = False
        else:
            show_individual_points = True
    else:
        show_individual_points = False

    # 2. Create box plot
    boxes = ax.boxplot(data, 
                       notch=True,
                       patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.5},
                       flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 3, 'alpha': 0.5},
                       whiskerprops={'linewidth': 1.5},
                       capprops={'linewidth': 1.5},
                       positions=range(1, len(categories) + 1),
                       showfliers=False)

    print(f"Panel {panel_label}: Box plots show median, quartiles, and confidence intervals around medians")

    # 3. Add jittered scatter points for individual data points (if signal-to-noise ratio is adequate)
    if show_individual_points:
        for i, d in enumerate(data):
            if len(d) > 0:
                x = np.random.normal(i + 1, 0.04, size=len(d))
                ax.scatter(x, d, c=colors[i % len(colors)], alpha=0.5, s=10, zorder=2, 
                          edgecolor='black', linewidth=0.2)
        print(f"Panel {panel_label}: Individual data points included (signal-to-noise ratio: {signal_to_noise_ratio:.3f})")
    else:
        print(f"Panel {panel_label}: Individual data points omitted due to low signal-to-noise ratio")

    print(f"Panel {panel_label}: Total sample sizes - {dict(zip([education_labels.get(level, f'Level {level}') for level in education_levels_ordered], sample_sizes))}")

    # 4. Style the box plot
    for i, box in enumerate(boxes['boxes']):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.8)
        box.set_edgecolor('black')
        box.set_linewidth(1.5)

    # 5. Set axis limits and labels
    ax.set_xlim(0.5, len(categories) + 0.5)
    
    # Set y-axis label with proper subscript encoding
    if y_col == 'percentile_range':
        y_label = 'Percentile Range'
    elif y_col == 'percentile_iqr':
        y_label = 'Percentile IQR'
    else:
        y_label = y_col.replace('_', ' ').title()
    
    ax.set_ylabel(y_label, fontsize=14, fontfamily='Arial', labelpad=10, fontweight='bold')
    ax.set_xticks(range(1, len(categories) + 1))
    
    # Always show x-axis labels for both panels to fix the missing labels issue
    labels_with_n = [f"{cat}\n(n={n})" for cat, n in zip(categories, sample_sizes)]
    ax.set_xticklabels(labels_with_n, rotation=45, ha='right', fontsize=11, fontweight='bold')
    
    # Only show x-axis title for bottom panel
    if panel_label == 'B':
        ax.set_xlabel('Education Level', fontsize=14, fontfamily='Arial', labelpad=15, fontweight='bold')
    else:
        ax.set_xlabel('')

    # 6. Style spines and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, pad=5)
    plt.setp(ax.get_yticklabels(), fontweight='bold')

    # 7. Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # 8. Set dynamic y-axis limits with 10% padding above and below data range
    if len(data) > 0:
        all_data_flat = np.concatenate([d for d in data if len(d) > 0])
        if len(all_data_flat) > 0:
            y_min = np.min(all_data_flat)
            y_max = np.max(all_data_flat)
            y_range = y_max - y_min
            
            # 10% padding above and below
            padding = 0.1 * y_range if y_range > 0 else 0.1 * y_max
            ax.set_ylim(max(0, y_min - padding), y_max + padding)
        else:
            ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, 1)

def main():
    """
    Main function to generate the complete 1x2 figure showing the relationship
    between educational attainment and cognitive profile heterogeneity.
    """
    try:
        print("Starting combined figure generation...")
        
        # --- Data Loading and Preparation ---
        search_patterns = [
            'analysis_outputs/step3_heterogeneity_metrics_*.csv',
            'step3_heterogeneity_metrics_*.csv'
        ]
        latest_data_file = None
        for pattern in search_patterns:
            print(f"Searching for heterogeneity metrics file in: {os.path.dirname(pattern) or '.'}")
            latest_data_file = find_latest_file(pattern)
            if latest_data_file:
                break
        
        if not latest_data_file:
            print(f"ERROR: No data file found matching patterns: {search_patterns}")
            return 1
            
        print(f"Using latest data file: {latest_data_file}")
        df = pd.read_csv(latest_data_file)
        
        required_cols = ['education_level', 'percentile_range', 'percentile_iqr', 'age', 'gender']
        if not all(col in df.columns for col in required_cols):
            print(f"ERROR: Data file is missing one or more required columns: {required_cols}")
            return 1
        
        # --- Figure Setup ---
        # Define consistent ordering and labeling for education levels
        education_order = [1, 2, 3, 8, 4, 6, 5, 7]  # Some HS, HS/GED, Some College, Associate's, College, Master's, Professional, Ph.D.
        education_labels = {
            1: 'Some HS', 2: 'HS/GED', 3: 'Some College', 4: 'College',
            5: 'Professional', 6: "Master's", 7: 'Ph.D.', 8: "Associate's"
        }
        
        # Define specific color palette with improved differentiation
        colors = ['#2b7dd2', '#ff9233', '#00c2bc', '#ff6b6b', '#97dba1', '#d62728', '#9467bd', '#8c564b']
        
        # Set global font properties
        plt.rcParams['font.family'] = 'Arial'

        # Create a 1x2 horizontally stacked figure with adequate space for labels
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, 
            figsize=(12, 6),  # Increased width to accommodate labels
            dpi=300, 
            facecolor='white'
        )
        fig.set_facecolor('white')

        # --- Plot Panels ---
        plot_panel(ax1, df, 'percentile_range', 'A', colors, education_order, education_labels)
        plot_panel(ax2, df, 'percentile_iqr', 'B', colors, education_order, education_labels)

        # --- Final Figure Adjustments ---
        # Add panel labels positioned slightly above and to the left of the top of the y-axis
        # Get the position of each axis in figure coordinates
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        
        # Position panel labels slightly above and to the left of the y-axis top
        fig.text(pos1.x0 - 0.03, pos1.y1 + 0.02, 'A', fontsize=16, fontweight='bold', 
                va='bottom', ha='right', transform=fig.transFigure)
        fig.text(pos2.x0 - 0.03, pos2.y1 + 0.02, 'B', fontsize=16, fontweight='bold', 
                va='bottom', ha='right', transform=fig.transFigure)

        # Adjust layout to prevent overlap and ensure adequate space for rotated labels
        plt.subplots_adjust(left=0.08, right=0.95, wspace=0.25, bottom=0.2, top=0.9)

        # --- Save Figure ---
        output_dir = "figure_outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_dir, f"figure_1_{timestamp}.png")
        
        plt.savefig(output_filename, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        
        print(f"Successfully saved figure to: {output_filename}")
        print(f"Total data points: {len(df)}")
        print("Finished execution")
        return 0

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())