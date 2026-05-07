import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def process_ensemble(filename, num_runs, time_axis):
    """
    Loads a multi-sheet Excel file, interpolates each run onto a common time axis,
    and calculates the mean and standard deviation of the ensemble.
    """
    print(f"Processing '{filename}'...")
    all_runs_interp = []
    
    # Load all runs from the Excel file with a progress bar
    xls = pd.ExcelFile(filename)
    for i in tqdm(range(num_runs), desc=f"Loading {filename}"):
        sheet_name = f"Run_{i + 1}"
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Interpolate the data from this run onto the common time_axis
        df_interp = pd.DataFrame(columns=df.columns)
        df_interp['time'] = time_axis
        for col in df.columns:
            if col != 'time':
                df_interp[col] = np.interp(time_axis, df['time'], df[col])
        all_runs_interp.append(df_interp)

    # Concatenate all interpolated DataFrames
    concat_df = pd.concat(all_runs_interp)
    
    # Calculate mean and std dev at each time point
    mean_df = concat_df.groupby('time').mean().reset_index()
    std_df = concat_df.groupby('time').std().reset_index()
    
    return mean_df, std_df

def calculate_absolute_error(model_df, truth_df, species):
    """
    Calculates the Absolute Error between a model and the ground truth.
    """
    return np.abs(model_df[species] - truth_df[species])

def create_analysis_plots(ground_truth_mean, ground_truth_std, stqssa_mean, hybrid_mean, common_time_axis, detection_time_net):
    """
    Generates and saves two 2x2 subplot figures:
    1. A grid of error vs. thermodynamic power plots for each species.
    2. A grid of concentration trajectory plots for each species.
    """
    species_list = ['E', 'S', 'ES', 'P']
    
    # --- FIGURE 1: Error and Power Analysis (2x2 Grid) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    fig1.suptitle('Error and Thermodynamic Power Analysis', fontsize=20, fontweight='bold')
    
    for ax, sp in zip(axes1.flatten(), species_list):
        # Left Y-axis for Numerical Error
        color = 'tab:red'
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(f'Absolute Error ({sp})', color=color, fontsize=12)
        
        stqssa_error = calculate_absolute_error(stqssa_mean, ground_truth_mean, sp)
        hybrid_error = calculate_absolute_error(hybrid_mean, ground_truth_mean, sp)
        
        ax.plot(common_time_axis, stqssa_error, color=color, linestyle='-', label='stQSSA Error')
        ax.plot(common_time_axis, hybrid_error, color='tab:blue', linestyle='--', label='Hybrid Error')
        ax.plot(common_time_axis, ground_truth_std[sp], color='tab:orange', linestyle=':', label='Ground Truth Std Dev')
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax.set_ylim(bottom=0)

        # Right Y-axis for Thermodynamic Power
        ax2 = ax.twinx()
        ax2.set_ylabel('Thermodynamic Power (A⋅J)', fontsize=12)
        ax2.plot(common_time_axis, stqssa_mean['Power_Net'], color='tab:green', linestyle='-.', label='Net Power')
        ax2.plot(common_time_axis, stqssa_mean['Power_Bind'], color='tab:purple', linestyle=':', label='Binding Power')
        ax2.plot(common_time_axis, stqssa_mean['Power_Catalyze'], color='tab:pink', linestyle=':', label='Catalysis Power')

        if detection_time_net is not None:
            ax2.axvline(detection_time_net, color='brown', linestyle='--', label='Guardrail Trigger Time') 
        ax2.axhline(0, color='black', linewidth=1.5, linestyle='--')
        ax.set_title(f'Species: {sp}', fontsize=14)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.savefig('error_power_analysis_grid.png', dpi=300)
    
    # --- FIGURE 2: Concentration Trajectories (2x2 Grid) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    fig2.suptitle('Comparison of Species Concentration Trajectories', fontsize=20, fontweight='bold')

    for ax, sp in zip(axes2.flatten(), species_list):
        ax.plot(common_time_axis, ground_truth_mean[sp], color='black', linestyle='-', label='Ground Truth (Mean)')
        ax.fill_between(
            common_time_axis,
            ground_truth_mean[sp] - ground_truth_std[sp],
            ground_truth_mean[sp] + ground_truth_std[sp],
            color='orange', alpha=0.3, label='Ground Truth (Std Dev)'
        )
        ax.plot(common_time_axis, stqssa_mean[sp], color='red', linestyle='--', label='Pure stQSSA')
        ax.plot(common_time_axis, hybrid_mean[sp], color='blue', linestyle=':', label='Hybrid Model')
        if detection_time_net is not None:
            ax.axvline(detection_time_net, color='brown', linestyle='--', label='Guardrail Trigger Time')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(f'Concentration of {sp}', fontsize=12)
        ax.set_title(f'Species: {sp}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)

    plt.savefig('concentration_trajectories_grid.png', dpi=300)
    plt.close('all')
    print("\nAnalysis plots saved as 'error_power_analysis_grid.png' and 'concentration_trajectories_grid.png'")


# --- Main Execution Logic ---
if __name__ == '__main__':
    # 1. Configuration
    NUM_RUNS = 450
    MAX_TIME = 5.0 
    TIME_POINTS = 1000
    common_time_axis = np.linspace(0, MAX_TIME, TIME_POINTS)

    # 2. Data Processing
    ground_truth_mean, ground_truth_std = process_ensemble('GroundTruth.xlsx', NUM_RUNS, common_time_axis)
    stqssa_mean, _ = process_ensemble('Pure_stQSSA.xlsx', NUM_RUNS, common_time_axis)
    hybrid_mean, _ = process_ensemble('HybridModel.xlsx', NUM_RUNS, common_time_axis)

    # 3. Power Violation Analysis
    power_net_stqssa = stqssa_mean['Power_Net']
    power_bind_stqssa = stqssa_mean['Power_Bind']
    power_catalyze_stqssa = stqssa_mean['Power_Catalyze']

    power_signs = np.sign(power_net_stqssa)
    sign_changes = np.where(np.diff(power_signs) < 0)[0]

    # Find the first time point where the stQSSA's net power becomes negative.
    # This is the moment the guardrail would trigger the switch in the hybrid model.
    try:
        detection_time_net = stqssa_mean.loc[power_net_stqssa < 0, 'time'].iloc[0]
        print(f"\nGuardrail(NET) Trigger Time Identified: t = {detection_time_net:.4f}s")
    except IndexError:
        detection_time = None
        print("\nNo power violation was detected in the stQSSA run.")

    # 4. Visualization
    create_analysis_plots(
        ground_truth_mean,
        ground_truth_std,
        stqssa_mean,
        hybrid_mean,
        common_time_axis,
        detection_time_net
    )

    # 5. Efficiency Analysis
    print("\n--- Efficiency Analysis ---")
    timing_files = {
        'Full SSA (Ground Truth)': 'GroundTruth_timing.txt',
        'Pure stQSSA': 'Pure_stQSSA_timing.txt',
        'Hybrid Model': 'HybridModel_timing.txt'
    }
    times = {}
    for model_name, filename in timing_files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                times[model_name] = float(f.read().strip())
            gt_time = times.get('Full SSA (Ground Truth)')
            if gt_time is not None and model_name != 'Full SSA (Ground Truth)':
                speedup = gt_time / times[model_name] if times[model_name] > 0 else float('inf')
                print(f"Mean Time per Run ({model_name}): {times[model_name]:.4f}s (Speedup: {speedup:.2f}x)")
            else:
                print(f"Mean Time per Run ({model_name}): {times[model_name]:.4f}s")
        else:
            print(f"Warning: Timing file '{filename}' not found.")