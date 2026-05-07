# run_sweep_heatmap.py
#
# This script performs a parameter sweep to evaluate the robustness and
# comparative performance of the 'elementary' vs. 'net' power guardrails.

import numpy as np
import pandas as pd
import time
from tqdm import tqdm # For a nice progress bar
import matplotlib.pyplot as plt

# --- Import simulation functions from other files ---
# For this to work, the files must be in the same directory.
from run_ground_truth import run_full_ssa
from run_hybrid import run_simulation # The newly modified version

# --- 1. Simulation and Evaluation ---

def evaluate_performance(params, base_params, num_runs, max_time, time_axis):
    """
    For a given parameter set, this function runs ensembles for the ground truth
    and the two hybrid guardrail models, then calculates their error.
    """
    full_params = {**base_params, **params} # Combine simulation settings with kinetic rates
    sp = 'P' # Species of interest
    # 1. Run Ground Truth (Full SSA)
    truth_runs = [run_full_ssa(full_params, max_time, 1) for _ in range(num_runs)]
    truth_mean = pd.concat([pd.DataFrame({'time': time_axis, sp: np.interp(time_axis, df['time'], df[sp])}) for df in truth_runs])\
                   .groupby('time').mean().reset_index()

    # 2. Run Hybrid with Elementary Guardrail
    hybrid_elem_runs = [run_simulation(full_params, max_time, True, 0.0, 1, 'elementary') for _ in range(num_runs)]
    hybrid_elem_mean = pd.concat([pd.DataFrame({'time': time_axis, sp: np.interp(time_axis, df['time'], df[sp])}) for df in hybrid_elem_runs])\
                         .groupby('time').mean().reset_index()

    # 3. Run Hybrid with Net Power Guardrail
    hybrid_net_runs = [run_simulation(full_params, max_time, True, 0.0, 1, 'net') for _ in range(num_runs)]
    hybrid_net_mean = pd.concat([pd.DataFrame({'time': time_axis, sp: np.interp(time_axis, df['time'], df[sp])}) for df in hybrid_net_runs])\
                        .groupby('time').mean().reset_index()
    
    # 4. Calculate Integrated Absolute Error (IAE) for both hybrid models
    iae_elem = np.trapezoid(np.abs(hybrid_elem_mean[sp] - truth_mean[sp]), x=time_axis)
    iae_net = np.trapezoid(np.abs(hybrid_net_mean[sp] - truth_mean[sp]), x=time_axis)

    return {'iae_elem': iae_elem, 'iae_net': iae_net}

# --- 2. Parameter Sweep and Heatmap drawing ---
def draw_heatmap():
    # Create parameter ranges (log-spaced)
    k1_range = np.logspace(-2, 2, 10)    # 0.01 to 100
    k_1_range = np.logspace(-2, 2, 10)   # 0.01 to 100
    k2_range = np.logspace(-2, 2, 10)    # 0.01 to 100
    K_eq = 10000  # Equilibirum constant same as in the paper's demonstration
    
    # Initialize arrays to store results
    bind_free_diff = np.zeros((10, 10))      # k1 vs k-1
    bind_cat_diff = np.zeros((10, 10))       # k1 vs k2
    free_cat_diff = np.zeros((10, 10))       # k-1 vs k2
    
    # Base parameters
    base_params = {'E0': 10, 'S0': 10}
    max_time = 50.0
    time_axis = np.linspace(0, max_time, int(max_time * 20) + 1)
    num_runs = 10  # Number of ensemble runs per parameter set
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. k1 vs k-1 (fixing k2 at median value)
    k2_med = np.sqrt(k2_range[0] * k2_range[-1])
    print("Computing k1 vs k-1 heatmap...")
    for i, k1 in enumerate(tqdm(k1_range)):
        for j, k_1 in enumerate(k_1_range):
            params = {'k1': k1, 'k_1': k_1, 'k2': k2_med, 'k_2': k1 * k2_med / (k_1 * K_eq)}
            perf = evaluate_performance(params, base_params, num_runs, max_time, time_axis)
            bind_free_diff[i, j] = perf['iae_elem'] - perf['iae_net']
    
    # 2. k1 vs k2 (fixing k-1 at median value)
    k_1_med = np.sqrt(k_1_range[0] * k_1_range[-1])
    print("Computing k1 vs k2 heatmap...")
    for i, k1 in enumerate(tqdm(k1_range)):
        for j, k2 in enumerate(k2_range):
            params = {'k1': k1, 'k_1': k_1_med, 'k2': k2, 'k_2': k1 * k2 / (k_1_med * K_eq)}
            perf = evaluate_performance(params, base_params, num_runs, max_time, time_axis)
            bind_cat_diff[i, j] = perf['iae_elem'] - perf['iae_net']
    
    # 3. k-1 vs k2 (fixing k1 at median value)
    k1_med = np.sqrt(k1_range[0] * k1_range[-1])
    print("Computing k-1 vs k2 heatmap...")
    for i, k_1 in enumerate(tqdm(k_1_range)):
        for j, k2 in enumerate(k2_range):
            params = {'k1': k1_med, 'k_1': k_1, 'k2': k2, 'k_2': k1_med * k2 / (k_1 * K_eq)}
            perf = evaluate_performance(params, base_params, num_runs, max_time, time_axis)
            free_cat_diff[i, j] = perf['iae_elem'] - perf['iae_net']
    
    # Plot heatmaps
    # Use a symmetric colormap centered at zero
    vmax = max(np.abs(bind_free_diff).max(), np.abs(bind_cat_diff).max(), np.abs(free_cat_diff).max())
    vmin = -vmax    
    
    im1 = ax1.imshow(bind_free_diff, cmap='RdBu', aspect='auto',
                     extent=[np.log10(k_1_range[0]), np.log10(k_1_range[-1]),
                            np.log10(k1_range[0]), np.log10(k1_range[-1])],
                     vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(bind_cat_diff, cmap='RdBu', aspect='auto',
                     extent=[np.log10(k2_range[0]), np.log10(k2_range[-1]),
                            np.log10(k1_range[0]), np.log10(k1_range[-1])],
                     vmin=vmin, vmax=vmax)
    im3 = ax3.imshow(free_cat_diff, cmap='RdBu', aspect='auto',
                     extent=[np.log10(k2_range[0]), np.log10(k2_range[-1]),
                            np.log10(k_1_range[0]), np.log10(k_1_range[-1])],
                     vmin=vmin, vmax=vmax)
    
    # Add colorbars and labels
    fig.colorbar(im1, ax=ax1, label='IAE difference\n(Elementary - Net)')
    fig.colorbar(im2, ax=ax2, label='IAE difference\n(Elementary - Net)')
    fig.colorbar(im3, ax=ax3, label='IAE difference\n(Elementary - Net)')
    
    # Add labels and titles
    ax1.set_xlabel('log10(k₋₁)')
    ax1.set_ylabel('log10(k₁)')
    ax1.set_title('Binding vs Unbinding\n(k₂ fixed)')
    
    ax2.set_xlabel('log10(k₂)')
    ax2.set_ylabel('log10(k₁)')
    ax2.set_title('Binding vs Catalysis\n(k₋₁ fixed)')
    
    ax3.set_xlabel('log10(k₂)')
    ax3.set_ylabel('log10(k₋₁)')
    ax3.set_title('Unbinding vs Catalysis\n(k₁ fixed)')
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_heatmaps.png', dpi=300, bbox_inches='tight')
    print("\nHeatmaps saved as 'parameter_sweep_heatmaps.png'")

# --- 3. Main Sweep Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    NUM_PARAM_SETS = 20    # Number of different kinetic parameter sets to test
    NUM_RUNS_PER_SET = 10  # Ensemble size for each parameter set (higher is better but slower)
    MAX_TIME = 50.0
    
    # Base parameters (initial conditions)
    BASE_PARAMS = {'E0': 100, 'S0': 100}

    # Common time axis for error integration
    TIME_AXIS = np.linspace(0, MAX_TIME, int(MAX_TIME * 20) + 1) # 20 points per time unit
    
    print(f"--- Starting Parameter Sweep ---")
    print(f"Sets to test: {NUM_PARAM_SETS}")
    print(f"Runs per set: {NUM_RUNS_PER_SET}\n")

    draw_heatmap()