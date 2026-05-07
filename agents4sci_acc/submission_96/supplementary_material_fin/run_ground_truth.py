# run_ground_truth.py
#
# This script runs the full Gillespie Stochastic Simulation Algorithm (SSA)
# for the elementary Michaelis-Menten reaction network. This serves as the
# "ground truth" for the model comparisons.

import numpy as np
import math
import pandas as pd
import openpyxl
import time
from tqdm import tqdm  # Import tqdm for progress bar

# --- 1. Functions ---

def calculate_thermodynamics(state_dict, kinetic_params, T=310.15):
    """
    Calculates the chemical affinities for the elementary reactions.
    
    Args:
        state_dict (dict): A dictionary of current species counts {'E', 'S', 'ES', 'P'}.
        kinetic_params (dict): A dictionary of kinetic rate constants.
        T (float): Temperature in Kelvin.
        
    Returns:
        dict: A dictionary containing the affinities for the binding and catalysis steps.
    """
    R = 8.314  # Ideal gas constant
    k1, k_1, k2, k_2 = kinetic_params['k1'], kinetic_params['k_1'], kinetic_params['k2'], kinetic_params['k_2']

    # Standard Gibbs free energy of formation for each species, derived from rate constants
    g_S, g_E = 0, 0
    g_ES = math.log(k_1 / k1) if k1 > 0 else 0
    g_P = math.log((k_1 * k_2) / (k1 * k2)) if k1 > 0 and k2 > 0 else g_ES - 10
    gibbs_numbers = {'S': g_S, 'E': g_E, 'ES': g_ES, 'P': g_P}

    # Calculate chemical potential for each species
    mus = {}
    for species, count in state_dict.items():
        concentration = count if count > 1e-9 else 1e-9  # Avoid log(0)
        mus[species] = R * T * (gibbs_numbers[species] + math.log(concentration))

    # Calculate affinities for the two elementary reactions
    A_bind = mus['E'] + mus['S'] - mus['ES']
    A_catalyze = mus['ES'] - (mus['E'] + mus['P'])
    A_net = mus['S'] - mus['P']
    
    return {'bind': A_bind, 'catalyze': A_catalyze, 'net': A_net}

def run_full_ssa(params, max_time, s_threshold):
    """
    Implements the Gillespie SSA for the full Michaelis-Menten model.
    """
    time = 0.0
    state = np.array([params['E0'], params['S0'], 0, 0])  # [E, S, ES, P]
    
    # Initialize all power columns for consistency
    results = [{'E': state[0], 'S': state[1], 'ES': state[2], 'P': state[3], 'time': time, 
                'Power_Bind': 0.0, 'Power_Catalyze': 0.0, 'Power_Net': 0.0}]
    
    stoichiometry = np.array([
        [-1,  1,  1, -1],
        [-1,  1,  0,  0],
        [ 1, -1, -1,  1],
        [ 0,  0,  1, -1]
    ])

    while time < max_time and state[1] + state[2] > s_threshold: # Check total substrate
        E, S, ES, P = state
        
        propensities = np.array([
            params['k1'] * E * S,
            params['k_1'] * ES,
            params['k2'] * ES,
            params['k_2'] * E * P
        ])
        
        a0 = np.sum(propensities)
        if a0 <= 1e-9: break

        tau = (1.0 / a0) * math.log(1.0 / np.random.rand())
        reaction_index = np.searchsorted(np.cumsum(propensities), np.random.rand() * a0)

        # --- POWER CALCULATION ---
        affinities = calculate_thermodynamics(dict(zip(['E','S','ES','P'], state)), params)
        
        # Calculate net fluxes for each elementary step
        J_bind = propensities[0] - propensities[1]
        J_catalyze = propensities[2] - propensities[3]

        # Calculate power for each step independently
        power_bind = affinities['bind'] * J_bind
        power_catalyze = affinities['catalyze'] * J_catalyze
        # Net power for S -> P is driven by the net catalysis flux
        power_net = affinities['net'] * J_catalyze

        # Update state and time
        state += stoichiometry[:, reaction_index]
        time += tau
        
        # MODIFIED: Append all power values to the results
        results.append({
            'E': state[0], 'S': state[1], 'ES': state[2], 'P': state[3], 'time': time, 
            'Power_Bind': power_bind, 
            'Power_Catalyze': power_catalyze, 
            'Power_Net': power_net
        })

    return pd.DataFrame(results)

# --- 2. Main Execution Logic ---
if __name__ == '__main__':
    SQSSA_UNFRIENDLY_PARAMS = {
        'k1': 100, 'k_1': 1, 'k2': 1.0, 'k_2': 0.01,
        'E0': 10, 'S0': 10
    }
    NUM_RUNS = 450
    MAX_TIME = 50.0
    S_THRESHOLD = 1
    OUTPUT_FILENAME = 'GroundTruth.xlsx'
    TIMING_FILENAME = 'GroundTruth_timing.txt'
    
    print(f"--- Running Ground Truth (Full SSA) Simulations ---")
    
    total_start_time = time.time()
    run_times = []

    with pd.ExcelWriter(OUTPUT_FILENAME, engine='openpyxl') as writer:
        for i in tqdm(range(NUM_RUNS), desc="Running simulations"):
            run_start_time = time.time()
            results_df = run_full_ssa(
                params=SQSSA_UNFRIENDLY_PARAMS, max_time=MAX_TIME, s_threshold=S_THRESHOLD
            )
            run_end_time = time.time()
            elapsed = run_end_time - run_start_time
            run_times.append(elapsed)
            results_df.to_excel(writer, sheet_name=f"Run_{i + 1}", index=False)
            
    total_end_time = time.time()
    mean_run_time = np.mean(run_times)

    with open(TIMING_FILENAME, 'w') as f:
        f.write(str(mean_run_time))
        
    print(f"\nSuccessfully completed {NUM_RUNS} runs.")
    print(f"Total execution time: {total_end_time - total_start_time:.2f}s")
    print(f"Mean time per run: {mean_run_time:.4f}s")
    print(f"Ground truth data saved to '{OUTPUT_FILENAME}'")
    print(f"Timing data saved to '{TIMING_FILENAME}'")

