# run_hybrid.py
#
# This script runs simulations using either the pure stochastic Quasi-Steady-State
# Approximation (stQSSA) or the self-correcting hybrid model. The hybrid model
# starts with stQSSA and switches to a full SSA if a thermodynamic inconsistency
# is detected.

import numpy as np
import math
import pandas as pd
import openpyxl
import time

# --- 1. Functions ---

def stochastic_integer_mapping(continuous_state, conserved_species_indices):
    """
    Converts a continuous state vector to an integer state vector while preserving
    the total counts of conserved quantities using stochastic rounding.

    Args:
        continuous_state (np.array): The continuous-valued species counts.
        conserved_species_indices (list of lists): e.g., [[0, 2]] for E_T = E + ES

    Returns:
        np.array: The integer-valued state vector.
    """
    integer_state = np.floor(continuous_state).astype(int)
    fractions = continuous_state - integer_state

    for indices in conserved_species_indices:
        # Calculate the total from the continuous state and the current integer sum
        total_continuous = np.sum(continuous_state[indices])
        total_integer = np.sum(integer_state[indices])

        # Determine the number of molecules to add to preserve the rounded total
        discrepancy = int(round(total_continuous) - total_integer)
        
        if discrepancy > 0:
            # Get the fractional parts of the species involved in this conservation law
            relevant_fractions = fractions[indices]
            
            # Normalize the fractions to use as probabilities
            prob_sum = np.sum(relevant_fractions)
            if prob_sum > 0:
                probabilities = relevant_fractions / prob_sum
            else: # If all fractions are zero, distribute evenly
                probabilities = np.ones_like(relevant_fractions) / len(relevant_fractions)

            # Distribute the missing molecules stochastically
            choices = np.random.choice(indices, size=discrepancy, p=probabilities)
            for choice in choices:
                integer_state[choice] += 1
                
    return integer_state




def calculate_thermodynamics(state_dict, kinetic_params, T=310.15):
    # Calculates the chemical affinities for the elementary reactions.
    R = 8.314
    k1, k_1, k2, k_2 = kinetic_params['k1'], kinetic_params['k_1'], kinetic_params['k2'], kinetic_params['k_2']
    E_T = kinetic_params['E0']
    S0 = kinetic_params['S0']
    
    # --- stQSSA STATE RECONSTRUCTION ---
    # 1. Determine the total quantities (the slow variables)
    S_T = S0 - state_dict['P']
    P = state_dict['P'] # Get current product concentration

    # 2. Solve for the complex (ES_t) using the reversible tQSSA quadratic equation
    # Define terms for the reversible quadratic equation
    km = (k_1 + k2) / k1
    p_term = (k_2 / k1) * P
    
    b = E_T + S_T + km + p_term
    c = E_T * (S_T + p_term)
    discriminant = max(0, b**2 - 4 * c) # Ensure non-negative discriminant
    ES_t = 0.5 * (b - math.sqrt(discriminant))


    # 3. Solve for free species using conservation laws
    S_t = max(0, S_T - ES_t)
    E_t = max(0, E_T - ES_t)
    
    reconstructed_state = {'E': E_t, 'S': S_t, 'ES': ES_t, 'P': state_dict['P']}

    
    reconstructed_state = {'E': E_t, 'S': S_t, 'ES': ES_t, 'P': state_dict['P']}

    # Standard Gibbs free energy of formation for each species
    g_S, g_E = 0, 0
    g_ES = math.log(k_1 / k1) if k1 > 0 else 0
    g_P = math.log((k_1 * k_2) / (k1 * k2)) if k1 > 0 and k2 > 0 else g_ES - 10
    gibbs_numbers = {'S': g_S, 'E': g_E, 'ES': g_ES, 'P': g_P}

    # Calculate chemical potential for each species
    mus = {}
    for species, count in reconstructed_state.items():
        concentration = count if count > 1e-9 else 1e-9 # Avoid log(0)
        mus[species] = R * T * (gibbs_numbers[species] + math.log(concentration))

    # Net affinity for the overall S -> P reaction
    A_net = mus['S'] - mus['P']
    #Calculate affinities for the two elementary reactions
    A_bind = mus['E'] + mus['S'] - mus['ES']
    A_catalyze = mus['ES'] - (mus['E'] + mus['P'])

    
    affinities = {'bind': A_bind, 'catalyze': A_catalyze, 'net': A_net}
    return affinities, reconstructed_state

def run_simulation(params, max_time, enable_switching, thermo_threshold, s_threshold, guardrail_type):
    """
    Runs either a pure stQSSA or a hybrid simulation.
    """
    time_sim = 0.0
    state = np.array([params['E0'], params['S0'], 0, 0])
    results = [{'E': state[0], 'S': state[1], 'ES': state[2], 'P': state[3], 'time': time_sim, 'Power_Bind': 0.0, 'Power_Catalyze': 0.0, 'Power_Net': 0.0}]
    
    stoichiometry = np.array([
        [-1,  1,  1, -1], [-1,  1,  0,  0], [ 1, -1, -1,  1], [ 0,  0,  1, -1]
    ])

    mode = 'stQSSA'
    
    while time_sim < max_time:
        power_bind, power_catalyze, power_net = 0.0, 0.0, 0.0
        
        if mode == 'stQSSA':
            # --- Reversible stQSSA Propensity Calculation ---
            E_T = params['E0']
            S_T = params['S0'] - state[3]
            P = state[3] # Get current product
            if S_T < s_threshold:
                break

            # Calculate the complex (ES) using the reversible tQSSA formula
            if S_T > 0 or P > 0:
                km = (params['k_1'] + params['k2']) / params['k1']
                p_term = (params['k_2'] / params['k1']) * P
                b = E_T + S_T + km + p_term
                c = E_T * (S_T + p_term)
                discriminant = max(0, b**2 - 4 * c)
                n_ES_tq = 0.5 * (b - math.sqrt(discriminant))
            else:
                n_ES_tq = 0
            
            # The propensity of the effective reaction (S_total -> P) is the catalytic rate k2 * n_ES minus the reverse reaction k-2 * [E][P]
            n_E_free = E_T - n_ES_tq
            propensity_forward = params['k2'] * n_ES_tq
            propensity_reverse = params['k_2'] * n_E_free * P

            propensities = np.array([propensity_forward, propensity_reverse])
            a0 = np.sum(propensities)

            
            if a0 <= 1e-9: break
            
            tau = (1.0 / a0) * math.log(1.0 / np.random.rand())
            reaction_index = np.searchsorted(np.cumsum(propensities), np.random.rand() * a0)
            
            # --- State Update (Before Thermodynamic Check) ---
            if reaction_index == 0:
                # Effective reaction: S_total -> P
                state[3] += 1 # Increase product count by 1
            else: 
                state[3] -= 1 # Decrease product count by 1
            
            # 1. Get the new totals based on the updated P
            new_S_T = params['S0'] - state[3]
            E_T = params['E0']

            # 2. Re-solve the reversible tQSSA for the new state
            if new_S_T > 0 or state[3] > 0:
                km = (params['k_1'] + params['k2']) / params['k1']
                p_term = (params['k_2'] / params['k1']) * state[3]
                b = E_T + new_S_T + km + p_term
                c = E_T * (new_S_T + p_term)
                discriminant = max(0, b**2 - 4 * c)
                new_ES = 0.5 * (b - math.sqrt(discriminant))
            else:
                new_ES = 0

            # 3. Calculate the new free species counts
            new_E = E_T - new_ES
            new_S = new_S_T - new_ES

            # 4. Update the entire state array to reflect the new quasi-steady state
            state[0] = new_E
            state[1] = new_S
            state[2] = new_ES
            # state[3] is already updated
   
            # --- THERMODYNAMIC GUARDRAIL CHECK ---
            affinities, reconstructed = calculate_thermodynamics(
                state_dict={'S': state[1], 'P': state[3]}, kinetic_params=params
            )
            
            # Reconstruct fluxes of elementary reactions
            E_r, S_r, ES_r, P_r = reconstructed['E'], reconstructed['S'], reconstructed['ES'], reconstructed['P']
            J_bind = (params['k1'] * E_r * S_r) - (params['k_1'] * ES_r)
            J_catalyze = (params['k2'] * ES_r) - (params['k_2'] * E_r * P_r)
            

            power_bind = affinities['bind'] * J_bind
            power_catalyze = affinities['catalyze'] * J_catalyze

            #Net flux is the propensity of the overall reaction
            net_flux =  propensity_forward - propensity_reverse
            power_net = affinities['net'] * net_flux

            # --- SWITCHING LOGIC BASED ON GUARDRAIL TYPE---
            if enable_switching:
                violation = False
                if guardrail_type == 'elementary':
                    if power_bind < thermo_threshold or power_catalyze < thermo_threshold:
                        violation = True
                        print(f"    -> ELEM. VIOLATION at t={time_sim:.4f}. Switching.")
                elif guardrail_type == 'net':
                    if power_net < thermo_threshold:
                        violation = True
                        print(f"    -> NET VIOLATION at t={time_sim:.4f}. Switching.")
                
                if violation:
                    mode = 'full_model'
                    #1. Get slow and integer count species P
                    current_P = state[3] 
                    # 2. Get the conserved total E from parameters.
                    total_E = params['E0']
                    # 3. Calculate the current total S from parameters and current product.
                    total_S = params['S0'] - current_P
                    # 4. Create the state vector (all species are integers now)
                    state = np.array([total_E, total_S, 0, current_P])
                

            
    
        else: # mode == 'full_model'
            E, S, ES, P = state
            propensities = np.array([
                params['k1'] * E * S, params['k_1'] * ES,
                params['k2'] * ES, params['k_2'] * E * P
            ])
            a0 = np.sum(propensities)
            if a0 <= 1e-9: break
            if state[1] + state[2] < s_threshold:
                break
            
            tau = (1.0 / a0) * math.log(1.0 / np.random.rand())
            reaction_index = np.searchsorted(np.cumsum(propensities), np.random.rand() * a0)
            state += stoichiometry[:, reaction_index]

        time_sim += tau
        results.append({'E': state[0], 'S': state[1], 'ES': state[2], 'P': state[3], 'time': time_sim, 'Power_Bind': power_bind, 'Power_Catalyze': power_catalyze, 'Power_Net': power_net})

    return pd.DataFrame(results)

# --- 2. Main Execution Logic ---
if __name__ == '__main__':
    stQSSA_UNFRIENDLY_PARAMS = {
        'k1': 100, 'k_1': 1, 'k2': 1.0, 'k_2': 0.01,
        'E0': 10, 'S0': 10
    }
    NUM_RUNS = 450
    MAX_TIME = 50.0
    ENABLE_SWITCHING = 1
    GUARDRAIL_TYPE = 'net'  # 'elementary' or 'net'. Our results show that net guardrail performs better

    if ENABLE_SWITCHING:
        output_filename = 'HybridModel.xlsx'
        timing_filename = 'HybridModel_timing.txt'
        print("--- CONFIGURATION: Hybrid Model (Switcher Enabled) ---")
    else:
        output_filename = 'Pure_stQSSA.xlsx'
        timing_filename = 'Pure_stQSSA_timing.txt'
        print("--- CONFIGURATION: Pure stQSSA Model (Switcher Disabled) ---")
        
    print(f"Parameters: {stQSSA_UNFRIENDLY_PARAMS}")
    
    # MODIFIED: Added timing logic
    total_start_time = time.time()
    run_times = []

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        for i in range(NUM_RUNS):
            print(f"Running simulation {i + 1}/{NUM_RUNS}...", end="")
            run_start_time = time.time()
            results_df = run_simulation(
                params=stQSSA_UNFRIENDLY_PARAMS, max_time=MAX_TIME, 
                enable_switching=ENABLE_SWITCHING, thermo_threshold=0.0, s_threshold=1, guardrail_type=GUARDRAIL_TYPE
            )
            run_end_time = time.time()
            elapsed = run_end_time - run_start_time
            run_times.append(elapsed)
            print(f" done in {elapsed:.4f}s.")
            results_df.to_excel(writer, sheet_name=f"Run_{i + 1}", index=False)
    
    total_end_time = time.time()
    mean_run_time = np.mean(run_times)

    with open(timing_filename, 'w') as f:
        f.write(str(mean_run_time))

    print(f"\nSuccessfully completed {NUM_RUNS} runs.")
    print(f"Total execution time: {total_end_time - total_start_time:.2f}s")
    print(f"Mean time per run: {mean_run_time:.4f}s")
    print(f"Data saved to '{output_filename}'")
    print(f"Timing data saved to '{timing_filename}'")
