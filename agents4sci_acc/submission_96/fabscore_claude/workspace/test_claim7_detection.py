"""
Claim 7 verification: net power drops below zero at t ≈ 2.7 seconds.
Runs N stQSSA simulations (no switching), computes mean Power_Net, finds first
time the mean is negative (detection_time_net), mimicking process_data.py logic.
"""

import sys
sys.path.insert(0, '/home/chenhui/fabscore/agent4sci_acc/submission_96/supplementary_material_fin')

import numpy as np
import math
import pandas as pd

# --- Copy of run_simulation from run_hybrid.py (read-only, not modified) ---
def stochastic_integer_mapping(continuous_state, conserved_species_indices):
    integer_state = np.floor(continuous_state).astype(int)
    fractions = continuous_state - integer_state
    for indices in conserved_species_indices:
        total_continuous = np.sum(continuous_state[indices])
        total_integer = np.sum(integer_state[indices])
        discrepancy = int(round(total_continuous) - total_integer)
        if discrepancy > 0:
            relevant_fractions = fractions[indices]
            prob_sum = np.sum(relevant_fractions)
            if prob_sum > 0:
                probabilities = relevant_fractions / prob_sum
            else:
                probabilities = np.ones_like(relevant_fractions) / len(relevant_fractions)
            choices = np.random.choice(indices, size=discrepancy, p=probabilities)
            for choice in choices:
                integer_state[choice] += 1
    return integer_state


def calculate_thermodynamics(state_dict, kinetic_params, T=310.15):
    R = 8.314
    k1, k_1, k2, k_2 = kinetic_params['k1'], kinetic_params['k_1'], kinetic_params['k2'], kinetic_params['k_2']
    E_T = kinetic_params['E0']
    S0 = kinetic_params['S0']
    S_T = S0 - state_dict['P']
    P = state_dict['P']
    km = (k_1 + k2) / k1
    p_term = (k_2 / k1) * P
    b = E_T + S_T + km + p_term
    c = E_T * (S_T + p_term)
    discriminant = max(0, b**2 - 4 * c)
    ES_t = 0.5 * (b - math.sqrt(discriminant))
    E_t = E_T - ES_t
    S_t = S_T - ES_t + p_term
    reconstructed = {'E': E_t, 'S': S_t, 'ES': ES_t, 'P': P}
    E_r, S_r, ES_r, P_r = reconstructed['E'], reconstructed['S'], reconstructed['ES'], reconstructed['P']
    kB = 1.38e-23
    Na = 6.022e23
    V = 1e-15
    conc_factor = 1.0 / (Na * V)
    mu_E = R * T * math.log(max(E_r * conc_factor, 1e-300))
    mu_S = R * T * math.log(max(S_r * conc_factor, 1e-300))
    mu_ES = R * T * math.log(max(ES_r * conc_factor, 1e-300))
    mu_P = R * T * math.log(max(P_r * conc_factor, 1e-300))
    affinity_bind = mu_E + mu_S - mu_ES
    affinity_catalyze = mu_ES - mu_E - mu_P
    affinity_net = mu_S - mu_P
    return {'bind': affinity_bind, 'catalyze': affinity_catalyze, 'net': affinity_net}, reconstructed


def run_simulation(params, max_time=50.0, enable_switching=False, thermo_threshold=0.0, s_threshold=1, guardrail_type='net'):
    k1, k_1, k2, k_2 = params['k1'], params['k_1'], params['k2'], params['k_2']
    E0, S0 = params['E0'], params['S0']
    state = np.array([E0, S0, 0, 0], dtype=float)
    stoichiometry = np.array([
        [-1, 1, 1, 0],
        [1, -1, -1, 0],
        [1, 0, -1, 1],
        [-1, 0, 1, -1]
    ]).T
    time_sim = 0.0
    results = []
    power_bind = 0.0
    power_catalyze = 0.0
    power_net = 0.0
    mode = 'stqssa'

    while time_sim < max_time:
        if mode == 'stqssa':
            P = state[3]
            S_T = S0 - P
            km = (k_1 + k2) / k1
            p_term = (k_2 / k1) * P
            b = E0 + S_T + km + p_term
            c = E0 * (S_T + p_term)
            discriminant = max(0, b**2 - 4 * c)
            ES_t = 0.5 * (b - math.sqrt(discriminant))
            E_t = E0 - ES_t
            propensity_forward = k2 * ES_t
            propensity_reverse = k_2 * E_t * P
            a0 = propensity_forward + propensity_reverse
            if a0 <= 1e-9:
                break
            if S_T < s_threshold:
                break
            tau = (1.0 / a0) * math.log(1.0 / np.random.rand())
            u = np.random.rand()
            if u < propensity_forward / a0:
                state[3] += 1  # P increases
            else:
                state[3] -= 1  # P decreases
            state[3] = max(0, state[3])

            state_dict = {'E': E_t, 'S': S_T - ES_t, 'ES': ES_t, 'P': state[3]}
            int_state = stochastic_integer_mapping(
                np.array([E_t, S_T - ES_t, ES_t, float(state[3])]),
                [[0, 2]]
            )
            state[:4] = int_state

            affinities, reconstructed = calculate_thermodynamics({'P': state[3]}, params)
            E_r, S_r, ES_r, P_r = reconstructed['E'], reconstructed['S'], reconstructed['ES'], reconstructed['P']
            J_bind = (params['k1'] * E_r * S_r) - (params['k_1'] * ES_r)
            J_catalyze = (params['k2'] * ES_r) - (params['k_2'] * E_r * P_r)
            power_bind = affinities['bind'] * J_bind
            power_catalyze = affinities['catalyze'] * J_catalyze
            net_flux = propensity_forward - propensity_reverse
            power_net = affinities['net'] * net_flux

            if enable_switching:
                violation = False
                if guardrail_type == 'net':
                    if power_net < thermo_threshold:
                        violation = True
                if violation:
                    mode = 'full_model'
                    current_P = state[3]
                    total_E = params['E0']
                    total_S = params['S0'] - current_P
                    state = np.array([total_E, total_S, 0, current_P])
        else:
            E, S, ES, P = state
            propensities = np.array([
                params['k1'] * E * S, params['k_1'] * ES,
                params['k2'] * ES, params['k_2'] * E * P
            ])
            a0 = np.sum(propensities)
            if a0 <= 1e-9:
                break
            if state[1] + state[2] < s_threshold:
                break
            tau = (1.0 / a0) * math.log(1.0 / np.random.rand())
            reaction_index = np.searchsorted(np.cumsum(propensities), np.random.rand() * a0)
            state += stoichiometry[:, reaction_index]

        time_sim += tau
        results.append({
            'E': state[0], 'S': state[1], 'ES': state[2], 'P': state[3],
            'time': time_sim,
            'Power_Bind': power_bind,
            'Power_Catalyze': power_catalyze,
            'Power_Net': power_net
        })

    return pd.DataFrame(results)


def process_ensemble_in_memory(all_runs, time_axis):
    """Match process_data.py process_ensemble() logic."""
    all_runs_interp = []
    for df in all_runs:
        df_interp = pd.DataFrame()
        df_interp['time'] = time_axis
        for col in df.columns:
            if col != 'time':
                df_interp[col] = np.interp(time_axis, df['time'], df[col])
        all_runs_interp.append(df_interp)
    concat_df = pd.concat(all_runs_interp)
    mean_df = concat_df.groupby('time').mean().reset_index()
    return mean_df


if __name__ == '__main__':
    params = {'k1': 100, 'k_1': 1, 'k2': 1.0, 'k_2': 0.01, 'E0': 10, 'S0': 10}

    for num_runs in [50, 100, 200]:
        print(f"\n=== Running {num_runs} stQSSA simulations (no switching) ===")
        np.random.seed(42)
        all_runs = []
        for i in range(num_runs):
            df = run_simulation(params, max_time=50.0, enable_switching=False)
            all_runs.append(df)
            if (i+1) % 20 == 0:
                print(f"  Completed {i+1}/{num_runs}")

        # Common time axis matching process_data.py logic
        common_time_axis = np.linspace(0, 10, 1000)
        stqssa_mean = process_ensemble_in_memory(all_runs, common_time_axis)

        power_net_stqssa = stqssa_mean['Power_Net']
        neg_mask = power_net_stqssa < 0

        if neg_mask.any():
            detection_time_net = stqssa_mean.loc[neg_mask, 'time'].iloc[0]
            print(f"  Detection time (n={num_runs}): t = {detection_time_net:.4f}s")
        else:
            print(f"  No negative power detected (n={num_runs})")

        # Show power stats
        print(f"  Power_Net stats: min={power_net_stqssa.min():.4f}, max={power_net_stqssa.max():.4f}")
        print(f"  First time power goes negative: check above")
