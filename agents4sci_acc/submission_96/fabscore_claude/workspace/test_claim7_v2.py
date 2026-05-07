"""
Claim 7 verification using actual run_simulation from run_hybrid.py.
Run N stQSSA simulations (no switching), compute mean Power_Net, find first
time mean is negative (detection_time_net), matching process_data.py logic.
"""

import sys
import os
sys.path.insert(0, '/home/chenhui/fabscore/agent4sci_acc/submission_96/supplementary_material_fin')

import numpy as np
import pandas as pd

# Import the actual simulation function
from run_hybrid import run_simulation

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

    for num_runs in [50, 150]:
        print(f"\n=== Running {num_runs} stQSSA simulations (ENABLE_SWITCHING=False) ===")
        np.random.seed(42)
        all_runs = []
        for i in range(num_runs):
            df = run_simulation(
                params=params, max_time=50.0,
                enable_switching=False,
                thermo_threshold=0.0,
                s_threshold=1,
                guardrail_type='net'
            )
            all_runs.append(df)
            if (i+1) % 25 == 0:
                print(f"  Completed {i+1}/{num_runs}")

        # Check first run stats
        print(f"\n  First run Power_Net: min={all_runs[0]['Power_Net'].min():.4f}, max={all_runs[0]['Power_Net'].max():.4f}")
        print(f"  First run time range: {all_runs[0]['time'].min():.4f} to {all_runs[0]['time'].max():.4f}")

        # Common time axis matching process_data.py (linspace with 1000 points from 0 to max)
        # Check what time axis process_data.py uses
        common_time_axis = np.linspace(0, 10, 1000)
        stqssa_mean = process_ensemble_in_memory(all_runs, common_time_axis)

        power_net_stqssa = stqssa_mean['Power_Net']
        neg_mask = power_net_stqssa < 0

        print(f"\n  Power_Net mean stats (n={num_runs}):")
        print(f"    min={power_net_stqssa.min():.4f}, max={power_net_stqssa.max():.4f}")
        print(f"    any negative: {neg_mask.any()}")

        if neg_mask.any():
            detection_time_net = stqssa_mean.loc[neg_mask, 'time'].iloc[0]
            print(f"  Detection time (n={num_runs}): t = {detection_time_net:.4f}s")
            print(f"  Paper claims: t ≈ 2.7s")
        else:
            print(f"  No negative power detected (n={num_runs})")
