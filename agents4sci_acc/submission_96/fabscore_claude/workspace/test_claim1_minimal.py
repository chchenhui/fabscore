"""
Minimal test script for claim 1 verification.
Runs a small number of simulations (5 runs each) to verify the pipeline works
and that the output data structure matches what process_data.py expects.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add the supplementary_material_fin to path for importing functions
sys.path.insert(0, '/home/chenhui/fabscore/agent4sci_acc/submission_96/supplementary_material_fin')

# Import the simulation functions
from run_ground_truth import run_full_ssa
from run_hybrid import run_simulation

# Change to workspace directory so output files go there
workspace_dir = '/home/chenhui/fabscore/agent4sci_acc/submission_96/fabscore_claude/workspace'
os.chdir(workspace_dir)

# Parameters from the paper
PARAMS = {'k1': 100, 'k_1': 1, 'k2': 1.0, 'k_2': 0.01, 'E0': 10, 'S0': 10}
NUM_RUNS = 5  # Minimal run count for testing
MAX_TIME = 50.0
S_THRESHOLD = 1

print("=== Step 1: Running Ground Truth (Full SSA) ===")
with pd.ExcelWriter('GroundTruth_test.xlsx', engine='openpyxl') as writer:
    for i in range(NUM_RUNS):
        df = run_full_ssa(params=PARAMS, max_time=MAX_TIME, s_threshold=S_THRESHOLD)
        df.to_excel(writer, sheet_name=f"Run_{i+1}", index=False)
        print(f"  Run {i+1}: {len(df)} time steps, columns: {list(df.columns)}")

print("\n=== Step 2: Running Pure stQSSA ===")
with pd.ExcelWriter('Pure_stQSSA_test.xlsx', engine='openpyxl') as writer:
    for i in range(NUM_RUNS):
        df = run_simulation(params=PARAMS, max_time=MAX_TIME, enable_switching=False,
                           thermo_threshold=0.0, s_threshold=S_THRESHOLD, guardrail_type='net')
        df.to_excel(writer, sheet_name=f"Run_{i+1}", index=False)
        print(f"  Run {i+1}: {len(df)} time steps, columns: {list(df.columns)}")

print("\n=== Step 3: Running Hybrid Model ===")
with pd.ExcelWriter('HybridModel_test.xlsx', engine='openpyxl') as writer:
    for i in range(NUM_RUNS):
        df = run_simulation(params=PARAMS, max_time=MAX_TIME, enable_switching=True,
                           thermo_threshold=0.0, s_threshold=S_THRESHOLD, guardrail_type='net')
        df.to_excel(writer, sheet_name=f"Run_{i+1}", index=False)
        print(f"  Run {i+1}: {len(df)} time steps, columns: {list(df.columns)}")

print("\n=== Step 4: Processing Data (matching process_data.py logic) ===")
from process_data import process_ensemble, calculate_absolute_error, create_analysis_plots

TIME_POINTS = 100
common_time_axis = np.linspace(0, 5.0, TIME_POINTS)

gt_mean, gt_std = process_ensemble('GroundTruth_test.xlsx', NUM_RUNS, common_time_axis)
stqssa_mean, _ = process_ensemble('Pure_stQSSA_test.xlsx', NUM_RUNS, common_time_axis)
hybrid_mean, _ = process_ensemble('HybridModel_test.xlsx', NUM_RUNS, common_time_axis)

print(f"\nGround truth mean columns: {list(gt_mean.columns)}")
print(f"stQSSA mean columns: {list(stqssa_mean.columns)}")
print(f"Hybrid mean columns: {list(hybrid_mean.columns)}")

# Check Power_Net column exists in stQSSA
print(f"\nstQSSA Power_Net stats: min={stqssa_mean['Power_Net'].min():.4f}, max={stqssa_mean['Power_Net'].max():.4f}")

# Find trigger time
power_net_stqssa = stqssa_mean['Power_Net']
detection_time_net = None
try:
    detection_time_net = stqssa_mean.loc[power_net_stqssa < 0, 'time'].iloc[0]
    print(f"\nGuardrail Trigger Time: t = {detection_time_net:.4f}s")
except (IndexError, KeyError):
    print("\nNo trigger time detected in this short run (may need more runs)")

# Generate the figure
create_analysis_plots(gt_mean, gt_std, stqssa_mean, hybrid_mean, common_time_axis, detection_time_net)

# Verify files exist
import os
png_files = ['error_power_analysis_grid.png', 'concentration_trajectories_grid.png']
for f in png_files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"\nGenerated: {f} ({size} bytes)")
    else:
        print(f"\nMISSING: {f}")

print("\n=== DONE ===")
print("The create_analysis_plots() function uses:")
print("  - LEFT axis: stQSSA error in TAB:RED, hybrid error in TAB:BLUE")
print("  - RIGHT axis: Power_Net in TAB:GREEN (+ Power_Bind purple, Power_Catalyze pink)")
print("  - VERTICAL LINE at detection_time_net (trigger time)")
