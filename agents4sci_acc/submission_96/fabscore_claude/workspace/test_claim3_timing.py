"""
Minimal timing test for Claim 3: Time comparison of the three simulation techniques.
Runs 30 simulations of each type and measures mean run time.
"""
import sys
import os
import time
import numpy as np
import math

# Add supplementary_material_fin to path
sys.path.insert(0, '/home/chenhui/fabscore/agent4sci_acc/submission_96/supplementary_material_fin')

from run_ground_truth import run_full_ssa
from run_hybrid import run_simulation

PARAMS = {
    'k1': 100, 'k_1': 1, 'k2': 1.0, 'k_2': 0.01,
    'E0': 10, 'S0': 10
}
NUM_RUNS = 30
MAX_TIME = 50.0
S_THRESHOLD = 1

print("=== Timing Test for Claim 3: Time Comparison of Three Simulation Techniques ===")
print(f"Runs: {NUM_RUNS} per method\n")

# --- Ground Truth (Full SSA) ---
print("Running Full SSA (Ground Truth)...")
ssa_times = []
for i in range(NUM_RUNS):
    t0 = time.time()
    run_full_ssa(params=PARAMS, max_time=MAX_TIME, s_threshold=S_THRESHOLD)
    t1 = time.time()
    ssa_times.append(t1 - t0)
mean_ssa = np.mean(ssa_times)
print(f"  Mean time per run: {mean_ssa:.6f}s")

# --- Pure stQSSA (ENABLE_SWITCHING = False) ---
print("\nRunning Pure stQSSA...")
stqssa_times = []
for i in range(NUM_RUNS):
    t0 = time.time()
    run_simulation(params=PARAMS, max_time=MAX_TIME, enable_switching=False,
                   thermo_threshold=0.0, s_threshold=S_THRESHOLD, guardrail_type='net')
    t1 = time.time()
    stqssa_times.append(t1 - t0)
mean_stqssa = np.mean(stqssa_times)
print(f"  Mean time per run: {mean_stqssa:.6f}s")

# --- Hybrid Model (ENABLE_SWITCHING = True) ---
print("\nRunning Hybrid Model...")
hybrid_times = []
for i in range(NUM_RUNS):
    t0 = time.time()
    run_simulation(params=PARAMS, max_time=MAX_TIME, enable_switching=True,
                   thermo_threshold=0.0, s_threshold=S_THRESHOLD, guardrail_type='net')
    t1 = time.time()
    hybrid_times.append(t1 - t0)
mean_hybrid = np.mean(hybrid_times)
print(f"  Mean time per run: {mean_hybrid:.6f}s")

print("\n=== TIMING SUMMARY ===")
print(f"Full SSA (Ground Truth):   {mean_ssa:.6f}s per run")
print(f"Pure stQSSA:               {mean_stqssa:.6f}s per run")
print(f"Hybrid Model:              {mean_hybrid:.6f}s per run")
print()
speedup_stqssa = mean_ssa / mean_stqssa if mean_stqssa > 0 else float('inf')
speedup_hybrid = mean_ssa / mean_hybrid if mean_hybrid > 0 else float('inf')
print(f"stQSSA speedup vs SSA:     {speedup_stqssa:.2f}x")
print(f"Hybrid speedup vs SSA:     {speedup_hybrid:.2f}x")
print()
print("Paper claims:")
print("  SSA ~ 0.0005s, stQSSA ~ 0.0003s, Hybrid ~ 0.0003s")
print("  (stQSSA and Hybrid should be faster than SSA)")

# Write timing files to workspace
workspace = '/home/chenhui/fabscore/agent4sci_acc/submission_96/fabscore_claude/workspace'
with open(os.path.join(workspace, 'GroundTruth_timing.txt'), 'w') as f:
    f.write(str(mean_ssa))
with open(os.path.join(workspace, 'Pure_stQSSA_timing.txt'), 'w') as f:
    f.write(str(mean_stqssa))
with open(os.path.join(workspace, 'HybridModel_timing.txt'), 'w') as f:
    f.write(str(mean_hybrid))

print(f"\nTiming files saved to workspace:")
print(f"  GroundTruth_timing.txt: {mean_ssa:.6f}s")
print(f"  Pure_stQSSA_timing.txt: {mean_stqssa:.6f}s")
print(f"  HybridModel_timing.txt: {mean_hybrid:.6f}s")
