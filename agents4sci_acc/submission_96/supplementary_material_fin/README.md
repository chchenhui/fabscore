# Thermodynamic Guardrails: Supplementary Code

This repository contains the source code to reproduce the simulations and figures presented in the paper "Thermodynamic Guardrails: A Bond Graph-Based Method for Self-Correcting Model Reduction in Autonomous Scientific Discovery."

## Overview

The code is organized into four main Python scripts:

  - `run_ground_truth.py`: Simulates the full elementary Michaelis-Menten reaction network using the Gillespie Stochastic Simulation Algorithm (SSA). This generates the baseline "ground truth" data.
  - `run_hybrid.py`: Simulates both the pure stochastic Quasi-Steady-State Approximation (stQSSA) and the self-correcting hybrid model. A configuration flag within the script (`ENABLE_SWITCHING`) determines which model is run.
  - `process_data.py`: Takes the raw output from the simulation scripts, processes the data (interpolation, averaging), calculates error metrics, and generates the final figures used in the paper (Figure 1 and Figure 2).
  - `run_sweep_heatmap.py`: Runs a parameter sweep across various kinetic rates to evaluate the robustness of the guardrail trigger and generates Figure 4.

## Dependencies

[cite\_start]All scripts are written in Python 3. The required packages are listed in `requirements.txt`[cite: 552]. You can install them using pip:

`pip install -r requirements.txt`

## Instructions for Reproduction

To reproduce the figures from the paper, follow these steps in order.

### Step 1: Generate the Ground Truth Data

Run the full SSA simulation to create the ground truth dataset. This script will generate the files `GroundTruth.xlsx` and `GroundTruth_timing.txt`.

`python run_ground_truth.py`

### Step 2: Generate the Pure stQSSA Data

Run the hybrid/stQSSA script with the switcher disabled. First, open `run_hybrid.py` and ensure the configuration flag is set to `False`:

#### In run_hybrid.py
`ENABLE_SWITCHING = False`

Then, run the script from your terminal. This will generate the file `Pure_stQSSA.xlsx` and its corresponding timing file.

`python run_hybrid.py`

### Step 3: Generate the Hybrid Model Data

Now, run the script with the switcher enabled. Open `run_hybrid.py` again and set the flag to `True`:

#### In run_hybrid.py
`ENABLE_SWITCHING = True`

Run the script again. This will generate the file `HybridModel.xlsx` and its timing file.

`python run_hybrid.py`

### Step 4: Process Data and Generate Figures 1 & 2

Run the data processing script. This script will read the three `.xlsx` files you just created and generate the final plots showing the sequestration failure and thermodynamic analysis. It will also read the three `.txt` timing files to provide the quantitative efficiency comparison for Figure 3.

`python process_data.py`

After this step, you will have two new image files in your directory: `error_analysis_plot_P.png` and `concentration_trajectories_plot_P.png`, which correspond to the figures presented in the Results section of the paper.

### Step 5: Run Parameter Sweep and Generate Figure 4

Finally, run the parameter sweep script. This script directly imports functions from `run_hybrid.py` and must be in the same directory. In larger simulations, this may take a significant amount of time to complete.

`python run_sweep_heatmap.py`