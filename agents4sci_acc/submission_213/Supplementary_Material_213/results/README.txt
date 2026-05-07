results/ README
====================
This folder contains intermediate outputs prepared for Agents4Science review.

* llm_eval_scores.csv — automatic scoring of provided LLM outputs vs reference.
* compute_runtimes_times.json — environment metadata and placeholder timings (training not executed in this environment).
* MISSING (to be produced on a DL-enabled machine): US_trajectory.csv, UK_trajectory.csv, sensitivity_*.csv, robustness_*.csv.

Why some files are missing
--------------------------
The current sandbox does not include deep-learning dependencies (e.g., PyTorch). 
Please run `train.py`, `sensitivity.py`, and `robustness.py` on a local machine with the dependencies listed in requirements.txt to generate the remaining artefacts.
