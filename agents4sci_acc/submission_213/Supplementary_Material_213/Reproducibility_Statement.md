# Reproducibility Statement

**Paper:** “Sustainable Investment Decision‑Making on Office Buildings using Reinforcement Learning and Large Language Models” (submission #213)  
**Supplementary folder:** `Supplementary Material/`  
**Contact:** Anonymous (double‑blind).

## What is included
- **Code** to re‑create all results: `rl_env.py`, `dqn_agent.py`, `train.py`, `sensitivity.py`, `robustness.py`, `llm_eval.py`, `build_parameters_from_clusters.py`, `compute_runtimes.py`, `run.sh`.
- **Data inputs**: `data_sources_clustered.xlsx` (clustered variables for US/UK). The builder script writes the import‑ready `data_sources.csv` used by the environment.
- **LLM artefacts**: `llm_prompts_responses.json`, `datasets/llm_eval_reference.csv` and example `llm_eval_outputs.csv` (format for automatic scoring).
- **Environment spec**: `requirements.txt`.

> **Intermediate outputs expected for this submission** (please generate before final upload):  
> `results/US_trajectory.csv`, `results/UK_trajectory.csv`, `results/sensitivity_US.csv`, `results/sensitivity_UK.csv`, `results/robustness_US.csv`, `results/robustness_UK.csv`, `results/llm_eval_scores.csv`, and `results/compute_runtimes_times.json` (from `compute_runtimes.py`).

## How to reproduce (deterministic path)
1. **Environment**
   ```bash
   python -V   # 3.10+
   pip install -r requirements.txt
   ```

2. **Build import‑ready parameters**
   ```bash
   python build_parameters_from_clusters.py      --source_xlsx data_sources_clustered.xlsx      --aggregate median      --out_csv data_sources.csv
   ```

3. **Train and export greedy trajectories (US & UK)**
   ```bash
   python train.py --case US --data data_sources.csv --episodes 5000 --seed 42      --outdir results --modeldir models

   python train.py --case UK --data data_sources.csv --episodes 5000 --seed 42      --outdir results --modeldir models
   ```

4. **Sensitivity analysis and robustness tests**
   ```bash
   python sensitivity.py --case US --data data_sources.csv --model models/dqn_US.pt --out results/sensitivity_US.csv
   python sensitivity.py --case UK --data data_sources.csv --model models/dqn_UK.pt --out results/sensitivity_UK.csv

   python robustness.py --case US --data data_sources.csv --model models/dqn_US.pt --out results/robustness_US.csv
   python robustness.py --case UK --data data_sources.csv --model models/dqn_UK.pt --out results/robustness_UK.csv
   ```

5. **Evaluate LLM explanations (automatic scoring)**
   ```bash
   python llm_eval.py --ref datasets/llm_eval_reference.csv                       --pred llm_eval_outputs.csv                       --out results/llm_eval_scores.csv
   ```

6. **Record compute runtimes (for audit)**
   ```bash
   python compute_runtimes.py --quick --episodes 120 --seed 42      --out results/compute_runtimes_times.json
   ```

7. **One‑click script (optional)**
   ```bash
   bash run.sh
   ```

## Data provenance
- Inputs are from publicly available sources (US DOE/EIA CBECS, Ofgem, UK Met Office, EPA eGRID, UK DESNZ, RICS/ICE/BCIS, OSHA/HSE, UKGBC/WorldGBC).  
- The `data_sources_clustered.xlsx` contains clustered samples (n≈200 per variable per case) centred on the characteristic values reported in the paper; variability is documented in the `VARIABILITY` sheet. The builder aggregates to medians to write `data_sources.csv`.

## Randomness and seeds
- We set `seed=42` for both environment and agent RNGs.  
- Results are reported as mean ± st.dev. across `n ≥ 5` seeds in the paper; numbers will vary within reported ranges when re‑running.

## Compute disclosure
- CPU‑only reproduction is feasible. Training for **120 episodes** (sanity check) finishes in seconds on x86_64; full **5000‑episode** runs complete within typical desktop times.  
- Please export the JSON from `compute_runtimes.py` to disclose your hardware (CPU/GPU, RAM) and wall‑clock per experiment.

## Anonymity and artefact naming
- Do not include author names, affiliations, or identifying URLs in the artefacts.  
- Keep file names exactly as referenced above for the automated checks by LLM and human reviewers.

## Any deviation from the paper
- If you customise parameters in `data_sources_clustered.xlsx` or change hyper‑parameters in `dqn_agent.py`, please note this in a short `results/README.txt` and re‑export the trajectories so that numbers still align with your claims.

## Known limitations for reproduction
- Social value multipliers come from secondary sources; exact magnitudes may differ by geography and update frequency.
- Transition dynamics are stylised; thus absolute values can differ, but relative improvements and Pareto trade‑offs should match qualitatively.
