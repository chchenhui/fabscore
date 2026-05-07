# Energy-Guided Program Selection Experiments

This repository contains the code, data, and scripts for the paper "Green by Design: Energy-Guided Reranking of LLM-Generated Programs," accepted at the Agents4Science 2025 conference. The goal of the study is to evaluate how large language models (LLMs) generate functionally correct Python programs and how energy consumption can guide the selection of the most efficient candidate.

## Usage Overview

### Phase 1: Setup

1. Python 3.10.

2. Install dependencies and setup environment:

   Open ollama and load model:

   ```
   ollama serve
   ollama pull codellama:70b-instruct
   ```

   In another terminal:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel
   
   
   # ensure you're in the repo root and .venv is activated
   pip install -r requirements.txt
   
   bash scripts/setup.sh
   bash scripts/pin_threads.sh
   
   python scripts/hw_report.py
   python scripts/sanity_check.py
   



### Phase 2: Generate Candidate & Correctness Filtering

Generate candidates using codellama:70b-instruct (4Q_0)

Filter generated candidates based on correctness tests.

Saves filtered candidates and logs per-task results.

```bash
cd phase2_candidate_generation
python generate_candidates.py
python filter_candidates.py
```

Due to Windows filename restrictions, There are slightly modification ("codellama:70b-instruct"->"codellama_70b-instruct") for some folder names in the compressed archive. Please revise the folder names or code to make it consistent when running the scripts after this phase.

The necessary output files from phases 2 to 4 are copied to the folder of each subsequent phase.

### Phase 3: Measurement

```bash
cd phase3_measurement
python phase3_runner_patched.py \     
  --filtered_dir ./filtered_candidates-codellama:70b-instruct-v5 \               
  --out_dir ./outputs/phase3 \
  --scales 10000 100000 1000000 \
  --runs 5 \
  --timeout 600


python aggregate_multirun_results.py \
    --run_dirs ./outputs/phase3 ./output/phase3_rerun ./output/phase3_rerun_t10 \ #If rerun some specific tasks, add all output path from last step to --run_dirs.
    --output_csv ./final_summary.csv    


```

In our experiment, there are some bugs in task 3 and task 10. We asked LLMs to fix it and reran the task 3 and task 10 measurement.
The output in phase3_rerun_t3 and phase3_rerun_t10 are the correct output for task 3 and task 10 in phase 3, others in phase3 are correct output for other tasks in phase 3.

Measures energy (kWh) and runtime for each candidate at multiple input scales.

Logs median and standard deviation values per candidate.

Generates per-task results and an aggregated summary CSV.

Phase 4: Candidate Selection

```bash
cd phase4_reranking 
python phase4_reranker.py \           
    --input_csv ./final_summary.csv \
    --output_dir ./phase4_analysis
```

Performs energy-guided reranking of candidates.

Compares selections against top-1 and best-time baselines.

Produces per-task and aggregate evaluation metrics.

Phase 5: Evaluation

To get final results:

```bash
cd phase5_evaluation 
python phase5_evaluation.py \         
    --phase3_dirs ./outputs/phase3 ./outputs/phase3_rerun ./outputs/phase3_rerun_t10 \ #If rerun some specific tasks, add all output path from last step to --phase3_dirs.
    --phase4_dir ./phase4_analysis \
    --candidates_dir ./filtered_candidates-codellama:70b-instruct-v5 \
    --output_dir ./phase5_evaluation

```



Reproducibility
All raw candidates (.py files), test suites, and measurement scripts are included.

Metadata logs (candidate ID, source hash) are available for all valid candidates.



