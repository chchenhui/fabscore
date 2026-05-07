# Reproduction Guide for Canine Cardiac Health Assessment Research

This README provides a comprehensive guide to reproduce the experiments, tables, and figures presented in our research on Multi-Task Learning for Canine Cardiomegaly Classification and VHS Keypoint Detection.

## Project Structure

The primary project directory is named `Project`. Its structure is as follows:

```
Project/
├── code/
│   ├── run_experiments.py
│   ├── create_tables.py
│   ├── create_figures.py
│   ├── plot_confusion_matrix.py
│   ├── requirements.txt
│   ├── smoke_test.py
│   └── ablation_scripts/
│       └── ... (individual ablation scripts)
├── Data/
│   ├── metadata.json
│   ├── Test/
│   │   ├── Images/
│   │   ├── Images_classes/
│   │   └── Labels/
│   ├── Train/
│   │   ├── Images/
│   │   ├── Images_classes/
│   │   └── Labels/
│   └── Valid/
│       ├── Images/
│       ├── Images_classes/
│       └── Labels/
├── results/
│   ├── results.json
│   ├── figures/ (example)
│   └── ... (experiment outputs)
```


## 1. Environment Setup

To set up the necessary environment, we used Python 3.7.4 for the experiments. All required Python packages are listed in `requirements.txt`.

```bash
# Navigate to the primary project directory (e.g., 'Project')
# cd /path/to/your/Project

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r code/requirements.txt
```

## 2. Data Preparation

Our research utilizes the `DogHeart` dataset, as described in "Regressive vision transformer for dog cardiomegaly assessment" (https://www.nature.com/articles/s41598-023-50063-x). This dataset is an existing collection of canine thoracic X-rays with corresponding keypoint and VHS annotations.

**Important: Data Request and Download**

The dataset used in this research is proprietary and requires a formal request for access.

Once access is granted and the data is downloaded, ensure the `Data` folder is structured as follows and placed directly in the root of your `Project` setup:

```
Data/
├── metadata.json       # Metadata file for the dataset
├── Test/
│   ├── Images/            # Contains X-ray image files (e.g., .png)
│   ├── Images_classes/    # Contains subfolders for classification labels (e.g., Normal, Large, Small)
│   └── Labels/            # Contains .mat files with ground truth keypoints and VHS scores
├── Train/
│   ├── Images/
│   ├── Images_classes/
│   └── Labels/
└── Valid/
    ├── Images/
    ├── Images_classes/
    └── Labels/
```

**Note:** The `Data` folder should be placed directly in the root of your `Project` setup.

## 3. Running Experiments

The main entry point for running the baseline experiment is `code/run_experiments.py`. This script trains the full MT-ViT-CCHA model and evaluates its performance.

```bash
# Ensure you are in the primary project directory (e.g., 'Project')

# Run the baseline experiment (full model)
python code/run_experiments.py
```

For ablation studies, separate scripts are provided in the `code/ablation_scripts/` folder. Each script corresponds to a specific ablation experiment.

```bash
# Ensure you are in the primary project directory (e.g., 'Project')

# Example: Run the classification-only ablation study
python code/ablation_scripts/ablation_classification_only.py

# Example: Run the no-cross-attention ablation study
python code/ablation_scripts/ablation_no_cross_attention.py

# You can find other ablation scripts in the 'ablation_scripts' folder.
# For a quick verification (smoke test) of any script:
python code/run_experiments.py --smoke-test # For baseline
```

Experiment results (metrics and training history) will be saved as `results.json` files within timestamped subdirectories in the `results/` and `ablation_results/` folders (relative to the `Project` root).

### Running Synthetic Smoke Test

For a quick verification of the pipeline without requiring the actual dataset, you can run the synthetic smoke test. This test uses randomly generated data and runs for a single epoch.

```bash
python code/smoke_test_synthetic.py
```

This script will create a `results_synthetic_YYYYMMDD_HHMMSS` folder in the root directory with dummy results.

## 4. Generating Tables

After running the experiments, you can generate the performance comparison and ablation study tables using `code/create_tables.py`.

```bash
# Ensure you are in the primary project directory (e.g., 'Project')

# Generate tables
python code/create_tables.py
```

This will generate `performance_comparison.json`, `multi_task_ablation.json`, `cross_attention_kp_head_ablation.json`, and `loss_ablation.json` in the `results/` directory (relative to the `Project` root).

## 5. Generating Figures

To generate the figures presented in the paper, including the loss/accuracy curves and model architecture diagram, run `code/create_figures.py`.

```bash
# Ensure you are in the primary project directory (e.g., 'Project')

# Generate figures
python code/create_figures.py
```

This will save the figures (e.g., `curves.png`, `model_architecture.png`) in the `paper/Figures/` directory (relative to the `Project` root).

## 6. Generating Confusion Matrices

To generate the combined confusion matrix figure for validation and test datasets, run `code/plot_confusion_matrix.py`.

```bash
# Ensure you are in the primary project directory (e.g., 'Project')

# Generate confusion matrices
python code/plot_confusion_matrix.py
```

This will save the combined confusion matrix figure (e.g., `combined_confusion_matrix.png`) in the `paper/Figures/` directory (relative to the `Project` root).

## 7. Advanced Analysis

This section describes how to run the multi-seed experiments to evaluate the robustness of the model.

### Multi-Seed Experiments

To run the experiment with multiple random seeds, you can use the `run_parallel.bat` script. This script will run the experiment 5 times with different seeds.

```bash
# Navigate to the code directory
cd code

# Run the multi-seed experiment
run_parallel.bat
```

This will create a new directory `../Results/multiseed_finetune` and save the results of each run in a separate JSON file.

**Note:** The `run_parallel.bat` script is configured to run the fine-tuning experiment for 50 epochs, starting from the `best_model.pth` in the `Results` directory. You can edit the script to change the number of epochs, the seeds, or the output directory.

### Analyzing Multi-Seed Results

After the multi-seed experiment is finished, you can use the `analyze_results.py` script to calculate the mean and standard deviation of the test accuracy.

```bash
# Navigate to the code directory
cd code

# Analyze the results
python analyze_results.py
```

This will print the mean and standard deviation of the test accuracy from the multi-seed experiment.