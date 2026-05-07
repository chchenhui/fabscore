#!/usr/bin/env python3
"""
Main experiment runner for Fairness-Aware Classification with Synthetic Tabular Data
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime

from dataset import SyntheticFairnessDataset
from train import ExperimentRunner
from evaluate import ModelEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

def setup_experiment_environment():
    """Setup directories and logging for the experiment"""
    results_dir = "../results"
    figures_dir = f"{results_dir}/figures"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    return results_dir, figures_dir

def run_single_experiment(n_samples=1000, bias_strength=0.3, random_state=42):
    """Run a single experiment with given parameters"""
    print(f"Running experiment: n_samples={n_samples}, bias_strength={bias_strength}")

    experiment = ExperimentRunner(
        n_samples=n_samples,
        bias_strength=bias_strength,
        random_state=random_state
    )

    results = experiment.run_full_experiment()
    return results

def run_bias_strength_analysis(n_samples=1000, bias_strengths=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """Analyze performance across different bias strengths"""
    print("Running bias strength analysis...")

    bias_analysis_results = []

    for bias_strength in bias_strengths:
        print(f"\nAnalyzing bias strength: {bias_strength}")

        experiment = ExperimentRunner(
            n_samples=n_samples,
            bias_strength=bias_strength,
            random_state=42
        )

        results = experiment.run_full_experiment()

        # Extract key metrics for each model
        for result in results['detailed_results']:
            bias_analysis_results.append({
                'bias_strength': bias_strength,
                'model': result['model_name'],
                'accuracy': result['accuracy'],
                'demographic_parity': result['demographic_parity'],
                'equal_opportunity': result['equal_opportunity'],
                'equalized_odds': result['equalized_odds']
            })

    return pd.DataFrame(bias_analysis_results)

def save_experiment_results(results, results_dir):
    """Save all experiment results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save main results
    results['results_df'].to_csv(f"{results_dir}/model_comparison_{timestamp}.csv", index=False)

    # Save ablation results
    ablation_df = pd.DataFrame(results['ablation_results'])
    ablation_df.to_csv(f"{results_dir}/ablation_study_{timestamp}.csv", index=False)

    # Save dataset statistics
    with open(f"{results_dir}/dataset_stats_{timestamp}.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        stats = results['dataset_stats'].copy()
        for key, value in stats.items():
            if isinstance(value, np.number):
                stats[key] = float(value)
            elif isinstance(value, pd.DataFrame):
                stats[key] = value.to_dict()
        json.dump(stats, f, indent=2)

    # Save detailed results as JSON
    detailed_results_json = []
    for result in results['detailed_results']:
        json_result = {}
        for key, value in result.items():
            if key == 'confusion_matrices':
                # Convert confusion matrices to lists
                json_result[key] = {k: v.tolist() for k, v in value.items()}
            elif isinstance(value, np.number):
                json_result[key] = float(value)
            else:
                json_result[key] = value
        detailed_results_json.append(json_result)

    with open(f"{results_dir}/detailed_results_{timestamp}.json", 'w') as f:
        json.dump(detailed_results_json, f, indent=2)

    print(f"Results saved with timestamp: {timestamp}")
    return timestamp

def create_summary_metrics_file(results, results_dir, timestamp):
    """Create a summary metrics.json file as required"""

    # Extract key metrics from the best performing models
    results_df = results['results_df']

    # Find best baseline and best fairness model
    baseline_models = results_df[~results_df['model'].str.contains('Fairness|Adversarial')]
    fairness_models = results_df[results_df['model'].str.contains('Fairness|Adversarial')]

    best_baseline = baseline_models.loc[baseline_models['accuracy'].idxmax()] if len(baseline_models) > 0 else None
    best_fairness = fairness_models.loc[fairness_models['demographic_parity'].idxmin()] if len(fairness_models) > 0 else None

    summary_metrics = {
        "experiment_info": {
            "timestamp": timestamp,
            "dataset_size": int(results['dataset_stats']['group_statistics']['count'].sum()),
            "bias_strength": float(results['dataset_stats']['bias_strength']),
            "bias_difference": float(results['dataset_stats']['bias_difference'])
        },
        "best_baseline_model": {
            "name": best_baseline['model'] if best_baseline is not None else "N/A",
            "accuracy": float(best_baseline['accuracy']) if best_baseline is not None else 0.0,
            "demographic_parity": float(best_baseline['demographic_parity']) if best_baseline is not None else 0.0,
            "equal_opportunity": float(best_baseline['equal_opportunity']) if best_baseline is not None else 0.0
        } if best_baseline is not None else {},
        "best_fairness_model": {
            "name": best_fairness['model'] if best_fairness is not None else "N/A",
            "accuracy": float(best_fairness['accuracy']) if best_fairness is not None else 0.0,
            "demographic_parity": float(best_fairness['demographic_parity']) if best_fairness is not None else 0.0,
            "equal_opportunity": float(best_fairness['equal_opportunity']) if best_fairness is not None else 0.0
        } if best_fairness is not None else {},
        "all_models_summary": {
            "num_models_evaluated": len(results_df),
            "avg_accuracy": float(results_df['accuracy'].mean()),
            "avg_demographic_parity": float(results_df['demographic_parity'].mean()),
            "avg_equal_opportunity": float(results_df['equal_opportunity'].mean())
        }
    }

    with open(f"{results_dir}/metrics.json", 'w') as f:
        json.dump(summary_metrics, f, indent=2)

    print("Summary metrics saved to metrics.json")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "torch>=1.9.0",
        "jupyter>=1.0.0"
    ]

    with open("../code/requirements.txt", 'w') as f:
        f.write("\n".join(requirements))

    print("Requirements.txt created")

def create_readme():
    """Create README.md file for the code directory"""
    readme_content = """# Fairness-Aware Classification with Synthetic Tabular Data

## Overview
This directory contains the implementation for fairness-aware classification experiments using synthetic tabular data.

## Files Description

- `dataset.py`: Synthetic dataset generation with configurable bias injection
- `model.py`: Implementation of baseline and fairness-aware classification models
- `train.py`: Training pipeline for all models including ablation studies
- `evaluate.py`: Comprehensive fairness and accuracy evaluation metrics
- `run_experiments.py`: Main experiment runner and analysis pipeline
- `requirements.txt`: Python package dependencies

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the complete experiment:
   ```bash
   python run_experiments.py
   ```

3. Run individual components:
   ```bash
   python dataset.py    # Test dataset generation
   python model.py      # Test model implementations
   python train.py      # Run training pipeline
   python evaluate.py   # Test evaluation metrics
   ```

## Models Implemented

### Baseline Models
- Logistic Regression
- Random Forest

### Fairness-Aware Models
- Fairness-Aware Logistic Regression (with reweighting)
- Adversarial Debiasing Neural Network

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Demographic Parity**: |P(ŷ=1|a=0) - P(ŷ=1|a=1)|
- **Equal Opportunity**: |P(ŷ=1|y=1,a=0) - P(ŷ=1|y=1,a=1)|
- **Equalized Odds**: max(|TPR_diff|, |FPR_diff|)

## Output

Results are saved to `../results/`:
- `model_comparison.csv`: Comparison of all models
- `ablation_study.csv`: Fairness penalty ablation results
- `metrics.json`: Summary metrics
- `figures/`: Generated plots and visualizations

## Configuration

Key parameters can be modified in `run_experiments.py`:
- `n_samples`: Dataset size (default: 1000)
- `bias_strength`: Bias injection strength (default: 0.3)
- `random_state`: Random seed for reproducibility (default: 42)
"""

    with open("../code/README.md", 'w') as f:
        f.write(readme_content)

    print("README.md created")

def main():
    """Main experiment execution function"""
    print("=" * 80)
    print("FAIRNESS-AWARE CLASSIFICATION EXPERIMENT")
    print("Synthetic Tabular Data Analysis")
    print("=" * 80)

    # Setup environment
    results_dir, figures_dir = setup_experiment_environment()

    # Create supporting files
    create_requirements_file()
    create_readme()

    # Run main experiment
    print("\n1. Running main experiment...")
    results = run_single_experiment(n_samples=1000, bias_strength=0.3, random_state=42)

    # Save results
    print("\n2. Saving results...")
    timestamp = save_experiment_results(results, results_dir)
    create_summary_metrics_file(results, results_dir, timestamp)

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Results saved to: {results_dir}")
    print(f"Number of models evaluated: {len(results['detailed_results'])}")
    print(f"Dataset bias strength: {results['dataset_stats']['bias_strength']}")
    print(f"Dataset bias difference: {results['dataset_stats']['bias_difference']:.3f}")

    # Show top performing models
    results_df = results['results_df']
    print("\nTop models by accuracy:")
    top_acc = results_df.nlargest(3, 'accuracy')[['model', 'accuracy', 'demographic_parity']]
    print(top_acc.to_string(index=False))

    print("\nTop models by fairness (lowest demographic parity):")
    top_fair = results_df.nsmallest(3, 'demographic_parity')[['model', 'accuracy', 'demographic_parity']]
    print(top_fair.to_string(index=False))

if __name__ == "__main__":
    main()