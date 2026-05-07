# Fairness-Aware Classification with Synthetic Tabular Data - Code

## Overview
This directory contains the implementation for fairness-aware classification experiments using synthetic tabular data.

## Files Description

- `dataset.py`: Synthetic dataset generation with configurable bias injection
- `model.py`: Implementation of baseline and fairness-aware classification models
- `train.py`: Training pipeline for all models including ablation studies
- `evaluate.py`: Comprehensive fairness and accuracy evaluation metrics
- `run_experiments.py`: Main experiment runner and analysis pipeline
- `create_figures.py`: Visualization generation for paper figures
- `simple_experiment.py`: Lightweight version for systems without full dependencies

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Complete Experiment
```bash
python run_experiments.py
```

### Generate Figures
```bash
python create_figures.py
```

### Run Individual Components
```bash
python dataset.py    # Test dataset generation
python model.py      # Test model implementations
python train.py      # Run training pipeline
python evaluate.py   # Test evaluation metrics
```

## Models Implemented

### Baseline Models
- **Logistic Regression**: Standard logistic regression classifier
- **Random Forest**: Ensemble method with 100 trees

### Fairness-Aware Models
- **Fairness-Aware Logistic Regression**: Uses group reweighting strategy
- **Adversarial Debiasing Neural Network**: Minimax optimization with adversarial training

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

## Dependencies

Required Python packages:
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- torch>=1.9.0 (for adversarial models)

## System Requirements

- Python 3.8+
- 2GB RAM
- ~100MB disk space
- Runtime: <10 minutes for complete experiment

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Reduce `n_samples` for constrained systems
3. **PyTorch issues**: Install CPU-only version if no GPU

### Performance Notes

- All experiments use CPU computation
- Results are deterministic with fixed random seeds
- Synthetic data generation is lightweight and fast