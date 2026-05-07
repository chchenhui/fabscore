# BiCA: Bidirectional Cognitive Adaptation Framework

This repository contains the complete implementation of the **Bidirectional Cognitive Adaptation (BiCA)** framework, a novel approach for human-AI collaboration that enables mutual adaptation through protocol learning, representation alignment, and instructor-guided interactions.

## 🎯 Overview

BiCA addresses the challenge of effective human-AI collaboration by implementing bidirectional adaptation mechanisms:

- **AI adapts to human communication patterns** through protocol learning
- **Humans adapt to AI capabilities** through instructor-guided interventions  
- **Shared representation alignment** minimizes cognitive gaps
- **Budget-constrained optimization** prevents cognitive drift

## 📁 Repository Structure

```
bica/
├── envs/                    # Environment implementations
│   ├── maptalk.py          # 8x8 gridworld with OOD variants
│   └── __init__.py
├── models/                  # Neural network models
│   ├── policy.py           # AI policy and value networks
│   ├── human_surrogate.py  # Human surrogate with protocol table
│   ├── protocol.py         # Gumbel-Softmax protocol generator
│   ├── rep_mapper.py       # Representation mapper T_ψ
│   ├── instructor.py       # Instructor model π_I
│   └── __init__.py
├── losses/                  # Loss function implementations
│   ├── ppo.py              # PPO with KL-to-prior regularization
│   ├── ib.py               # Information Bottleneck and MDL losses
│   ├── repgap.py           # Representation gap (Wasserstein + CCA)
│   └── __init__.py
├── eval/                    # Evaluation and metrics
│   ├── metrics_bas_ccm.py  # BAS and CCM metric computation
│   ├── ood_suites.py       # Out-of-distribution evaluation
│   └── __init__.py
├── viz/                     # Visualization tools
│   ├── plots.py            # Training curves, BAS radar, OOD bars
│   ├── analysis.py         # Statistical analysis and comparisons
│   └── __init__.py
├── latent_nav_lite/        # Auxiliary latent navigation experiment
│   ├── vae_model.py        # β-VAE and projection network
│   ├── navigator.py        # Latent space navigation
│   ├── data_loader.py      # dSprites and synthetic datasets
│   ├── evaluator.py        # Navigation evaluation
│   ├── ui_interface.py     # Interactive UI
│   └── __init__.py
├── configs/                 # Configuration files
│   ├── maptalk_main.yaml   # Main experiment configuration
│   ├── maptalk_ablation.yaml # Ablation study settings
│   └── baselines.yaml      # Baseline method configurations
├── seeds/                   # Reproducibility seeds
│   └── seed_list.txt
└── train_maptalk.py        # Main training script
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/bica.git
cd bica
```

2. Create and activate the environment (2-step process):

First, create the minimal conda environment with PyTorch. This should be fast.
```bash
conda env create -f environment.yml
conda activate bica
```

Second, install all other dependencies using pip.
```bash
pip install -r requirements.txt
```

For development installation:
```bash
pip install -e .
```

### Running the Main Experiment

#### Option 1: Individual Experiments

1. **Train BiCA on MapTalk:**
```bash
# Default epochs from config
python bica/train_maptalk.py --config bica/configs/maptalk_main.yaml

# Override with custom epochs
python bica/train_maptalk.py --config bica/configs/maptalk_main.yaml --epochs 10
```

2. **Run single-directional baseline:**
```bash
# Default epochs from config  
python bica/train_maptalk.py --config bica/configs/maptalk_one_way_baseline.yaml

# Override with custom epochs
python bica/train_maptalk.py --config bica/configs/maptalk_one_way_baseline.yaml --epochs 10
```

3. **Ablation studies:**
```bash
python run_experiment.py --experiment ablations --output_dir results/ablations
```

#### Option 2: Automated Baseline Comparison

**Quick comparison (5 epochs, ~15 minutes):**
```bash
python run_maptalk_comparison.py
```

**Medium comparison (10 epochs, ~30 minutes):**
```bash
python run_maptalk_comparison.py --epochs 10
```

**Full comparison (50 epochs, ~2.5 hours):**
```bash
python run_maptalk_comparison.py --epochs 50
```

This will automatically:
1. Run single-directional baseline experiment
2. Run BiCA co-alignment experiment  
3. Generate comprehensive comparison report with performance metrics

#### Epoch Parameter Guide

- **`--epochs`**: Overrides the episode count in configuration files
- **Default**: 5 epochs (160 episodes with batch_episodes=32)
- **Calculation**: `episodes = epochs × batch_episodes`
- **Recommended values**:
  - `--epochs 5`: Quick testing and development
  - `--epochs 10`: Medium-scale experiments
  - `--epochs 50`: Full-scale experiments for paper results

### Running the Latent Navigator Experiment

```python
from bica.latent_nav_lite import create_latent_navigator, create_navigator_ui

# Create navigator
config = {...}  # Your configuration
navigator = create_latent_navigator(config)

# Interactive UI
ui = create_navigator_ui(navigator, config)
ui.run()
```

## 📊 Key Features

### 1. Bidirectional Adaptation
- **Protocol Learning**: AI learns compact communication protocols using Gumbel-Softmax
- **Human Adaptation**: GRU-based surrogate models human protocol learning
- **Instructor Interventions**: Automated teaching based on uncertainty and errors

### 2. Representation Alignment
- **Wasserstein Distance**: Minimizes distributional gaps between human and AI representations
- **CCA Correlation**: Maximizes canonical correlations for better alignment
- **Differentiable Optimization**: End-to-end training with gradient-based updates

### 3. Budget-Constrained Learning
- **KL Regularization**: Prevents excessive drift from prior policies
- **Dual Variable Updates**: Automatically adjusts regularization weights
- **Intervention Costs**: Balances teaching effectiveness with fatigue

### 4. Comprehensive Evaluation
- **BAS Metrics**: 5-component bidirectional alignment score
- **CCM Metrics**: Cognitive complementarity measurement
- **OOD Robustness**: Performance across distribution shifts
- **Statistical Analysis**: Rigorous comparison with baselines

## 🧪 Experiments

### MapTalk Environment
- **Grid Size**: 8×8 with configurable obstacles
- **Observations**: AI (3×3 egocentric) vs Human (full map)
- **Communication**: Bidirectional discrete messages
- **OOD Variants**: Higher obstacles, sensor noise, structural patterns

### Latent Navigator
- **Datasets**: dSprites, synthetic geometric shapes
- **Models**: β-VAE (latent_dim=16) + 2D projection network
- **Task**: Navigate latent space to maximize hidden scoring function
- **Metrics**: Best score, novelty, cognitive gains

## 📈 Expected Results

Based on the paper specifications:

### MapTalk (In-Distribution)
- **Steps**: 20-35% reduction vs baselines
- **Tokens**: 40-60% reduction in communication cost
- **BAS Score**: +0.10 absolute improvement vs One-Way baseline

### MapTalk (Out-of-Distribution)
- **Success Rate**: 8-15% improvement on OOD variants
- **Collision Rate**: >20% reduction
- **Representation Compatibility**: Improved OT distance and CCA correlation

### CCM Trajectory
- **Initial Phase**: Dip to 0.3-0.5 range as agents adapt
- **Stabilization**: Convergence to stable complementarity

## 🔧 Configuration

The framework uses YAML configuration files with the following key sections:

```yaml
# Environment settings
env:
  grid_size: 8
  obstacle_rate: [0.2, 0.3]
  max_steps: 60

# Model architecture
model:
  embed_dim: 128
  code_dim: 16
  gru_hidden: 128

# Training parameters
train:
  episodes: 2000
  batch_episodes: 32
  lr: 3e-4

# Regularization coefficients
regularizers:
  lambda_A: 0.02    # AI KL budget
  lambda_H: 0.01    # Human KL budget
  beta_ib: 1.0      # Information bottleneck
  mu_rep: 0.1       # Representation gap
  kappa_teach: 0.05 # Instructor cost
```

## 📊 Visualization

The framework includes comprehensive visualization tools:

```python
from bica.viz import create_visualizer, create_experiment_analyzer

# Training visualizations
visualizer = create_visualizer(config)
fig = visualizer.plot_training_curves(training_data)
fig = visualizer.plot_bas_radar(bas_scores, baseline_scores)
fig = visualizer.plot_ood_performance(ood_results)

# Statistical analysis
analyzer = create_experiment_analyzer(config)
analysis = analyzer.analyze_experiment_results(all_results)
report = analyzer.generate_analysis_report(analysis)
```

## 🧮 Metrics

### BAS (Bidirectional Alignment Score)
- **MP**: Mutual Predictability between agents
- **BS**: Bidirectional Steerability via protocol perturbation
- **RC**: Representational Compatibility (Wasserstein + CCA)
- **SS**: Shift-Robust Safety on OOD data
- **CE**: Cognitive Offloading Efficiency

### CCM (Cognitive Complementarity Metric)
- **Diversity**: HSIC-based measure of agent differences
- **Synergy**: Team performance beyond individual capabilities

## 🔬 Baselines

The repository implements several baseline methods:

1. **One-Way (RLHF-style)**: Unidirectional AI adaptation
2. **No-Budget**: Remove KL penalties and representation gap
3. **No-IB**: Disable information bottleneck regularization
4. **No-Mapper**: Remove representation alignment
5. **No-Teacher**: Disable instructor interventions

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{bica2024,
  title={Bidirectional Cognitive Adaptation: A Framework for Human-AI Collaboration},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues and Support

For bugs, feature requests, or questions:
- Open an [issue](https://github.com/your-repo/bica/issues)
- Check our [FAQ](docs/FAQ.md)
- Review the [troubleshooting guide](docs/TROUBLESHOOTING.md)

## 📚 Documentation

Additional documentation is available in the `docs/` directory:
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Extending BiCA](docs/extending.md)
- [Performance Tuning](docs/performance.md)

## 🔍 Reproducibility

All experiments use fixed random seeds and deterministic operations. The `seeds/seed_list.txt` file contains the seeds used in the paper. To reproduce exact results:

1. Use the provided configuration files
2. Set the random seeds from `seed_list.txt`
3. Use the same software versions (see `environment.yml`)
4. Follow the experimental protocol in the paper

## ⚡ Performance Tips

- Use GPU acceleration: Set `device: cuda` in config
- Enable mixed precision: Set `mixed_precision: true`
- Adjust batch sizes based on available memory
- Use parallel data loading: Set `num_workers > 0`
- Monitor training with Weights & Biases: Set `use_wandb: true`

---

**Built with ❤️ for advancing human-AI collaboration research**