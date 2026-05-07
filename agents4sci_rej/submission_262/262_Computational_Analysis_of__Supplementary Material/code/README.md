# Cryptographic Hash Function Analysis

This repository contains the complete implementation and experimental results for the research paper "Computational Analysis of Cryptographic Hash Function Performance and Security".

## Overview

This project provides a comprehensive analysis of four cryptographic hash functions:
- MD5 (128-bit)
- SHA-256 (256-bit) 
- SHA-3/Keccak-256 (256-bit)
- BLAKE2b (256-bit)

The analysis evaluates both performance characteristics and security properties across different input sizes and data patterns.

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

To reproduce all experimental results:

```bash
python3 code/run_experiments.py
```

This will:
- Generate 60,000 test vectors across multiple configurations
- Analyze performance and security properties of all hash functions
- Generate comprehensive results and visualizations
- Save results to `results/metrics.json`
- Create publication-quality figures in `results/figures/`

## Experimental Configuration

- **Data sizes**: 1KB, 10KB, 100KB, 1MB, 10MB
- **Data types**: random, structured, edge_case
- **Test iterations**: 1,000 per configuration
- **Total tests**: 60,000

## Results

The experiments generate:
- Performance metrics (throughput, processing time, memory usage)
- Security metrics (avalanche effect, collision resistance, distribution uniformity)
- Statistical analysis and comparisons
- Publication-quality visualizations

Key findings:
- SHA-256 achieves highest throughput (1,809 MB/s average)
- BLAKE2b shows best avalanche effect (99.52%)
- All algorithms demonstrate adequate distribution uniformity (~73%)
- MD5 shows reduced avalanche effect (25%) confirming vulnerabilities

## File Structure

```
├── code/
│   ├── hash_function_analysis.py  # Main analysis implementation
│   ├── run_experiments.py         # Experiment runner
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # This file
├── data/
│   └── metadata.json             # Research metadata
├── results/
│   ├── metrics.json              # Experimental results
│   ├── experiment_summary.txt    # Summary report
│   └── figures/                  # Generated visualizations
└── paper/
    └── main.tex                  # LaTeX paper
```

## Reproducibility

All experiments use a fixed random seed (42) for reproducibility. The complete analysis runs in approximately 9 minutes on standard hardware and generates deterministic results.

## Citation

If you use this code or results in your research, please cite:

```
AI Research Agent. (2025). Computational Analysis of Cryptographic Hash Function Performance and Security. 
In Proceedings of the 1st Open Conference of AI Agents for Science.
```

## License

This project is released under the MIT License. See LICENSE file for details.



