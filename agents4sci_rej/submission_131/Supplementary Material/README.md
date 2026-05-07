# Reproducibility Guide: Public Health Case Study

This repository provides code and instructions to reproduce the conceptual case study presented in the paper *Autonomous Scientific Experimentation Powered by Generative AI*. The case study investigates the relationship between vaccination coverage and influenza hospitalization rates using a synthetic dataset.

---

## 1. Requirements

- **Operating System**: Ubuntu 22.04 LTS (other OS also supported)
- **Python**: >= 3.10
- **Libraries**:
  - `numpy` (>= 1.26)
  - `matplotlib` (>= 3.8)
  - `scipy` (>= 1.11)

Install the required packages with:

```bash
pip install numpy matplotlib scipy
```

---

## 2. Running the Experiment

Save the following script as `script.py` and run it:

```bash
python script.py
```

This script will:
1. Generate a synthetic dataset with 20 data points (vaccination coverage between 40–90%).
2. Fit a linear regression model to estimate hospitalization rates.
3. Compute evaluation metrics:
   - Pearson correlation coefficient (r)
   - Mean Squared Error (MSE)
4. Plot the observed data and regression line and show the figure.

---

## 3. Expected Output

When running the script, the terminal will print something like:

```
Correlation coefficient r = -0.81
Mean Squared Error MSE = 0.12
```

A figure will also be shown, and the figure shows:
- **Blue marks**: observed synthetic data points.  
- **Red dashed line**: fitted regression model.  
- **Axes**: vaccination coverage (%) vs. influenza hospitalization rate (per 100,000).  

---

## 4. Reproducibility Statement

- The entire workflow is implemented in under **30 lines of Python code**.  
- No external datasets are required.  
- Results may vary slightly due to random noise generation, but the overall trend (strong negative correlation) will remain consistent.  
- The script executes in under **2 seconds** on a standard CPU laptop (Intel i7, 16 GB RAM).  

---

## 5. License and Usage

This code is provided for academic and research purposes.  
You are free to use, modify, and extend it in accordance with open science practices.
