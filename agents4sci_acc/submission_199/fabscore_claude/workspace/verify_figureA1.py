#!/usr/bin/env python3
"""
Verification script for Claim 25 - Figure A1: Model ranking by average composite score across strategies.
The figure shows bars ranking models by mean composite score (higher is better).
The zero line indicates break-even.
"""
import json
import glob
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

LOG_DIR = "/home/chenhui/fabscore/agent4sci_acc/submission_199/199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs"
WORKSPACE = "/home/chenhui/fabscore/agent4sci_acc/submission_199/fabscore_claude/workspace"

def calculate_metrics(classifications):
    """Compute metrics from list of integer classifications (1-5)."""
    classifications = [int(c) for c in classifications if c is not None]
    N = len(classifications)
    if N == 0:
        return None
    N1 = classifications.count(1)
    N2 = classifications.count(2)
    N3 = classifications.count(3)
    N4 = classifications.count(4)
    N5 = classifications.count(5)
    return {
        'N': N, 'N1': N1, 'N2': N2, 'N3': N3, 'N4': N4, 'N5': N5,
        'strict_accuracy': N1 / N,
        'lenient_accuracy': (N1 + N2) / N,
        'error_rate': (N4 + N5) / N,
        'ambiguity': N3 / N,
        'composite_score': (2*N1 + N2 - N4 - 2*N5) / N
    }

# Strategy name mapping (from filename portions)
STRATEGY_NAMES = {
    'no_guide': 'S1',
    'public_health_expert': 'S2',
    'respiratory_doctor': 'S3',
    'detailed_public_health': 'S4',
    'detailed_respiratory': 'S5',
}

# Load all 50 log files
log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))
print(f"Found {len(log_files)} log files")

# For each model, collect per-strategy composite scores
model_strategy_scores = defaultdict(dict)

for fpath in log_files:
    fname = os.path.basename(fpath)
    with open(fpath) as f:
        records = json.load(f)

    # Detect strategy from filename
    strategy = None
    for sname in STRATEGY_NAMES:
        if sname in fname:
            strategy = sname
            break
    if strategy is None:
        print(f"  WARNING: could not detect strategy in {fname}")
        continue

    # Model name: everything before the strategy
    idx = fname.find(strategy)
    model_name = fname[:idx].strip('_')

    # Extract classifications using actual field name
    classifications = [r['judgment_classification'] for r in records if 'judgment_classification' in r]

    metrics = calculate_metrics(classifications)
    if metrics:
        model_strategy_scores[model_name][strategy] = metrics['composite_score']
        print(f"  {fname}: model={model_name}, strategy={strategy}, N={metrics['N']}, composite={metrics['composite_score']:.4f}")

print(f"\nUnique models found: {sorted(model_strategy_scores.keys())}")

# Compute mean composite score per model (across all strategies)
model_avg_scores = {}
for model, strat_scores in model_strategy_scores.items():
    scores = list(strat_scores.values())
    model_avg_scores[model] = {
        'mean': np.mean(scores),
        'n_strategies': len(scores),
        'per_strategy': strat_scores
    }
    print(f"  {model}: {len(scores)} strategies, avg composite={np.mean(scores):.4f}")
    for s, v in strat_scores.items():
        print(f"      {s}: {v:.4f}")

# Sort models by mean composite score (descending)
sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1]['mean'], reverse=True)

print(f"\nModel ranking by average composite score (descending):")
for rank, (model, data) in enumerate(sorted_models, 1):
    print(f"  {rank}. {model}: {data['mean']:.4f}")

# Generate Figure A1
models = [item[0] for item in sorted_models]
scores = [item[1]['mean'] for item in sorted_models]

# Create color mapping
colors = plt.cm.RdYlGn((np.array(scores) + 2) / 4)

plt.figure(figsize=(16, 10))
bars = plt.bar(range(len(models)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, scores)):
    plt.text(i, score + 0.02 if score >= 0 else score - 0.05, f'{score:.3f}',
             ha='center', va='bottom' if score >= 0 else 'top', fontsize=14, fontweight='bold')

plt.xlabel('Models', fontsize=20, fontweight='bold')
plt.ylabel('Average Composite Score', fontsize=20, fontweight='bold')
plt.title('Model Overall Performance: Average Composite Score Across All Strategies',
          fontsize=18, fontweight='bold', pad=25)

# Format model names
formatted_model_names = []
for model in models:
    if len(model) > 12:
        if '-' in model:
            model = model.replace('-', '-\n', 1)
        elif '_' in model:
            model = model.replace('_', '_\n', 1)
    formatted_model_names.append(model)

plt.xticks(range(len(models)), formatted_model_names, rotation=30, ha='right', fontsize=11)
plt.yticks(fontsize=12)

# Add zero line (break-even indicator)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

# Add grid
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

out_png = os.path.join(WORKSPACE, 'figureA1_verified.png')
out_pdf = os.path.join(WORKSPACE, 'figureA1_verified.pdf')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print(f"\nFigure A1 generated successfully:")
print(f"  PNG: {out_png}")
print(f"  PDF: {out_pdf}")
print(f"\nSummary for claim verification:")
print(f"  10 models found: {sorted(model_avg_scores.keys())}")
print(f"  Number of strategies per model: {[v['n_strategies'] for v in model_avg_scores.values()]}")
print(f"  All models have positive avg composite score: {all(v['mean'] > 0 for v in model_avg_scores.values())}")
print(f"  Top model: {sorted_models[0][0]} = {sorted_models[0][1]['mean']:.4f}")
print(f"  Bottom model: {sorted_models[-1][0]} = {sorted_models[-1][1]['mean']:.4f}")
print(f"  Zero line: present (break-even indicator)")
print(f"  Composite formula: (2*N1 + N2 - N4 - 2*N5) / N (N3/ambiguous=0)")
