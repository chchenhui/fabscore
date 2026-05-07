#!/usr/bin/env python3
"""
Verification script for Claim 21 - Figure 1: Aggregate performance across all 50 model-prompt conditions.
This script reads the log files using the actual field name 'judgment_classification'
and generates the figure to verify the claim.
"""
import json
import glob
import os
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

# Load all 50 log files
log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))
print(f"Found {len(log_files)} log files")

model_classifications = defaultdict(list)
all_classifications = []

for fpath in log_files:
    fname = os.path.basename(fpath)
    with open(fpath) as f:
        records = json.load(f)

    # Extract model name from filename
    name = fname.replace('.json', '')
    parts = name.split('_')
    strategy_keywords = ['detailed', 'no', 'public', 'respiratory']
    strategy_start = -1
    for i, part in enumerate(parts):
        if part in strategy_keywords:
            strategy_start = i
            break
    if strategy_start > 0:
        model_name = '_'.join(parts[:strategy_start])
    else:
        model_name = name

    # Extract classifications using actual field name
    classifications = [r['judgment_classification'] for r in records if 'judgment_classification' in r]
    print(f"  {fname}: model={model_name}, records={len(records)}, classifications={len(classifications)}")

    model_classifications[model_name].extend(classifications)
    all_classifications.extend(classifications)

print(f"\nTotal classifications: {len(all_classifications)}")

# Compute global metrics
global_metrics = calculate_metrics(all_classifications)
print(f"\nGlobal metrics:")
print(f"  Strict Accuracy: {global_metrics['strict_accuracy']*100:.1f}%")
print(f"  Lenient Accuracy: {global_metrics['lenient_accuracy']*100:.1f}%")
print(f"  Error Rate: {global_metrics['error_rate']*100:.1f}%")
print(f"  Ambiguity: {global_metrics['ambiguity']*100:.2f}%")
print(f"  Composite Score: {global_metrics['composite_score']:.4f}")

# Compute per-model metrics
model_data = []
for model_name, cls_list in model_classifications.items():
    m = calculate_metrics(cls_list)
    if m:
        model_data.append({
            'model': model_name,
            'strict_accuracy': m['strict_accuracy'] * 100,
            'lenient_accuracy': m['lenient_accuracy'] * 100,
            'error_rate': m['error_rate'] * 100,
            'ambiguity': m['ambiguity'] * 100,
            'composite_score': m['composite_score'],
            'total_samples': m['N']
        })

# Sort by composite score
model_data.sort(key=lambda x: x['composite_score'], reverse=True)

print(f"\nPer-model metrics (sorted by composite score):")
for d in model_data:
    print(f"  {d['model']}: strict={d['strict_accuracy']:.1f}%, lenient={d['lenient_accuracy']:.1f}%, "
          f"error={d['error_rate']:.1f}%, composite={d['composite_score']:.4f}")

# Insert "all models" as first
overall_data = {
    'model': 'all models',
    'strict_accuracy': global_metrics['strict_accuracy'] * 100,
    'lenient_accuracy': global_metrics['lenient_accuracy'] * 100,
    'error_rate': global_metrics['error_rate'] * 100,
    'ambiguity': global_metrics['ambiguity'] * 100,
    'composite_score': global_metrics['composite_score'],
    'total_samples': global_metrics['N']
}
model_data.insert(0, overall_data)

# Generate Figure 1
models = [d['model'] for d in model_data]
strict_acc = [d['strict_accuracy'] for d in model_data]
lenient_acc = [d['lenient_accuracy'] for d in model_data]
error_rate = [d['error_rate'] for d in model_data]
composite_scores = [d['composite_score'] for d in model_data]
ambiguity = [d['ambiguity'] for d in model_data]

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(20, 10))

ax1 = plt.gca()
ax2 = ax1.twinx()

x = np.arange(len(models))
width = 0.25

bars1 = ax1.bar(x - width, strict_acc, width, label='Strict %', color='#FF8C00', alpha=0.8)
bars2 = ax1.bar(x, lenient_acc, width, label='Lenient %', color='#1E90FF', alpha=0.8)
bars3 = ax1.bar(x + width, error_rate, width, label='Error %', color='#2E8B57', alpha=0.8)

line = ax2.plot(x, composite_scores, color='#FF6600', marker='o', linewidth=2,
                markersize=6, label='Composite (right axis)', zorder=10)

for i, (model, amb) in enumerate(zip(models, ambiguity)):
    ax1.text(i, max(strict_acc[i], lenient_acc[i], error_rate[i]) + 3,
             f'Amb {amb:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_xlabel('Models', fontsize=18, fontweight='bold')
ax1.set_ylabel('Percentage', fontsize=18, fontweight='bold')
ax2.set_ylabel('Composite Score', fontsize=18, fontweight='bold')

formatted_models = []
for model in models:
    if len(model) > 15:
        if '-' in model:
            model = model.replace('-', '-\n', 1)
        elif '_' in model:
            model = model.replace('_', '_\n', 1)
    formatted_models.append(model)

ax1.set_xticks(x)
ax1.set_xticklabels(formatted_models, rotation=30, ha='right', fontsize=14)
ax1.set_ylim(0, 100)
ax2.set_ylim(-2, 2)
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(loc='upper left', fontsize=14)
ax2.legend(loc='upper right', fontsize=14)

plt.title('CSMID: Aggregate and Per-Model Performance (5 prompts/model)',
          fontsize=20, fontweight='bold', pad=20)
plt.tight_layout()

out_png = os.path.join(WORKSPACE, 'figure1_verified.png')
out_pdf = os.path.join(WORKSPACE, 'figure1_verified.pdf')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print(f"\nFigure 1 generated successfully:")
print(f"  PNG: {out_png}")
print(f"  PDF: {out_pdf}")
print(f"\nKey values for claim verification:")
print(f"  All 50 model-prompt conditions: {len(log_files)} files processed")
print(f"  10 unique models found: {len(model_classifications)}")
print(f"  Models: {sorted(model_classifications.keys())}")
print(f"  Global strict accuracy: {global_metrics['strict_accuracy']*100:.1f}% (paper claims 48.0%)")
print(f"  Global lenient accuracy: {global_metrics['lenient_accuracy']*100:.1f}% (paper claims 61.2%)")
