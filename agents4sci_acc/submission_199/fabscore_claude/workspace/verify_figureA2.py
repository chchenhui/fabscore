#!/usr/bin/env python3
"""
Verification script for Claim 26 - Figure A2: Comprehensive analysis dashboard
with heatmap, box plot, bar chart, and pie chart of results.
"""
import json
import glob
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import defaultdict, Counter

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

# Strategy name mapping
STRATEGY_MAPPING = {
    'detailed_public_health': 'Public Health\n(with time&id)',
    'detailed_respiratory': 'Respiratory\n(with time&id)',
    'public_health_expert': 'Public Health\n(without time&id)',
    'respiratory_doctor': 'Respiratory\n(without time&id)'
}
STRATEGY_ORDER = ['detailed_public_health', 'public_health_expert', 'detailed_respiratory', 'respiratory_doctor']

# Load all 50 log files
log_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.json")))
print(f"Found {len(log_files)} log files")

# Parse log files
strategy_data = []  # list of (strategy, model, strict_accuracy)
all_classifications_4strat = []  # all classifications for the 4 strategies only
per_model_strategy = defaultdict(dict)  # model -> strategy -> strict_accuracy

for fpath in log_files:
    fname = os.path.basename(fpath)
    # Extract model and strategy from filename
    # Pattern: model_strategy_analyzed.json or similar
    m = re.match(r'^(.+?)_(no_guide|public_health_expert|respiratory_doctor|detailed_public_health|detailed_respiratory)(?:_analyzed)?\.json$', fname)
    if not m:
        print(f"  Could not parse: {fname}")
        continue
    model_name = m.group(1)
    strategy = m.group(2)

    with open(fpath) as f:
        records = json.load(f)

    classifications = [r.get('judgment_classification') for r in records if r.get('judgment_classification') is not None]
    metrics = calculate_metrics(classifications)
    if metrics is None:
        continue

    # Only include the 4 strategies used in comprehensive analysis
    if strategy in STRATEGY_MAPPING:
        strategy_display = STRATEGY_MAPPING[strategy]
        strategy_data.append({
            'strategy': strategy_display,
            'model': model_name,
            'strict_accuracy': metrics['strict_accuracy'] * 100
        })
        per_model_strategy[model_name][strategy_display] = metrics['strict_accuracy'] * 100
        all_classifications_4strat.extend([int(c) for c in classifications])

print(f"\nProcessed {len(strategy_data)} (model, strategy) pairs for 4 strategies")
print(f"Models: {sorted(set(d['model'] for d in strategy_data))}")
print(f"Strategies: {sorted(set(d['strategy'] for d in strategy_data))}")

# Print strict accuracy stats per strategy for verification
strategy_display_order = [STRATEGY_MAPPING[s] for s in STRATEGY_ORDER]
print("\nPer-strategy strict accuracy distributions:")
for strat in strategy_display_order:
    vals = [d['strict_accuracy'] for d in strategy_data if d['strategy'] == strat]
    if vals:
        print(f"  {strat.replace(chr(10),' ')}: n={len(vals)}, mean={np.mean(vals):.2f}%, median={np.median(vals):.2f}%, min={np.min(vals):.2f}%, max={np.max(vals):.2f}%")

# Label composition
label_counts = Counter(all_classifications_4strat)
total_count = sum(label_counts.values())
print(f"\nLabel composition (4 strategies):")
for i in range(1, 6):
    count = label_counts.get(i, 0)
    print(f"  Label {i}: {count} ({count/total_count*100:.1f}%)")

# Build heatmap data
models = sorted(per_model_strategy.keys())
heatmap_data = np.zeros((len(models), len(strategy_display_order)))
for mi, model in enumerate(models):
    for si, strat in enumerate(strategy_display_order):
        heatmap_data[mi, si] = per_model_strategy[model].get(strat, np.nan)

print("\nHeatmap data (models x strategies):")
print(f"  Shape: {heatmap_data.shape}")

# ---- Create the comprehensive analysis dashboard ----
import seaborn as sns
sns.set_style("whitegrid")

fig = plt.figure(figsize=(20, 12))
gs_spec = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

# Panel 1 (top-left): Box plot - Strategy Performance Distribution
ax1 = fig.add_subplot(gs_spec[0, 0])
strategy_groups = []
strategy_labels_bp = []
colors_bp = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for strat in strategy_display_order:
    vals = [d['strict_accuracy'] for d in strategy_data if d['strategy'] == strat]
    if vals:
        strategy_groups.append(vals)
        strategy_labels_bp.append(strat)

bp = ax1.boxplot(strategy_groups, labels=strategy_labels_bp, patch_artist=True)
for patch, color in zip(bp['boxes'], colors_bp[:len(bp['boxes'])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_title('Strategy Performance Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Strict Accuracy (%)', fontsize=12)
ax1.set_xlabel('Strategy', fontsize=12)
ax1.tick_params(axis='x', labelsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2 (top-right): Heatmap - Model Performance
ax2 = fig.add_subplot(gs_spec[0, 1])
import pandas as pd
heatmap_df = pd.DataFrame(heatmap_data, index=models, columns=[s.replace('\n', ' ') for s in strategy_display_order])
sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2,
            cbar_kws={'label': 'Strict Accuracy (%)'},
            annot_kws={'size': 9})
ax2.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
ax2.set_xlabel('Strategy', fontsize=12)
ax2.set_ylabel('Model', fontsize=12)
ax2.tick_params(axis='x', labelsize=8, rotation=15)
ax2.tick_params(axis='y', labelsize=8, rotation=0)

# Panel 3 (bottom-left): Bar chart - Average Performance by Strategy
ax3 = fig.add_subplot(gs_spec[1, 0])
strategy_avgs = []
strategy_names_bar = []
for strat in strategy_display_order:
    vals = [d['strict_accuracy'] for d in strategy_data if d['strategy'] == strat]
    if vals:
        strategy_avgs.append(np.mean(vals))
        strategy_names_bar.append(strat.replace('\n', ' '))

# Sort by value ascending (horizontal bar)
sorted_pairs = sorted(zip(strategy_avgs, strategy_names_bar))
sorted_vals, sorted_names = zip(*sorted_pairs)
bars = ax3.barh(sorted_names, sorted_vals, color='steelblue', alpha=0.7)
for bar, value in zip(bars, sorted_vals):
    ax3.text(value + 0.5, bar.get_y() + bar.get_height()/2, f'{value:.1f}%',
             va='center', ha='left', fontsize=11, fontweight='bold')
ax3.set_title('Average Performance by Strategy', fontsize=14, fontweight='bold')
ax3.set_xlabel('Average Strict Accuracy (%)', fontsize=12)
ax3.tick_params(axis='y', labelsize=9)
ax3.grid(True, alpha=0.3, axis='x')

# Panel 4 (bottom-right): Pie chart - Judge-label Composition
ax4 = fig.add_subplot(gs_spec[1, 1])
labels_pie = []
sizes_pie = []
colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i in range(1, 6):
    count = label_counts.get(i, 0)
    pct = (count / total_count) * 100
    labels_pie.append(f'Label {i}\n{pct:.1f}%')
    sizes_pie.append(count)

ax4.pie(sizes_pie, labels=labels_pie, colors=colors_pie, startangle=90,
        textprops={'fontsize': 11}, autopct='')
ax4.set_title('Judge-label Composition', fontsize=14, fontweight='bold')

plt.suptitle('Comprehensive Analysis Dashboard', fontsize=20, fontweight='bold', y=0.99)
out_png = os.path.join(WORKSPACE, 'figureA2_verified.png')
out_pdf = os.path.join(WORKSPACE, 'figureA2_verified.pdf')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
print("\n=== FIGURE A2 VERIFICATION SUMMARY ===")
print("4-panel layout confirmed:")
print("  Top-left: Box plot (Strategy Performance Distribution)")
print("  Top-right: Heatmap (Model Performance Heatmap)")
print("  Bottom-left: Bar chart (Average Performance by Strategy)")
print("  Bottom-right: Pie chart (Judge-label Composition)")
print(f"Data: {len(strategy_data)} (model,strategy) entries, {total_count} total classifications")
print("Claim states: heatmap, box plot, bar chart, pie chart — ALL 4 PRESENT.")
