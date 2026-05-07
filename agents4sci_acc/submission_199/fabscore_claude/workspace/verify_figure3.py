#!/usr/bin/env python3
"""
Verification script for Claim 23 - Figure 3:
Box plots of cross-model metric distributions per strategy.
5 metrics: Strict, Lenient, Error, Ambiguity, Composite Score.
Claim says:
  - Expert-only prompts (no time/id = S1/S2/S3) yield higher strict/lenient, lower error
  - Adding time/id (S4/S5) nudges models toward acceptance (higher error, similar or slightly lower ambiguity)
  - Composite score panel includes a zero reference line
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_DIR = "/home/chenhui/fabscore/agent4sci_acc/submission_199/199_Multi_LLM_and_Multi_Prompt_Supplementary Material/Logs"
WORKSPACE = "/home/chenhui/fabscore/agent4sci_acc/submission_199/fabscore_claude/workspace"

STRATEGY_KEYS = ['no_guide', 'public_health_expert', 'respiratory_doctor', 'detailed_public_health', 'detailed_respiratory']
STRATEGY_LABELS = ['S1: no_guide', 'S2: public_health\nexpert', 'S3: respiratory\ndoctor', 'S4: detailed\npublic_health', 'S5: detailed\nrespiratory']

MODELS = [
    'chatgpt-4o-latest',
    'deepseek-r1-250528',
    'deepseek-v3',
    'doubao-seedream',
    'gemini-2.5-pro',
    'glm-4-airx',
    'llama-4-maverick',
    'mistral-large-latest',
    'qwen3-235b-a22b-thinking-2507',
    'qwen3-235b',
]

def calculate_metrics(classifications):
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
        'strict_accuracy': N1 / N * 100,
        'lenient_accuracy': (N1 + N2) / N * 100,
        'error_rate': (N4 + N5) / N * 100,
        'ambiguity': N3 / N * 100,
        'composite_score': (2*N1 + N2 - N4 - 2*N5) / N,
    }

# Build per-strategy, per-model data
strategy_metrics = {s: {'strict_accuracy': [], 'lenient_accuracy': [], 'error_rate': [], 'ambiguity': [], 'composite_score': []} for s in STRATEGY_KEYS}

print("Loading all 50 log files...")
for model in MODELS:
    for strat in STRATEGY_KEYS:
        fpath = os.path.join(LOG_DIR, f"{model}_{strat}.json")
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found!")
            continue
        with open(fpath) as f:
            records = json.load(f)
        classifications = [r['judgment_classification'] for r in records if 'judgment_classification' in r]
        m = calculate_metrics(classifications)
        if m:
            for k in ['strict_accuracy', 'lenient_accuracy', 'error_rate', 'ambiguity', 'composite_score']:
                strategy_metrics[strat][k].append(m[k])

print("\n=== PER-STRATEGY CROSS-MODEL DISTRIBUTIONS ===")
print(f"{'Strategy':<30} {'Strict(med)':>12} {'Lenient(med)':>12} {'Error(med)':>12} {'Ambig(med)':>12} {'Composite(med)':>14}")
print("-" * 95)

medians = {}
iqrs = {}
for strat in STRATEGY_KEYS:
    medians[strat] = {}
    iqrs[strat] = {}
    row = [f"{strat:<30}"]
    for k in ['strict_accuracy', 'lenient_accuracy', 'error_rate', 'ambiguity', 'composite_score']:
        vals = strategy_metrics[strat][k]
        med = np.median(vals)
        q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
        medians[strat][k] = med
        iqrs[strat][k] = (q25, q75)
        row.append(f"{med:>11.2f}")
    print("".join(row))

print("\n=== PER-STRATEGY IQR (Q25–Q75) ===")
for strat in STRATEGY_KEYS:
    print(f"\n{strat}:")
    for k in ['strict_accuracy', 'lenient_accuracy', 'error_rate', 'ambiguity', 'composite_score']:
        q25, q75 = iqrs[strat][k]
        print(f"  {k:<25}: Q25={q25:.2f}, Q75={q75:.2f}, IQR={q75-q25:.2f}")

print("\n=== CLAIM VERIFICATION ===")
print("\nClaim: Expert-only prompts (S1/S2/S3, no time/id) yield higher strict/lenient, lower error")
no_time_id = ['no_guide', 'public_health_expert', 'respiratory_doctor']
with_time_id = ['detailed_public_health', 'detailed_respiratory']

for k in ['strict_accuracy', 'lenient_accuracy', 'error_rate']:
    no_time_meds = [medians[s][k] for s in no_time_id]
    with_time_meds = [medians[s][k] for s in with_time_id]
    print(f"\n  {k}:")
    print(f"    No time/id (S1-S3) medians: {[f'{v:.2f}' for v in no_time_meds]}")
    print(f"    With time/id (S4-S5) medians: {[f'{v:.2f}' for v in with_time_meds]}")
    print(f"    Avg no-time/id: {np.mean(no_time_meds):.2f} vs avg with-time/id: {np.mean(with_time_meds):.2f}")
    if k in ['strict_accuracy', 'lenient_accuracy']:
        print(f"    Claim check (no-time/id > with-time/id): {np.mean(no_time_meds) > np.mean(with_time_meds)}")
    elif k == 'error_rate':
        print(f"    Claim check (no-time/id < with-time/id for error): {np.mean(no_time_meds) < np.mean(with_time_meds)}")

print("\n  Ambiguity (claim: similar or slightly lower with time/id):")
for s in STRATEGY_KEYS:
    print(f"    {s}: ambiguity median = {medians[s]['ambiguity']:.2f}%")

print("\n  Composite Score includes zero reference line: always True (hard-coded in script)")
print("  Composite score medians:")
for s in STRATEGY_KEYS:
    print(f"    {s}: composite_score median = {medians[s]['composite_score']:.4f}")

# Generate Figure 3 box plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Figure 3: Comprehensive Evaluation Metrics Across Strategies (Box Plots)', fontsize=18, fontweight='bold')

metrics = ['strict_accuracy', 'lenient_accuracy', 'error_rate', 'ambiguity', 'composite_score']
titles = ['Strict Accuracy (%)', 'Lenient Accuracy (%)', 'Error Rate (%)', 'Ambiguity Rate (%)', 'Composite Score']
colors = ['lightblue', 'lightcoral', 'lightgreen', 'plum', 'lightyellow']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    data_for_boxplot = [strategy_metrics[s][metric] for s in STRATEGY_KEYS]
    bp = ax.boxplot(data_for_boxplot, labels=STRATEGY_LABELS, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.3)
    if metric == 'composite_score':
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='zero reference')
        ax.legend(fontsize=8)

# Hide 6th subplot
axes[1, 2].set_visible(False)

plt.tight_layout()

out_png = os.path.join(WORKSPACE, 'figure3_verified.png')
out_pdf = os.path.join(WORKSPACE, 'figure3_verified.pdf')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()

print(f"\nFigure 3 box plots generated:")
print(f"  PNG: {out_png}")
print(f"  PDF: {out_pdf}")
