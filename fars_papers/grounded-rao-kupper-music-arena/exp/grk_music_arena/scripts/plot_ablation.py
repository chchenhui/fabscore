# Plot ablation study results:
# 1. GRK ablation bar chart (BT, AB-MNL, GRK-no-grounding, GRK)
# 2. AB-MNL regularization sensitivity curve with GRK reference line

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def plot_grk_ablation():
    bt = load_json('bt_baseline_global.json')
    abmnl = load_json('abmnl_baseline_global.json')
    noground = load_json('grk_no_ground_global.json')
    grk = load_json('grk_global.json')

    models = ['BT', 'AB-MNL', 'GRK-No-Ground', 'GRK']
    nlls = [
        bt['four_way_nll'],
        abmnl['four_way_nll'],
        noground['four_way_nll'],
        grk['four_way_nll'],
    ]
    ci_lowers = [
        bt['bootstrap_four_way_nll']['ci_lower'],
        abmnl['bootstrap_four_way_nll']['ci_lower'],
        noground['bootstrap_four_way_nll']['ci_lower'],
        grk['bootstrap_four_way_nll']['ci_lower'],
    ]
    ci_uppers = [
        bt['bootstrap_four_way_nll']['ci_upper'],
        abmnl['bootstrap_four_way_nll']['ci_upper'],
        noground['bootstrap_four_way_nll']['ci_upper'],
        grk['bootstrap_four_way_nll']['ci_upper'],
    ]

    errors_lower = [nll - ci_l for nll, ci_l in zip(nlls, ci_lowers)]
    errors_upper = [ci_u - nll for nll, ci_u in zip(nlls, ci_uppers)]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ['#999999', '#5B9BD5', '#ED7D31', '#70AD47']
    x = np.arange(len(models))

    bars = ax.bar(x, nlls, yerr=[errors_lower, errors_upper],
                  capsize=5, color=colors, edgecolor='black', linewidth=0.5,
                  width=0.6)

    for bar, nll_val in zip(bars, nlls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{nll_val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('4-way NLL (lower is better)', fontsize=11)
    ax.set_title('GRK Ablation: Effect of Grounding Mechanism', fontsize=12)

    y_min = min(nlls) * 0.85
    y_max = max(ci_uppers) * 1.08
    if y_max > 3:
        y_max = max(nlls[1:]) * 1.15
        ax.set_ylim(0, y_max)
        ax.annotate(f'BT: {nlls[0]:.1f}', xy=(0, y_max * 0.95),
                    ha='center', fontsize=9, color='#999999', fontweight='bold')
        bars[0].set_height(y_max * 0.92)
        bars[0].set_y(0)
    else:
        ax.set_ylim(y_min, y_max)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.subplots_adjust(bottom=0.12, top=0.9, left=0.12, right=0.95)

    path = os.path.join(FIGURES_DIR, 'grk_ablation.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def plot_abmnl_reg_curve():
    data = load_json('abmnl_reg_sensitivity.json')
    grk_nll = data['grk_reference_nll']

    l2_vals = []
    nlls = []
    briers = []
    pearson_corrs = []
    spearman_corrs = []

    for r in data['results']:
        l2_vals.append(r['l2_rho'] if r['l2_rho'] > 0 else 1e-4)
        nlls.append(r['four_way_nll'])
        briers.append(r['brier_score_bothbad'])
        pearson_corrs.append(r['pearson_corr_rho_neg_beta'])
        spearman_corrs.append(r['spearman_corr_rho_neg_beta'])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_nll = '#2E75B6'
    ax1.plot(l2_vals, nlls, 'o-', color=color_nll, linewidth=2, markersize=6, label='AB-MNL 4-way NLL')
    ax1.axhline(y=grk_nll, color='#70AD47', linestyle='--', linewidth=1.5,
                label=f'GRK 4-way NLL ({grk_nll:.4f})')
    ax1.set_xscale('log')
    ax1.set_xlabel('L2 Penalty on rho (log scale)', fontsize=11)
    ax1.set_ylabel('Test 4-way NLL', fontsize=11, color=color_nll)
    ax1.tick_params(axis='y', labelcolor=color_nll)

    true_l2_labels = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    ax1.set_xticks(l2_vals)
    ax1.set_xticklabels([str(v) for v in true_l2_labels], fontsize=8, rotation=45)

    ax2 = ax1.twinx()
    color_corr = '#C00000'
    ax2.plot(l2_vals, pearson_corrs, 's--', color=color_corr, linewidth=1.5,
             markersize=5, alpha=0.8, label='Pearson corr(rho, -beta)')
    ax2.plot(l2_vals, spearman_corrs, 'd:', color='#FF6600', linewidth=1.5,
             markersize=5, alpha=0.8, label='Spearman corr(rho, -beta)')
    ax2.set_ylabel('Correlation(rho, -beta)', fontsize=11, color=color_corr)
    ax2.tick_params(axis='y', labelcolor=color_corr)
    ax2.set_ylim(-1.0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    ax1.set_title('AB-MNL Regularization Sensitivity vs GRK Reference', fontsize=12)
    ax1.spines['top'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'abmnl_reg_curve.pdf')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved {path}")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_grk_ablation()
    plot_abmnl_reg_curve()
    print("Done!")


if __name__ == '__main__':
    main()
