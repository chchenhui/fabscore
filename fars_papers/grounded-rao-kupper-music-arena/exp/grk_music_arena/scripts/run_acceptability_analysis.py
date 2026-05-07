# Per-model acceptability analysis: empirical BOTH_BAD rates, GRK acceptability
# validation, AB-MNL rho-vs-beta coupling test, and leaderboard comparison table.

import sys, os, json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from grk_music_arena.data.load_music_arena import load_and_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def save_json(obj, name):
    with open(os.path.join(RESULTS_DIR, name), 'w') as f:
        json.dump(obj, f, indent=2)


def step1_empirical_bothbad(all_df):
    print("=== Step 1: Empirical BOTH_BAD Rates ===")
    systems = sorted(set(all_df['system_a']) | set(all_df['system_b']))
    results = {}

    for sys_name in systems:
        mask = (all_df['system_a'] == sys_name) | (all_df['system_b'] == sys_name)
        sys_battles = all_df[mask]
        n_battles = len(sys_battles)
        n_bothbad = (sys_battles['preference'] == 'BOTH_BAD').sum()
        rate = n_bothbad / n_battles if n_battles > 0 else 0.0

        rng = np.random.RandomState(42)
        boot_rates = np.zeros(1000)
        outcomes = (sys_battles['preference'] == 'BOTH_BAD').values.astype(float)
        for b in range(1000):
            idx = rng.choice(n_battles, size=n_battles, replace=True)
            boot_rates[b] = outcomes[idx].mean()

        results[sys_name] = {
            'n_battles': int(n_battles),
            'n_bothbad': int(n_bothbad),
            'empirical_rate': float(rate),
            'ci_lower': float(np.percentile(boot_rates, 2.5)),
            'ci_upper': float(np.percentile(boot_rates, 97.5)),
        }
        print(f"  {sys_name:30s}: {rate:.4f} [{results[sys_name]['ci_lower']:.4f}, {results[sys_name]['ci_upper']:.4f}] (n={n_battles})")

    save_json(results, 'empirical_bothbad_rates.json')
    print(f"  Saved to results/empirical_bothbad_rates.json")
    return results


def step2_grk_vs_empirical(empirical_rates):
    print("\n=== Step 2: GRK Acceptability vs Empirical BOTH_BAD ===")
    grk = load_json('grk_global.json')
    grk_accept = grk['acceptability']

    systems = sorted(empirical_rates.keys())
    emp = np.array([empirical_rates[s]['empirical_rate'] for s in systems])
    grk_imp = np.array([grk_accept[s] for s in systems])

    r_pearson, p_pearson = scipy_stats.pearsonr(emp, grk_imp)
    r_spearman, p_spearman = scipy_stats.spearmanr(emp, grk_imp)
    print(f"  Pearson  r={r_pearson:.4f}, p={p_pearson:.4e}")
    print(f"  Spearman r={r_spearman:.4f}, p={p_spearman:.4e}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emp, grk_imp, s=60, zorder=5, color='steelblue', edgecolors='k', linewidths=0.5)
    for i, s in enumerate(systems):
        ax.annotate(s, (emp[i], grk_imp[i]), fontsize=7,
                     xytext=(4, 4), textcoords='offset points')

    lo = min(emp.min(), grk_imp.min()) * 0.9
    hi = max(emp.max(), grk_imp.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='y = x')
    ax.set_xlabel('Empirical BOTH_BAD Rate')
    ax.set_ylabel('GRK-Implied P(BOTH_BAD | k vs avg)')
    ax.set_title(f'GRK Acceptability vs Empirical\nPearson r={r_pearson:.3f}, Spearman r={r_spearman:.3f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'grk_acceptability_vs_empirical.pdf'), dpi=150)
    plt.close(fig)
    print(f"  Saved figure to figures/grk_acceptability_vs_empirical.pdf")

    return {
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
    }


def step3_rho_vs_neg_beta():
    print("\n=== Step 3: AB-MNL rho vs -beta Correlation ===")
    abmnl = load_json('abmnl_baseline_global.json')
    beta_dict = abmnl['abmnl_scores']['beta']
    rho_dict = abmnl['abmnl_scores']['rho']

    systems = sorted(beta_dict.keys())
    beta_arr = np.array([beta_dict[s] for s in systems])
    rho_arr = np.array([rho_dict[s] for s in systems])
    neg_beta = -beta_arr

    r_pearson, p_pearson = scipy_stats.pearsonr(neg_beta, rho_arr)
    r_spearman, p_spearman = scipy_stats.spearmanr(neg_beta, rho_arr)
    print(f"  Pearson  r={r_pearson:.4f}, p={p_pearson:.4e}")
    print(f"  Spearman r={r_spearman:.4f}, p={p_spearman:.4e}")

    slope, intercept, r_ols, p_ols, se = scipy_stats.linregress(neg_beta, rho_arr)
    print(f"  OLS: rho = {slope:.4f} * (-beta) + {intercept:.4f}, R^2={r_ols**2:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(neg_beta, rho_arr, s=60, zorder=5, color='darkorange', edgecolors='k', linewidths=0.5)
    for i, s in enumerate(systems):
        ax.annotate(s, (neg_beta[i], rho_arr[i]), fontsize=7,
                     xytext=(4, 4), textcoords='offset points')

    x_line = np.linspace(neg_beta.min() - 0.2, neg_beta.max() + 0.2, 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', alpha=0.7,
            label=f'OLS: slope={slope:.4f}, R$^2$={r_ols**2:.4f}')
    ax.set_xlabel(r'$-\beta_k$ (negative skill, AB-MNL)')
    ax.set_ylabel(r'$\rho_k$ (badness, AB-MNL)')
    ax.set_title(f'AB-MNL: Badness vs Negative Skill\nPearson r={r_pearson:.3f}, Spearman r={r_spearman:.3f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'rho_vs_neg_beta.pdf'), dpi=150)
    plt.close(fig)
    print(f"  Saved figure to figures/rho_vs_neg_beta.pdf")

    corr_results = {
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_r': float(r_spearman),
        'spearman_p': float(p_spearman),
        'ols_slope': float(slope),
        'ols_intercept': float(intercept),
        'ols_r_squared': float(r_ols ** 2),
        'ols_p_value': float(p_ols),
        'ols_std_err': float(se),
        'per_system': {s: {'neg_beta': float(neg_beta[i]), 'rho': float(rho_arr[i])} for i, s in enumerate(systems)},
    }
    save_json(corr_results, 'rho_beta_correlation.json')
    print(f"  Saved to results/rho_beta_correlation.json")
    return corr_results


def step4_leaderboard(empirical_rates):
    print("\n=== Step 4: Leaderboard Comparison Table ===")
    bt = load_json('bt_baseline_global.json')
    grk = load_json('grk_global.json')
    abmnl = load_json('abmnl_baseline_global.json')

    bt_scores = bt['bt_scores']
    grk_scores = grk['grk_scores']['beta']
    grk_accept = grk['acceptability']
    abmnl_beta = abmnl['abmnl_scores']['beta']
    abmnl_rho = abmnl['abmnl_scores']['rho']

    systems = sorted(grk_scores.keys())

    rows = []
    for s in systems:
        rows.append({
            'system': s,
            'bt_score': bt_scores[s],
            'grk_score': grk_scores[s],
            'abmnl_skill': abmnl_beta[s],
            'abmnl_badness': abmnl_rho[s],
            'empirical_bothbad_pct': empirical_rates[s]['empirical_rate'] * 100,
            'grk_implied_bothbad_pct': grk_accept[s] * 100,
        })

    rows.sort(key=lambda r: r['grk_score'], reverse=True)
    for rank, r in enumerate(rows, 1):
        r['grk_rank'] = rank

    bt_ranking = {s: rank for rank, s in enumerate(
        sorted(systems, key=lambda s: bt_scores[s], reverse=True), 1)}
    grk_ranking = {r['system']: r['grk_rank'] for r in rows}

    bt_ranks = np.array([bt_ranking[s] for s in systems])
    grk_ranks = np.array([grk_ranking[s] for s in systems])
    kendall_tau, kendall_p = scipy_stats.kendalltau(bt_ranks, grk_ranks)
    print(f"  Kendall's tau (BT vs GRK rankings): {kendall_tau:.4f}, p={kendall_p:.4e}")

    for r in rows:
        r['bt_rank'] = bt_ranking[r['system']]

    table_json = {
        'rows': rows,
        'kendall_tau_bt_vs_grk': float(kendall_tau),
        'kendall_tau_p_value': float(kendall_p),
    }
    save_json(table_json, 'leaderboard_comparison.json')
    print(f"  Saved to results/leaderboard_comparison.json")

    header = "| Rank | System | BT Score | GRK Score | AB-MNL Skill | AB-MNL Badness | Emp. BOTH_BAD (%) | GRK BOTH_BAD (%) |"
    sep    = "|------|--------|----------|-----------|--------------|----------------|-------------------|------------------|"
    md_lines = [header, sep]
    for r in rows:
        md_lines.append(
            f"| {r['grk_rank']} | {r['system']} | {r['bt_score']:.3f} | {r['grk_score']:.3f} | "
            f"{r['abmnl_skill']:.3f} | {r['abmnl_badness']:.4f} | {r['empirical_bothbad_pct']:.1f} | "
            f"{r['grk_implied_bothbad_pct']:.1f} |"
        )
    md_lines.append("")
    md_lines.append(f"Kendall's tau (BT vs GRK rankings): {kendall_tau:.4f} (p={kendall_p:.4e})")

    md_text = '\n'.join(md_lines)
    with open(os.path.join(RESULTS_DIR, 'leaderboard_comparison.md'), 'w') as f:
        f.write(md_text)
    print(f"  Saved to results/leaderboard_comparison.md")
    print(f"\n{md_text}")

    return table_json


def main():
    splits, stats = load_and_split()
    all_df = pd.concat([splits['train'], splits['test']], ignore_index=True)
    print(f"Total battles: {len(all_df)}, Systems: {len(set(all_df['system_a']) | set(all_df['system_b']))}")

    empirical_rates = step1_empirical_bothbad(all_df)
    grk_corr = step2_grk_vs_empirical(empirical_rates)
    rho_beta_corr = step3_rho_vs_neg_beta()
    leaderboard = step4_leaderboard(empirical_rates)

    print("\n=== Summary ===")
    print(f"GRK acceptability vs empirical: Pearson r={grk_corr['pearson_r']:.4f}, Spearman r={grk_corr['spearman_r']:.4f}")
    print(f"AB-MNL rho vs -beta: Pearson r={rho_beta_corr['pearson_r']:.4f}, Spearman r={rho_beta_corr['spearman_r']:.4f}")
    print(f"BT vs GRK rank correlation: Kendall tau={leaderboard['kendall_tau_bt_vs_grk']:.4f}")


if __name__ == '__main__':
    main()
