# Compile main results table comparing BT, AB-MNL, and GRK across all test splits.
# Produces JSON and markdown summary tables.

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

METHOD_FILES = {
    'BT': {
        'global': 'bt_baseline_global.json',
        'instrumental': 'bt_baseline_instrumental.json',
        'vocal': 'bt_baseline_vocal.json',
    },
    'AB-MNL': {
        'global': 'abmnl_baseline_global.json',
        'instrumental': 'abmnl_baseline_instrumental.json',
        'vocal': 'abmnl_baseline_vocal.json',
    },
    'GRK': {
        'global': 'grk_global.json',
        'instrumental': 'grk_instrumental.json',
        'vocal': 'grk_vocal.json',
    },
}


def load_result(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def fmt_ci(boot_dict):
    if boot_dict is None:
        return "N/A"
    val = boot_dict['value']
    lo = boot_dict['ci_lower']
    hi = boot_dict['ci_upper']
    return f"{val:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_brier_ci(boot_dict):
    if boot_dict is None:
        return "N/A"
    val = boot_dict['value']
    lo = boot_dict['ci_lower']
    hi = boot_dict['ci_upper']
    return f"{val:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    for method, files in METHOD_FILES.items():
        all_results[method] = {}
        for split, fname in files.items():
            all_results[method][split] = load_result(fname)

    grk_global = all_results['GRK']['global']
    pairwise = grk_global.get('pairwise_bootstrap', {}) if grk_global else {}

    table_data = {'main_table': [], 'per_class_table': [], 'pairwise_significance': {}}

    splits = ['global', 'instrumental', 'vocal']
    methods = ['BT', 'AB-MNL', 'GRK']

    for split in splits:
        for method in methods:
            r = all_results[method][split]
            if r is None:
                continue

            row = {
                'method': method,
                'split': split,
                'n_battles': r.get('n_battles', 0),
                'four_way_nll': r.get('four_way_nll'),
                'four_way_nll_ci': fmt_ci(r.get('bootstrap_four_way_nll')),
                'brier_bothbad': r.get('brier_score_bothbad'),
                'brier_bothbad_ci': fmt_brier_ci(r.get('bootstrap_brier_bothbad')),
                'ece_bothbad': r.get('ece_bothbad'),
            }
            table_data['main_table'].append(row)

            cls_row = {
                'method': method,
                'split': split,
            }
            cls_nll = r.get('per_class_nll', {})
            for cls_name in ['A', 'B', 'TIE', 'BOTH_BAD']:
                cls_row[f'nll_{cls_name}'] = cls_nll.get(cls_name)
            table_data['per_class_table'].append(cls_row)

    if pairwise:
        table_data['pairwise_significance'] = pairwise

    json_path = os.path.join(RESULTS_DIR, 'main_results_table.json')
    with open(json_path, 'w') as f:
        json.dump(table_data, f, indent=2, default=str)
    print(f"Saved JSON table to {json_path}")

    md_lines = []
    md_lines.append("# Main Results: BT vs AB-MNL vs GRK\n")

    for split in splits:
        md_lines.append(f"\n## {split.title()} Test Set\n")
        md_lines.append("| Method | 4-way NLL (95% CI) | BOTH_BAD Brier (95% CI) | BOTH_BAD ECE |")
        md_lines.append("|--------|-------------------|------------------------|-------------|")

        for method in methods:
            r = all_results[method][split]
            if r is None:
                continue
            nll_str = fmt_ci(r.get('bootstrap_four_way_nll'))
            brier_str = fmt_brier_ci(r.get('bootstrap_brier_bothbad'))
            ece_val = r.get('ece_bothbad', 0)
            best_marker = ""
            if method == 'GRK':
                best_marker = " **"
            md_lines.append(f"| {method}{best_marker} | {nll_str} | {brier_str} | {ece_val:.4f} |")

    md_lines.append("\n## Per-Class NLL Breakdown (Global Test Set)\n")
    md_lines.append("| Method | A-wins NLL | B-wins NLL | TIE NLL | BOTH_BAD NLL |")
    md_lines.append("|--------|-----------|-----------|---------|-------------|")
    for method in methods:
        r = all_results[method]['global']
        if r is None:
            continue
        cls = r.get('per_class_nll', {})
        md_lines.append(
            f"| {method} | {cls.get('A', 0):.4f} | {cls.get('B', 0):.4f} "
            f"| {cls.get('TIE', 0):.4f} | {cls.get('BOTH_BAD', 0):.4f} |"
        )

    if pairwise:
        md_lines.append("\n## Pairwise Bootstrap Significance (Global Test Set)\n")
        md_lines.append("| Comparison | Metric | Diff (GRK - baseline) | 95% CI | Significant |")
        md_lines.append("|-----------|--------|----------------------|--------|-------------|")
        for key, label in [
            ('grk_vs_bt_nll', 'GRK vs BT | NLL'),
            ('grk_vs_abmnl_nll', 'GRK vs AB-MNL | NLL'),
            ('grk_vs_bt_brier', 'GRK vs BT | Brier'),
            ('grk_vs_abmnl_brier', 'GRK vs AB-MNL | Brier'),
        ]:
            pw = pairwise.get(key, {})
            diff = pw.get('base_diff', 0)
            ci_lo = pw.get('ci_lower', 0)
            ci_hi = pw.get('ci_upper', 0)
            sig = pw.get('significant', False)
            sig_str = "Yes" if sig else "No"
            md_lines.append(f"| {label} | {diff:.4f} | [{ci_lo:.4f}, {ci_hi:.4f}] | {sig_str} |")

    md_text = "\n".join(md_lines)
    md_path = os.path.join(RESULTS_DIR, 'main_results_table.md')
    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f"Saved markdown table to {md_path}")

    print("\n" + md_text)


if __name__ == '__main__':
    main()
