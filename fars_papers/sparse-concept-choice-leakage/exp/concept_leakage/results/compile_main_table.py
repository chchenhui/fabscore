"""Compile main results table from all experimental conditions.
Aggregates: Chance baseline, Isotropic (B), Anisotropic (A), Smoothed (C).
Outputs CSV and LaTeX table."""

import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_OPT_DIR = BASE_DIR / "results_opt"


def fmt(mean, std):
    return f"{mean:.3f} +/- {std:.3f}"


def fmt_latex(mean, std):
    return f"${mean:.3f} \\pm {std:.3f}$"


def main():
    iso_attack = json.load(open(RESULTS_DIR / "isotropic_attack_results.json"))
    iso_sts = json.load(open(RESULTS_DIR / "isotropic_results.json"))

    aniso_attack = json.load(open(RESULTS_OPT_DIR / "anisotropic_attack_results.json"))
    aniso_sts = json.load(open(RESULTS_OPT_DIR / "anisotropic_results.json"))

    smooth_attack = json.load(open(RESULTS_DIR / "smoothed_attack_results.json"))
    smooth_sts = json.load(open(RESULTS_DIR / "smoothed_results.json"))

    rows = [
        {
            "method": "Chance",
            "noise_covariance": "N/A",
            "acc_mean": 0.200, "acc_std": 0.0,
            "f1_mean": 0.200, "f1_std": 0.0,
            "sts_mean": None, "sts_std": None,
        },
        {
            "method": "Isotropic (B)",
            "noise_covariance": "Sigma = I",
            "acc_mean": iso_attack["accuracy_mean"],
            "acc_std": iso_attack["accuracy_std"],
            "f1_mean": iso_attack["macro_f1_mean"],
            "f1_std": iso_attack["macro_f1_std"],
            "sts_mean": iso_sts["sts12"]["noisy_pearson_mean"],
            "sts_std": iso_sts["sts12"]["noisy_pearson_std"],
        },
        {
            "method": "Anisotropic (A)",
            "noise_covariance": "Sigma = diag(m_k + delta)",
            "acc_mean": aniso_attack["accuracy_mean"],
            "acc_std": aniso_attack["accuracy_std"],
            "f1_mean": aniso_attack["macro_f1_mean"],
            "f1_std": aniso_attack["macro_f1_std"],
            "sts_mean": aniso_sts["sts12"]["noisy_pearson_mean"],
            "sts_std": aniso_sts["sts12"]["noisy_pearson_std"],
        },
        {
            "method": "Smoothed (C)",
            "noise_covariance": "(1-lam)*Sigma_k + lam*I",
            "acc_mean": smooth_attack["accuracy_mean"],
            "acc_std": smooth_attack["accuracy_std"],
            "f1_mean": smooth_attack["macro_f1_mean"],
            "f1_std": smooth_attack["macro_f1_std"],
            "sts_mean": smooth_sts["sts12"]["noisy_pearson_mean"],
            "sts_std": smooth_sts["sts12"]["noisy_pearson_std"],
        },
    ]

    csv_path = RESULTS_DIR / "main_results_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Noise Covariance", "Concept-ID Accuracy",
                          "Concept-ID Macro-F1", "STS12 Pearson"])
        for r in rows:
            acc = fmt(r["acc_mean"], r["acc_std"])
            f1 = fmt(r["f1_mean"], r["f1_std"])
            sts = fmt(r["sts_mean"], r["sts_std"]) if r["sts_mean"] is not None else "---"
            writer.writerow([r["method"], r["noise_covariance"], acc, f1, sts])
    print(f"CSV saved to {csv_path}")

    tex_path = RESULTS_DIR / "main_results_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Concept-choice leakage attack results across noise conditions. "
                "Accuracy and Macro-F1 for concept identification (higher = more leakage); "
                "STS12 Pearson for embedding utility (higher = better). "
                "Mean $\\pm$ std over 3 seeds.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Noise Covariance & Concept-ID Acc & Concept-ID F1 & STS12 Pearson \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            acc = fmt_latex(r["acc_mean"], r["acc_std"])
            f1 = fmt_latex(r["f1_mean"], r["f1_std"])
            if r["sts_mean"] is not None:
                sts = fmt_latex(r["sts_mean"], r["sts_std"])
            else:
                sts = "---"
            cov = r["noise_covariance"].replace("_", "\\_").replace("lam", "\\lambda")
            f.write(f"{r['method']} & {cov} & {acc} & {f1} & {sts} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX saved to {tex_path}")

    print("\n=== Main Results Table ===")
    header = f"{'Method':<20} {'Noise Covariance':<30} {'Acc (mean+/-std)':<20} {'F1 (mean+/-std)':<20} {'STS12 Pearson':<20}"
    print(header)
    print("-" * len(header))
    for r in rows:
        acc = fmt(r["acc_mean"], r["acc_std"])
        f1 = fmt(r["f1_mean"], r["f1_std"])
        sts = fmt(r["sts_mean"], r["sts_std"]) if r["sts_mean"] is not None else "---"
        print(f"{r['method']:<20} {r['noise_covariance']:<30} {acc:<20} {f1:<20} {sts:<20}")


if __name__ == "__main__":
    main()
