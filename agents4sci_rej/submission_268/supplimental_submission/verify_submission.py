#!/usr/bin/env python3
"""
Verification script to ensure all submission components are present and correct
"""

import os
import json
import csv
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and report status"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✅ {description}: {file_path} ({size} bytes)")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def verify_submission():
    """Verify all required submission components"""
    print("=" * 60)
    print("SUBMISSION VERIFICATION")
    print("=" * 60)

    base_dir = Path(".")
    issues = []

    # Required directories
    print("\n📁 DIRECTORY STRUCTURE")
    required_dirs = [
        "paper", "code", "data", "results", "results/figures",
        "paper/figures", "paper/statements", "prompts", "admin"
    ]

    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ Directory: {dir_path}/")
        else:
            print(f"❌ Directory: {dir_path}/ (MISSING)")
            issues.append(f"Missing directory: {dir_path}")

    # Paper files
    print("\n📄 PAPER FILES")
    paper_files = {
        "paper/main.tex": "Main LaTeX paper",
        "paper/refs.bib": "Bibliography",
        "paper/math_formulation.tex": "Mathematical formulation",
        "paper/agents4science_2025.sty": "Conference style file",
        "paper/compile_paper.sh": "PDF compilation script",
        "paper/PDF_GENERATION_INSTRUCTIONS.md": "PDF generation guide"
    }

    for file_path, desc in paper_files.items():
        if not check_file_exists(file_path, desc):
            issues.append(f"Missing paper file: {file_path}")

    # Statement files
    print("\n📋 STATEMENT FILES")
    statement_files = {
        "paper/statements/ai_contribution.tex": "AI contribution statement",
        "paper/statements/broader_impact.tex": "Broader impact statement",
        "paper/statements/reproducibility.tex": "Reproducibility statement"
    }

    for file_path, desc in statement_files.items():
        if not check_file_exists(file_path, desc):
            issues.append(f"Missing statement: {file_path}")

    # Figures
    print("\n🖼️  FIGURES")
    required_figures = [
        "ablation_study.pdf",
        "fairness_accuracy_tradeoff.pdf",
        "confusion_matrices.pdf",
        "dataset_visualization.pdf",
        "fairness_metrics_comparison.pdf"
    ]

    for fig in required_figures:
        paper_fig = f"paper/figures/{fig}"
        results_fig = f"results/figures/{fig}"

        paper_exists = check_file_exists(paper_fig, f"Paper figure: {fig}")
        results_exists = check_file_exists(results_fig, f"Results figure: {fig}")

        if not paper_exists:
            issues.append(f"Missing paper figure: {fig}")
        if not results_exists:
            issues.append(f"Missing results figure: {fig}")

    # Code files
    print("\n💻 CODE FILES")
    code_files = {
        "code/run_experiments.py": "Main experiment runner",
        "code/dataset.py": "Dataset generation",
        "code/model.py": "Model implementations",
        "code/train.py": "Training pipeline",
        "code/evaluate.py": "Evaluation metrics",
        "code/create_figures.py": "Figure generation",
        "code/simple_experiment.py": "Simplified experiment",
        "code/README.md": "Code documentation"
    }

    for file_path, desc in code_files.items():
        if not check_file_exists(file_path, desc):
            issues.append(f"Missing code file: {file_path}")

    # Results files
    print("\n📊 RESULTS FILES")
    results_files = {
        "results/metrics.json": "Summary metrics",
        "results/model_comparison.csv": "Model comparison",
        "results/ablation_study.csv": "Ablation study results"
    }

    for file_path, desc in results_files.items():
        if not check_file_exists(file_path, desc):
            issues.append(f"Missing results file: {file_path}")

    # Data and metadata
    print("\n📋 DATA FILES")
    data_files = {
        "data/metadata.json": "Experiment metadata"
    }

    for file_path, desc in data_files.items():
        if not check_file_exists(file_path, desc):
            issues.append(f"Missing data file: {file_path}")

    # Admin files
    print("\n🗂️  ADMIN FILES")
    admin_files = {
        "admin/openreview_id.txt": "OpenReview submission ID",
        "admin/checklist.md": "Submission checklist"
    }

    for file_path, desc in admin_files.items():
        if not check_file_exists(file_path, desc):
            issues.append(f"Missing admin file: {file_path}")

    # Verify JSON files
    print("\n🔍 DATA VALIDATION")
    try:
        with open("data/metadata.json", 'r') as f:
            metadata = json.load(f)
        print("✅ metadata.json is valid JSON")
        print(f"   - Authors: {len(metadata.get('authors', []))}")
        print(f"   - References: {len(metadata.get('references', []))}")
    except Exception as e:
        print(f"❌ metadata.json validation failed: {e}")
        issues.append("Invalid metadata.json")

    try:
        with open("results/metrics.json", 'r') as f:
            metrics = json.load(f)
        print("✅ metrics.json is valid JSON")
        print(f"   - Models evaluated: {metrics.get('all_models_summary', {}).get('num_models_evaluated', 'unknown')}")
    except Exception as e:
        print(f"❌ metrics.json validation failed: {e}")
        issues.append("Invalid metrics.json")

    # Verify CSV files
    try:
        with open("results/model_comparison.csv", 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        print(f"✅ model_comparison.csv has {len(rows)} rows (including header)")
    except Exception as e:
        print(f"❌ model_comparison.csv validation failed: {e}")
        issues.append("Invalid model_comparison.csv")

    # Final summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if not issues:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Submission package is complete and ready")
        print("📦 Total files verified: 25+")
        print("🚀 Ready for conference submission")
    else:
        print(f"⚠️  {len(issues)} ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n🔧 Please fix these issues before submission")

    print(f"\n📋 Verification complete: {len(issues)} issues found")
    return len(issues) == 0

if __name__ == "__main__":
    success = verify_submission()
    exit(0 if success else 1)