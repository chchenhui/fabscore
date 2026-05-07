#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis for LLM Inbreeding Research

Analyzes existing datasets and identifies gaps in coverage for comprehensive
LLM quality deterioration analysis.
"""

import os
import pandas as pd
import json
import zipfile
from pathlib import Path
import numpy as np
from collections import defaultdict

def analyze_csv_file(filepath):
    """Analyze a CSV dataset file"""
    try:
        df = pd.read_csv(filepath)
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'file_size_mb': os.path.getsize(filepath) / (1024*1024),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024*1024),
            'sample_data': df.head(2).to_dict('records') if len(df) > 0 else []
        }
    except Exception as e:
        return {'error': str(e), 'file_size_mb': os.path.getsize(filepath) / (1024*1024)}

def analyze_zip_file(filepath):
    """Analyze a ZIP dataset file"""
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            files = zf.namelist()
            return {
                'file_count': len(files),
                'contained_files': files[:10],  # First 10 files
                'file_size_mb': os.path.getsize(filepath) / (1024*1024),
                'total_uncompressed_size': sum(zf.getinfo(f).file_size for f in files) / (1024*1024)
            }
    except Exception as e:
        return {'error': str(e), 'file_size_mb': os.path.getsize(filepath) / (1024*1024)}

def main():
    print("🔍 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 80)
    
    data_dir = Path('data')
    
    if not data_dir.exists():
        print("❌ No data directory found!")
        return
    
    # Categorize datasets by type
    categories = {
        'evaluation': [],
        'reasoning': [],
        'coding': [],
        'knowledge': [],
        'raw': []
    }
    
    analysis_results = {}
    total_size = 0
    total_samples = 0
    
    # Scan all dataset files
    for category in categories.keys():
        cat_dir = data_dir / category
        if cat_dir.exists():
            for file_path in cat_dir.glob('*'):
                if file_path.is_file():
                    print(f"\n📊 Analyzing: {file_path}")
                    
                    if file_path.suffix == '.csv':
                        result = analyze_csv_file(file_path)
                        if 'rows' in result:
                            total_samples += result['rows']
                    elif file_path.suffix == '.zip':
                        result = analyze_zip_file(file_path)
                    else:
                        result = {'file_size_mb': os.path.getsize(file_path) / (1024*1024)}
                    
                    result['category'] = category
                    result['file_type'] = file_path.suffix
                    result['file_name'] = file_path.name
                    
                    total_size += result['file_size_mb']
                    analysis_results[str(file_path)] = result
                    categories[category].append(result)
    
    # Generate comprehensive analysis report
    print(f"\n" + "=" * 80)
    print("📈 DATASET ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"📁 Total Datasets: {len(analysis_results)}")
    print(f"💾 Total Size: {total_size:.1f} MB")
    print(f"📊 Total Samples: {total_samples:,}")
    
    # Category breakdown
    print(f"\n🗂️ BY CATEGORY:")
    for category, datasets in categories.items():
        if datasets:
            cat_size = sum(d['file_size_mb'] for d in datasets)
            cat_samples = sum(d.get('rows', 0) for d in datasets)
            print(f"  • {category.upper():12s}: {len(datasets)} files, {cat_size:6.1f}MB, {cat_samples:8,} samples")
    
    # Detailed analysis by category
    for category, datasets in categories.items():
        if not datasets:
            continue
            
        print(f"\n📋 {category.upper()} DATASETS:")
        for dataset in datasets:
            print(f"  📄 {dataset['file_name']}")
            print(f"     Size: {dataset['file_size_mb']:.1f}MB")
            
            if 'rows' in dataset:
                print(f"     Samples: {dataset['rows']:,}")
                print(f"     Features: {dataset['columns']}")
                if dataset['column_names']:
                    print(f"     Columns: {', '.join(dataset['column_names'][:5])}{' ...' if len(dataset['column_names']) > 5 else ''}")
                
            elif 'file_count' in dataset:
                print(f"     Files: {dataset['file_count']}")
                print(f"     Uncompressed: {dataset.get('total_uncompressed_size', 0):.1f}MB")
            
            if 'error' in dataset:
                print(f"     ⚠️  Error: {dataset['error']}")
    
    # Identify coverage gaps and recommendations
    print(f"\n🎯 COVERAGE ANALYSIS:")
    
    coverage_assessment = {
        'core_benchmarks': {
            'present': [],
            'missing': []
        },
        'capability_domains': {
            'covered': set(),
            'gaps': set()
        }
    }
    
    # Check for key benchmark datasets
    key_benchmarks = {
        'MMLU': 'mmlu',
        'GSM8K': 'gsm8k', 
        'HumanEval': 'humaneval',
        'HellaSwag': 'hellaswag',
        'TruthfulQA': 'truthful',
        'WinoGrande': 'winogrande',
        'ARC': 'arc'
    }
    
    for benchmark, keyword in key_benchmarks.items():
        found = any(keyword.lower() in str(path).lower() for path in analysis_results.keys())
        if found:
            coverage_assessment['core_benchmarks']['present'].append(benchmark)
        else:
            coverage_assessment['core_benchmarks']['missing'].append(benchmark)
    
    print(f"  ✅ Present Benchmarks: {', '.join(coverage_assessment['core_benchmarks']['present'])}")
    if coverage_assessment['core_benchmarks']['missing']:
        print(f"  ❌ Missing Benchmarks: {', '.join(coverage_assessment['core_benchmarks']['missing'])}")
    
    # Assess capability coverage
    capabilities = {
        'Mathematical Reasoning': any('gsm8k' in str(p).lower() or 'math' in str(p).lower() for p in analysis_results.keys()),
        'Code Generation': any('humaneval' in str(p).lower() or 'mbpp' in str(p).lower() for p in analysis_results.keys()),
        'Knowledge Retention': any('mmlu' in str(p).lower() or 'truthful' in str(p).lower() for p in analysis_results.keys()),
        'Language Understanding': any('hellaswag' in str(p).lower() or 'winogrande' in str(p).lower() for p in analysis_results.keys()),
        'Logical Reasoning': any('arc' in str(p).lower() for p in analysis_results.keys())
    }
    
    print(f"\n🧠 CAPABILITY COVERAGE:")
    for capability, covered in capabilities.items():
        status = "✅" if covered else "❌"
        print(f"  {status} {capability}")
    
    # Generate recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    recommendations = []
    
    # Size recommendations
    if total_size < 100:
        recommendations.append("Consider adding larger datasets for more comprehensive analysis")
    elif total_size > 1000:
        recommendations.append("Dataset collection is very comprehensive - consider performance optimization")
    else:
        recommendations.append("Dataset size is well-balanced for inbreeding analysis")
    
    # Coverage recommendations  
    missing_caps = [cap for cap, covered in capabilities.items() if not covered]
    if missing_caps:
        recommendations.append(f"Add datasets for: {', '.join(missing_caps)}")
    
    # Sample size recommendations
    if total_samples < 10000:
        recommendations.append("Consider larger datasets for statistical significance")
    elif total_samples > 1000000:
        recommendations.append("Very large sample size - excellent for robust analysis")
    
    # Quality recommendations
    high_quality_datasets = sum(1 for d in analysis_results.values() if d.get('rows', 0) > 1000)
    if high_quality_datasets < 3:
        recommendations.append("Add more substantial datasets (>1000 samples each)")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Save comprehensive analysis
    analysis_output = {
        'summary': {
            'total_datasets': len(analysis_results),
            'total_size_mb': total_size,
            'total_samples': total_samples,
            'categories': {cat: len(datasets) for cat, datasets in categories.items() if datasets}
        },
        'detailed_analysis': analysis_results,
        'coverage_assessment': coverage_assessment,
        'capabilities_covered': capabilities,
        'recommendations': recommendations,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save analysis report
    with open('data/comprehensive_analysis.json', 'w') as f:
        json.dump(analysis_output, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: data/comprehensive_analysis.json")
    
    # Dataset readiness assessment
    print(f"\n🚀 READINESS ASSESSMENT:")
    
    readiness_score = 0
    max_score = 10
    
    # Core benchmarks (4 points)
    core_present = len(coverage_assessment['core_benchmarks']['present'])
    readiness_score += min(4, core_present * 4 / 7)
    
    # Capability coverage (3 points)
    caps_covered = sum(capabilities.values())
    readiness_score += min(3, caps_covered * 3 / 5)
    
    # Dataset size (2 points)
    if total_size > 50:
        readiness_score += 2
    elif total_size > 10:
        readiness_score += 1
    
    # Sample diversity (1 point)
    if total_samples > 50000:
        readiness_score += 1
    elif total_samples > 10000:
        readiness_score += 0.5
    
    readiness_percentage = (readiness_score / max_score) * 100
    
    print(f"  📊 Readiness Score: {readiness_score:.1f}/{max_score} ({readiness_percentage:.1f}%)")
    
    if readiness_percentage >= 80:
        print("  ✅ EXCELLENT - Ready for comprehensive LLM inbreeding analysis!")
    elif readiness_percentage >= 60:
        print("  ✅ GOOD - Adequate for inbreeding analysis with minor gaps")
    elif readiness_percentage >= 40:
        print("  ⚠️  MODERATE - Usable but consider adding more datasets")
    else:
        print("  ❌ INSUFFICIENT - Significant dataset gaps need addressing")
    
    return analysis_output

if __name__ == "__main__":
    analysis = main()