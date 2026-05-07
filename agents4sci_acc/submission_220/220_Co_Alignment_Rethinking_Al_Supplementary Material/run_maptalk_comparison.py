#!/usr/bin/env python3
"""
MapTalk Comparison Runner
Runs single directional baseline first, then BiCA, and generates comparison report
"""

import os
import sys
import subprocess
import time
import json
import yaml
import argparse
from pathlib import Path

def run_experiment(config_path, experiment_name):
    """Run a single experiment and return the wandb run directory"""
    print(f"\n{'='*60}")
    print(f" Starting {experiment_name}")
    print(f" Config: {config_path}")
    print(f"{'='*60}\n")
    
    # Record start time
    start_time = time.time()
    
    # Run the experiment
    cmd = [sys.executable, "bica/train_maptalk.py", "--config", config_path]
    process = subprocess.run(cmd, capture_output=False, text=True)
    
    # Record end time
    end_time = time.time()
    duration = end_time - start_time
    
    if process.returncode == 0:
        print(f"\n {experiment_name} completed successfully!")
        print(f"  Duration: {duration/60:.1f} minutes")
        
        # Find the latest wandb run directory
        wandb_dir = Path("wandb")
        if wandb_dir.exists():
            run_dirs = [d for d in wandb_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
            if run_dirs:
                latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                print(f" Results saved in: {latest_run}")
                return str(latest_run)
    else:
        print(f"\n {experiment_name} failed with return code {process.returncode}")
        return None
    
    return None

def load_results(wandb_run_dir):
    """Load results from wandb run directory"""
    if not wandb_run_dir:
        return None
        
    summary_file = Path(wandb_run_dir) / "files" / "wandb-summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None

def generate_comparison_report(baseline_results, bica_results, baseline_run_dir, bica_run_dir):
    """Generate comparison report"""
    print(f"\n{'='*60}")
    print(" MAPTALK EXPERIMENT COMPARISON REPORT")
    print(f"{'='*60}")
    
    if not baseline_results or not bica_results:
        print(" Missing results data - cannot generate comparison")
        return
    
    # Extract key metrics
    metrics = [
        ('success_rate', 'Task Success Rate', '%'),
        ('eval_id_success_rate', 'In-Domain Success Rate', '%'),
        ('eval_ood_success_rate', 'Out-of-Domain Success Rate', '%'),
        ('episode_reward_mean', 'Average Episode Reward', ''),
        ('episode_length_mean', 'Average Episode Length', 'steps'),
        ('ai_total_loss', 'AI Total Loss', ''),
        ('human_loss', 'Human Loss', ''),
    ]
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'BiCA':<15} {'Improvement':<15}")
    print("-" * 70)
    
    improvements = {}
    
    for key, name, unit in metrics:
        baseline_val = baseline_results.get(key, 0)
        bica_val = bica_results.get(key, 0)
        
        if baseline_val != 0:
            improvement = ((bica_val - baseline_val) / abs(baseline_val)) * 100
        else:
            improvement = float('inf') if bica_val > 0 else 0
            
        improvements[key] = improvement
        
        if unit == '%':
            baseline_str = f"{baseline_val*100:.1f}%"
            bica_str = f"{bica_val*100:.1f}%"
        elif unit == 'steps':
            baseline_str = f"{baseline_val:.1f}"
            bica_str = f"{bica_val:.1f}"
        else:
            baseline_str = f"{baseline_val:.3f}"
            bica_str = f"{bica_val:.3f}"
            
        if improvement == float('inf'):
            imp_str = "INF%"
        elif improvement == float('-inf'):
            imp_str = "-INF%"
        else:
            imp_str = f"{improvement:+.1f}%"
            
        print(f"{name:<25} {baseline_str:<15} {bica_str:<15} {imp_str:<15}")
    
    # Co-alignment specific metrics
    print(f"\n{'='*60}")
    print(" CO-ALIGNMENT MECHANISMS")
    print(f"{'='*60}")
    
    co_alignment_metrics = [
        ('protocol_loss', 'Protocol Learning Loss'),
        ('rep_wasserstein_loss', 'Representation Alignment (Wasserstein)'),
        ('rep_cca_loss', 'Representation Alignment (CCA)'),
        ('instructor_loss', 'Adaptive Teaching Loss'),
        ('ai_kl_prior_loss', 'AI KL Prior Loss'),
        ('human_kl_loss', 'Human KL Loss'),
    ]
    
    print(f"\n{'Mechanism':<35} {'Baseline':<15} {'BiCA':<15}")
    print("-" * 65)
    
    for key, name in co_alignment_metrics:
        baseline_val = baseline_results.get(key, 0)
        bica_val = bica_results.get(key, 0)
        
        baseline_str = f"{baseline_val:.4f}"
        bica_str = f"{bica_val:.4f}"
        
        status = " Active" if bica_val > 0.001 else " Disabled"
        
        print(f"{name:<35} {baseline_str:<15} {bica_str:<15} {status}")
    
    # Save detailed report
    report = {
        "experiment_type": "maptalk_comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {
            "run_dir": baseline_run_dir,
            "results": baseline_results
        },
        "bica": {
            "run_dir": bica_run_dir,
            "results": bica_results
        },
        "improvements": improvements,
        "summary": {
            "success_rate_improvement": improvements.get('success_rate', 0),
            "ood_robustness_improvement": improvements.get('eval_ood_success_rate', 0),
            "reward_improvement": improvements.get('episode_reward_mean', 0),
            "co_alignment_active": {
                "protocol_learning": bica_results.get('protocol_loss', 0) > 0.001,
                "representation_alignment": bica_results.get('rep_wasserstein_loss', 0) > 0.001,
                "adaptive_teaching": bica_results.get('instructor_loss', 0) > 0.001,
            }
        }
    }
    
    # Save to file
    os.makedirs("results", exist_ok=True)
    report_file = "results/maptalk_comparison_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n Detailed report saved to: {report_file}")
    
    # Print conclusion
    print(f"\n{'='*60}")
    print(" CONCLUSION")
    print(f"{'='*60}")
    
    success_improvement = improvements.get('success_rate', 0)
    if success_improvement > 0:
        print(f" BiCA shows {success_improvement:.1f}% improvement in task success rate")
    else:
        print(f"  BiCA shows {success_improvement:.1f}% change in task success rate")
        
    ood_improvement = improvements.get('eval_ood_success_rate', 0)
    if ood_improvement > 0:
        print(f" BiCA shows {ood_improvement:.1f}% improvement in OOD robustness")
    else:
        print(f"  BiCA shows {ood_improvement:.1f}% change in OOD robustness")
    
    active_mechanisms = sum(1 for v in report['summary']['co_alignment_active'].values() if v)
    print(f" {active_mechanisms}/3 co-alignment mechanisms are active in BiCA")
    
    if success_improvement > 0 and active_mechanisms > 0:
        print("\n RESULT: BiCA's co-alignment approach shows measurable benefits over single-directional baseline!")
    else:
        print("\n RESULT: Results require further analysis - consider longer training or different hyperparameters")

def update_config_epochs(config_path, epochs):
    """Update the episodes count in config file based on epochs"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate episodes: epochs * batch_episodes
    batch_episodes = config.get('train', {}).get('batch_episodes', 32)
    episodes = epochs * batch_episodes
    
    # Update episodes
    if 'train' not in config:
        config['train'] = {}
    config['train']['episodes'] = episodes
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    print(f" Updated {config_path}: epochs={epochs}, episodes={episodes}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='MapTalk Sequential Comparison Experiment')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs (default: 5)')
    
    args = parser.parse_args()
    
    print("MapTalk Sequential Comparison Experiment")
    print("=" * 60)
    print(f"Training epochs: {args.epochs}")
    print("=" * 60)
    
    # Configuration files
    baseline_config = "bica/configs/maptalk_one_way_baseline.yaml"
    bica_config = "bica/configs/maptalk_main.yaml"
    
    # Verify config files exist
    if not os.path.exists(baseline_config):
        print(f" Baseline config not found: {baseline_config}")
        return 1
        
    if not os.path.exists(bica_config):
        print(f" BiCA config not found: {bica_config}")
        return 1
    
    # Update config files with specified epochs
    print(f" Updating configuration files for {args.epochs} epochs...")
    update_config_epochs(baseline_config, args.epochs)
    update_config_epochs(bica_config, args.epochs)
    
    # Step 1: Run Single Directional Baseline
    print("Step 1: Running Single Directional Baseline")
    baseline_run_dir = run_experiment(baseline_config, "Single Directional Baseline")
    
    if not baseline_run_dir:
        print(" Baseline experiment failed - aborting comparison")
        return 1
    
    # Step 2: Run BiCA
    print("\nStep 2: Running BiCA (Co-Alignment)")
    bica_run_dir = run_experiment(bica_config, "BiCA Co-Alignment")
    
    if not bica_run_dir:
        print(" BiCA experiment failed - aborting comparison")
        return 1
    
    # Step 3: Load results
    print("\nStep 3: Loading and Comparing Results")
    baseline_results = load_results(baseline_run_dir)
    bica_results = load_results(bica_run_dir)
    
    # Step 4: Generate comparison report
    generate_comparison_report(baseline_results, bica_results, baseline_run_dir, bica_run_dir)
    
    print(f"\n{'='*60}")
    print(" MapTalk Comparison Experiment Completed!")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
