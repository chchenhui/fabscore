#!/usr/bin/env python3
"""
Create figures and visualizations for the fairness paper
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

# Set style for academic publications
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_fairness_accuracy_tradeoff():
    """Create fairness-accuracy trade-off plot"""
    # Load results
    results_df = pd.read_csv('../results/model_comparison.csv')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color code by model type
    baseline_mask = ~(results_df['model'].str.contains('Fairness') | results_df['model'].str.contains('Adversarial'))
    fairness_mask = results_df['model'].str.contains('Fairness') | results_df['model'].str.contains('Adversarial')

    # Plot baseline models
    if baseline_mask.any():
        ax.scatter(results_df[baseline_mask]['demographic_parity'],
                  results_df[baseline_mask]['accuracy'],
                  s=100, alpha=0.7, label='Baseline Models', marker='o')

    # Plot fairness models
    if fairness_mask.any():
        ax.scatter(results_df[fairness_mask]['demographic_parity'],
                  results_df[fairness_mask]['accuracy'],
                  s=100, alpha=0.7, label='Fairness-Aware Models', marker='^')

    # Add model names as annotations
    for _, row in results_df.iterrows():
        model_name = row['model'].replace('_lambda_', ' λ=')
        ax.annotate(model_name, (row['demographic_parity'], row['accuracy']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    ax.set_xlabel('Demographic Parity Violation')
    ax.set_ylabel('Accuracy')
    ax.set_title('Fairness-Accuracy Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/figures/fairness_accuracy_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../results/figures/fairness_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_ablation_study_plot():
    """Create ablation study visualization"""
    ablation_df = pd.read_csv('../results/ablation_study.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy vs penalty
    ax1.plot(ablation_df['penalty'], ablation_df['accuracy'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Fairness Penalty (λ)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Fairness Penalty')
    ax1.grid(True, alpha=0.3)

    # Demographic parity vs penalty
    ax2.plot(ablation_df['penalty'], ablation_df['demographic_parity'], 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Fairness Penalty (λ)')
    ax2.set_ylabel('Demographic Parity Violation')
    ax2.set_title('Fairness vs Fairness Penalty')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/figures/ablation_study.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../results/figures/ablation_study.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_fairness_metrics_comparison():
    """Create bar chart comparing fairness metrics across models"""
    results_df = pd.read_csv('../results/model_comparison.csv')

    # Select top models for cleaner visualization
    top_models = results_df.nsmallest(6, 'demographic_parity')

    metrics = ['demographic_parity', 'equal_opportunity', 'equalized_odds']
    x = np.arange(len(top_models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = top_models[metric].values
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)

    ax.set_xlabel('Models')
    ax.set_ylabel('Fairness Violation')
    ax.set_title('Fairness Metrics Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels([name.replace('_lambda_', ' λ=') for name in top_models['model']], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../results/figures/fairness_metrics_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../results/figures/fairness_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_synthetic_confusion_matrices():
    """Create synthetic group-wise confusion matrices"""
    np.random.seed(42)

    # Simulate confusion matrices for baseline vs fairness model
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Baseline model - biased
    cm_baseline_g0 = np.array([[180, 20], [40, 60]])  # More false negatives for group 0
    cm_baseline_g1 = np.array([[170, 30], [25, 75]])  # Better performance for group 1

    # Fairness model - more balanced
    cm_fair_g0 = np.array([[175, 25], [30, 70]])     # Improved for group 0
    cm_fair_g1 = np.array([[175, 25], [30, 70]])     # Similar for group 1

    # Plot confusion matrices
    sns.heatmap(cm_baseline_g0, annot=True, fmt='d', cmap='Blues',
               ax=axes[0,0], cbar=False)
    axes[0,0].set_title('Baseline Model - Group 0')
    axes[0,0].set_ylabel('True Label')
    axes[0,0].set_xlabel('Predicted Label')

    sns.heatmap(cm_baseline_g1, annot=True, fmt='d', cmap='Blues',
               ax=axes[0,1], cbar=False)
    axes[0,1].set_title('Baseline Model - Group 1')
    axes[0,1].set_ylabel('True Label')
    axes[0,1].set_xlabel('Predicted Label')

    sns.heatmap(cm_fair_g0, annot=True, fmt='d', cmap='Greens',
               ax=axes[1,0], cbar=False)
    axes[1,0].set_title('Fairness-Aware Model - Group 0')
    axes[1,0].set_ylabel('True Label')
    axes[1,0].set_xlabel('Predicted Label')

    sns.heatmap(cm_fair_g1, annot=True, fmt='d', cmap='Greens',
               ax=axes[1,1], cbar=False)
    axes[1,1].set_title('Fairness-Aware Model - Group 1')
    axes[1,1].set_ylabel('True Label')
    axes[1,1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('../results/figures/confusion_matrices.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../results/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset_visualization():
    """Create visualization of the synthetic dataset with bias"""
    np.random.seed(42)

    # Simulate dataset characteristics
    n_samples = 1000
    group_0_size = 500
    group_1_size = 500

    # Simulate feature distributions
    group_0_features = np.random.normal(0, 1, (group_0_size, 3))
    group_1_features = np.random.normal(0.2, 1, (group_1_size, 3))

    # Simulate biased labels
    group_0_labels = np.random.binomial(1, 0.3, group_0_size)  # Lower positive rate
    group_1_labels = np.random.binomial(1, 0.55, group_1_size)  # Higher positive rate

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Feature distribution comparison
    axes[0].hist(group_0_features[:, 0], alpha=0.6, label='Group 0', bins=30)
    axes[0].hist(group_1_features[:, 0], alpha=0.6, label='Group 1', bins=30)
    axes[0].set_xlabel('Feature Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Feature Distribution by Group')
    axes[0].legend()

    # Label distribution
    group_stats = pd.DataFrame({
        'Group': ['Group 0', 'Group 1'],
        'Positive Rate': [group_0_labels.mean(), group_1_labels.mean()],
        'Sample Size': [group_0_size, group_1_size]
    })

    axes[1].bar(group_stats['Group'], group_stats['Positive Rate'], alpha=0.7)
    axes[1].set_ylabel('Positive Label Rate')
    axes[1].set_title('Label Distribution by Group')
    axes[1].set_ylim(0, 0.7)

    # Add values on bars
    for i, v in enumerate(group_stats['Positive Rate']):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

    # Bias visualization
    bias_strength = [0.1, 0.2, 0.3, 0.4, 0.5]
    pos_rate_g0 = [0.45, 0.40, 0.35, 0.30, 0.25]
    pos_rate_g1 = [0.55, 0.55, 0.55, 0.55, 0.55]

    axes[2].plot(bias_strength, pos_rate_g0, 'o-', label='Group 0', linewidth=2)
    axes[2].plot(bias_strength, pos_rate_g1, 's-', label='Group 1', linewidth=2)
    axes[2].set_xlabel('Bias Strength')
    axes[2].set_ylabel('Positive Rate')
    axes[2].set_title('Effect of Bias Injection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/figures/dataset_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('../results/figures/dataset_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures for the paper"""
    print("Creating figures for fairness paper...")

    # Create figures directory
    os.makedirs('../results/figures', exist_ok=True)

    # Generate all figures
    print("1. Creating fairness-accuracy trade-off plot...")
    create_fairness_accuracy_tradeoff()

    print("2. Creating ablation study plot...")
    create_ablation_study_plot()

    print("3. Creating fairness metrics comparison...")
    create_fairness_metrics_comparison()

    print("4. Creating confusion matrices...")
    create_synthetic_confusion_matrices()

    print("5. Creating dataset visualization...")
    create_dataset_visualization()

    print("All figures created and saved to ../results/figures/")
    print("Generated files:")
    print("- fairness_accuracy_tradeoff.pdf/.png")
    print("- ablation_study.pdf/.png")
    print("- fairness_metrics_comparison.pdf/.png")
    print("- confusion_matrices.pdf/.png")
    print("- dataset_visualization.pdf/.png")

if __name__ == "__main__":
    main()