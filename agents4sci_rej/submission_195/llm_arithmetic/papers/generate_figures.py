#!/usr/bin/env python3
"""
Generate figures for the Agents4Science 2025 paper on LLM arithmetic reasoning.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Figure 1: Model Performance Comparison
def create_performance_comparison():
    """Create a comprehensive performance comparison figure."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Data for step-by-step results
    models = ['Claude-Sonnet-4\n20250514', 'Llama-4-Maverick\n17B-FP8', 'Qwen3-235B\nA22B-Instruct', 'DeepSeek-V3', 'GPT-4o',
              'Claude-3.5\nHaiku', 'Qwen3-8B', 'Qwen3-4B', 'GPT-4o\nMini', 'Qwen3-0.6B']
    step_accuracy = [100.0, 100.0, 99.5, 99.1, 96.2, 99.5, 96.7, 96.2, 92.4, 85.8]
    step_time = [6.36, 2.02, 8.62, 4.50, 4.94, 2.55, 7.16, 5.95, 3.50, 2.41]

    # Direct answer results
    direct_accuracy = [99.5, 99.1, 99.5, 95.3, 98.6, 99.1, 95.7, 87.2, 91.0, 1.4]
    direct_time = [1.24, 0.14, 0.30, 0.93, 0.50, 0.60, 0.18, 0.20, 0.41, 0.44]

    # Colors for different model types
    colors = ['#9400D3', '#2E8B57', '#DC143C', '#FF6347', '#1E90FF', '#4169E1', '#FF8C00', '#FF6347', '#8A2BE2', '#B22222']

    # Subplot 1: Step-by-Step Accuracy
    bars1 = ax1.bar(models, step_accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Step-by-Step Reasoning Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim(80, 101)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)

    # Add perfect score annotation
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(6.5, 98, 'Perfect Score', fontsize=9, color='red', fontweight='bold')

    # Add value labels on bars
    for bar, acc in zip(bars1, step_accuracy):
        height = bar.get_height()
        if height >= 100:
            ax1.text(bar.get_x() + bar.get_width()/2., height - 2,
                    f'{acc}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 2: Direct Answer Accuracy
    bars2 = ax2.bar(models, direct_accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Direct Answer Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 101)
    ax2.tick_params(axis='x', rotation=45, labelsize=10)

    # Highlight the catastrophic failure
    bars2[9].set_color('#FF0000')  # Qwen3-0.6B is now at index 9
    ax2.text(9, 15, 'Format\nCompliance\nFailure', ha='center', va='bottom',
             fontsize=8, color='red', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars2, direct_accuracy)):
        height = bar.get_height()
        if i == 9:  # Qwen3-0.6B special case
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{acc}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 3: Response Time Comparison
    x = np.arange(len(models))
    width = 0.35

    bars3a = ax3.bar(x - width/2, step_time, width, label='Step-by-Step',
                     color='lightblue', alpha=0.8, edgecolor='black')
    bars3b = ax3.bar(x + width/2, direct_time, width, label='Direct Answer',
                     color='lightcoral', alpha=0.8, edgecolor='black')

    ax3.set_ylabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Response Time Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax3.legend(fontsize=11)
    ax3.set_yscale('log')

    # Subplot 4: Speed Improvement Factors
    speed_improvements = [5.13, 14.4, 28.7, 4.84, 9.88, 4.25, 39.8, 29.75, 8.54, 5.48]  # step_time/direct_time

    bars4 = ax4.bar(models, speed_improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Speed Improvement Factor', fontsize=12, fontweight='bold')
    ax4.set_title('Speed Improvement with Direct Prompting', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=10)

    # Highlight the best improvements
    max_idx = np.argmax(speed_improvements)
    bars4[max_idx].set_color('#FFD700')  # Gold for best
#     ax4.text(max_idx, speed_improvements[max_idx] + 2, f'{speed_improvements[max_idx]:.1f}x',
#              ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add value labels for top performers
    for i, (bar, improvement) in enumerate(zip(bars4, speed_improvements)):
        if improvement > 20:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{improvement:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figure1_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Figure 2: Evaluation Pipeline Diagram
def create_pipeline_diagram():
    """Create a comprehensive evaluation pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Define colors
    input_color = '#E6F3FF'
    process_color = '#FFE6E6'
    model_color = '#E6FFE6'
    output_color = '#FFFACD'

    # Title
    ax.text(5, 12.5, 'LLM Arithmetic Reasoning Evaluation Pipeline',
            ha='center', va='center', fontsize=18, fontweight='bold')

    # Input stage
    math401_box = FancyBboxPatch((0.5, 9.5), 2, 1.5, boxstyle="round,pad=0.1",
                                 facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(math401_box)
    ax.text(1.5, 10.25, 'MATH 401\nBenchmark\n(211 Problems)', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Problem distribution
    ax.text(1.5, 9.2, '• Addition: 60\n• Subtraction: 40\n• Multiplication: 25\n• Division: 25\n• Exponentiation: 25\n• Logarithm: 25\n• Trigonometry: 10\n• Complex: 1',
            ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Prompt strategies - unified box
    prompt_box = FancyBboxPatch((3.5, 9.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                              facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(4.75, 10.6, 'Prompting Strategies', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(4.75, 10.1, 'Step-by-Step Reasoning', ha='center', va='center',
            fontsize=10)
    ax.text(4.75, 9.8, 'Direct Answer Format', ha='center', va='center',
            fontsize=10)

    # Model categories
    api_models_box = FancyBboxPatch((7, 10), 2.5, 1.8, boxstyle="round,pad=0.1",
                                    facecolor=model_color, edgecolor='black', linewidth=2)
    ax.add_patch(api_models_box)
    ax.text(8.25, 11.6, 'API Models', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(8.25, 10.75, '• Claude-Sonnet-4\n• Llama-4-Maverick-17B\n• Qwen3-235B\n• DeepSeek-V3\n• GPT-4o\n• Claude-3.5-Haiku\n• GPT-4o-Mini',
            ha='center', va='center', fontsize=8)

    local_models_box = FancyBboxPatch((7, 7.5), 2.5, 1.8, boxstyle="round,pad=0.1",
                                      facecolor=model_color, edgecolor='black', linewidth=2)
    ax.add_patch(local_models_box)
    ax.text(8.25, 8.7, 'Local Models', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(8.25, 8.2, '• Qwen3-8B\n• Qwen3-4B\n• Qwen3-0.6B',
            ha='center', va='center', fontsize=9)

    # Evaluation components
    eval_box = FancyBboxPatch((1, 5.5), 8, 1.5, boxstyle="round,pad=0.1",
                              facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(5, 6.7, 'Evaluation Framework', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.text(5, 6.2, 'Strict Pattern Matching  •  Performance Metrics  •  Statistical Analysis',
            ha='center', va='center', fontsize=11)
    ax.text(5, 5.8, 'Temperature: 0.1  •  Max Tokens: 4000  •  Timeout: 120s',
            ha='center', va='center', fontsize=10, style='italic')

    # Results
    results_box = FancyBboxPatch((2, 3), 6, 1.8, boxstyle="round,pad=0.1",
                                 facecolor='#F0E68C', edgecolor='black', linewidth=2)
    ax.add_patch(results_box)
    ax.text(5, 4.4, 'Key Findings', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.text(5, 3.8, 'Perfect 100% Accuracy Achieved (Claude-Sonnet-4 & Llama-4-Maverick)',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.5, 'Up to 39.8x Speed Improvement with Direct Prompting',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 3.2, 'Multiple Architectures Achieve Excellent Performance',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Infrastructure note
    infra_box = FancyBboxPatch((0.5, 1), 9, 1, boxstyle="round,pad=0.1",
                               facecolor='#E0E0E0', edgecolor='black', linewidth=1)
    ax.add_patch(infra_box)
    ax.text(5, 1.5, 'Infrastructure: 8× NVIDIA H100 80GB GPUs  •  CUDA 12.9  •  652GB Total VRAM',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Add arrows
    # From MATH401 to prompts
    ax.annotate('', xy=(3.5, 10.25), xytext=(2.5, 10.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # From prompts to models
    ax.annotate('', xy=(7, 10.9), xytext=(6, 10.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7, 8.4), xytext=(6, 10.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # From models to evaluation
    ax.annotate('', xy=(5, 7), xytext=(8.25, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # From evaluation to results
    ax.annotate('', xy=(5, 4.8), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    plt.savefig('figure2_evaluation_pipeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_evaluation_pipeline.png', dpi=300, bbox_inches='tight')
    plt.show()

# Figure 3: Speed vs Accuracy Trade-off Analysis
def create_speed_accuracy_tradeoff():
    """Create a speed vs accuracy trade-off scatter plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Data
    models = ['Claude-Sonnet-4-20250514', 'Llama-4-Maverick-17B-FP8', 'Qwen3-235B-A22B-Instruct', 'DeepSeek-V3', 'GPT-4o',
              'Claude-3.5-Haiku', 'Qwen3-8B', 'Qwen3-4B', 'GPT-4o-Mini', 'Qwen3-0.6B']

    step_accuracy = [100.0, 100.0, 99.5, 99.1, 96.2, 99.5, 96.7, 96.2, 92.4, 85.8]
    step_time = [6.36, 2.02, 8.62, 4.50, 4.94, 2.55, 7.16, 5.95, 3.50, 2.41]

    direct_accuracy = [99.5, 99.1, 99.5, 95.3, 98.6, 99.1, 95.7, 87.2, 91.0, 1.4]
    direct_time = [1.24, 0.14, 0.30, 0.93, 0.50, 0.60, 0.18, 0.20, 0.41, 0.44]

    # Model categories for coloring
    model_types = ['API', 'MoE-API', 'MoE-API', 'API', 'API', 'API', 'Local', 'Local', 'API', 'Local']
    colors_map = {'MoE-API': '#2E8B57', 'API': '#4169E1', 'Local': '#DC143C'}
    colors = [colors_map[t] for t in model_types]

    sizes = [160, 150, 200, 140, 130, 120, 100, 90, 110, 80]  # Size based on parameter count

    # Subplot 1: Step-by-Step Results
    scatter1 = ax1.scatter(step_time, step_accuracy, c=colors, s=sizes, alpha=0.7,
                          edgecolors='black', linewidths=2)

    # Add model labels with more descriptive names
    model_short_names = ['Claude-Sonnet-4', 'Llama-4-Maverick', 'Qwen3-235B', 'DeepSeek-V3', 'GPT-4o',
                        'Claude-3.5-Haiku', 'Qwen3-8B', 'Qwen3-4B', 'GPT-4o-Mini', 'Qwen3-0.6B']

    for i, (x, y, short_name) in enumerate(zip(step_time, step_accuracy, model_short_names)):
        ax1.annotate(short_name,
                    (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax1.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Step-by-Step Reasoning: Speed vs Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(80, 101)

    # Add perfect accuracy line
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Perfect Accuracy')

    # Subplot 2: Direct Answer Results
    scatter2 = ax1.scatter(direct_time, direct_accuracy, c=colors, s=sizes, alpha=0.7,
                          edgecolors='black', linewidths=2, marker='^')

    # Add model labels for direct answers
    for i, (x, y, short_name) in enumerate(zip(direct_time, direct_accuracy, model_short_names)):
        if i != 9:  # Skip Qwen3-0.6B (index 9) for readability
            ax2.annotate(short_name,
                        (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Special annotation for the failure case (Qwen3-0.6B is at index 9)
    ax2.annotate('Qwen3-0.6B\n(Format Failure)',
                (direct_time[9], direct_accuracy[9]), xytext=(30, 30),
                textcoords='offset points', fontsize=8, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax2.scatter(direct_time, direct_accuracy, c=colors, s=sizes, alpha=0.7,
               edgecolors='black', linewidths=2, marker='^')

    ax2.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Direct Answer: Speed vs Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 101)

    # Add legends
    legend_elements = [
        mpatches.Patch(color='#2E8B57', label='MoE API Models'),
        mpatches.Patch(color='#4169E1', label='API Models'),
        mpatches.Patch(color='#DC143C', label='Local Models'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                  markersize=8, label='Step-by-Step', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                  markersize=8, label='Direct Answer', markeredgecolor='black')
    ]

    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Add efficiency regions
    ax1.axvspan(0, 3, alpha=0.1, color='green', label='Fast Response')
    ax1.axhspan(95, 101, alpha=0.1, color='blue', label='High Accuracy')

    ax2.axvspan(0, 0.5, alpha=0.1, color='green')
    ax2.axhspan(95, 101, alpha=0.1, color='blue')

    plt.tight_layout()
    plt.savefig('figure3_speed_accuracy_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating Figure 1: Performance Comparison...")
    create_performance_comparison()

    print("Generating Figure 2: Evaluation Pipeline...")
    create_pipeline_diagram()

    print("Generating Figure 3: Speed vs Accuracy Trade-off...")
    create_speed_accuracy_tradeoff()

    print("All figures generated successfully!")
    print("Files created:")
    print("- figure1_performance_comparison.pdf/png")
    print("- figure2_evaluation_pipeline.pdf/png")
    print("- figure3_speed_accuracy_tradeoff.pdf/png")