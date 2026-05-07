"""
Plotting and visualization module for experimental results
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

def generate_all_plots(data: Dict[str, Any], output_dir: Path):
    """Generate all visualization plots"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    
    # Generate individual plots
    generate_gpa_error_plot(data, output_dir)
    generate_decision_confusion_matrix(data, output_dir)
    generate_processing_time_plot(data, output_dir)
    generate_baseline_comparison(data, output_dir)
    
    logger.info(f"Generated plots in {output_dir}")

def generate_gpa_error_plot(data: Dict[str, Any], output_dir: Path):
    """Generate GPA prediction error histogram"""
    main_results = data.get("main_results", {}).get("transcript_results", [])
    if not main_results:
        return
    
    true_gpas = [r["ground_truth_gpa"] for r in main_results]
    pred_gpas = [r["predicted_gpa"] for r in main_results]
    errors = np.array(pred_gpas) - np.array(true_gpas)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('GPA Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of GPA Prediction Errors')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mae = np.mean(np.abs(errors))
    plt.text(0.02, 0.98, f'MAE: {mae:.3f}', transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save as both PNG and PDF
    plt.savefig(output_dir / 'gpa_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'gpa_error_distribution.pdf', bbox_inches='tight')
    plt.close()

def generate_decision_confusion_matrix(data: Dict[str, Any], output_dir: Path):
    """Generate decision confusion matrix visualization"""
    main_results = data.get("main_results", {}).get("transcript_results", [])
    if not main_results:
        return
    
    # Create simple decision accuracy plot instead of full confusion matrix
    decisions = [r["decision"] for r in main_results]
    decision_counts = {}
    for decision in decisions:
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    plt.figure(figsize=(8, 6))
    decisions_list = list(decision_counts.keys())
    counts = list(decision_counts.values())
    
    bars = plt.bar(decisions_list, counts, alpha=0.7, edgecolor='black')
    plt.xlabel('Decision Type')
    plt.ylabel('Count')
    plt.title('Distribution of Academic Decisions')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decision_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'decision_distribution.pdf', bbox_inches='tight')
    plt.close()

def generate_processing_time_plot(data: Dict[str, Any], output_dir: Path):
    """Generate processing time comparison"""
    main_results = data.get("main_results", {})
    processing_times = main_results.get("processing_times", [])
    
    if not processing_times:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of processing times
    plt.subplot(1, 2, 1)
    plt.hist(processing_times, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Processing Time (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Processing Times')
    plt.grid(True, alpha=0.3)
    
    # Plot comparison with manual time
    plt.subplot(1, 2, 2)
    manual_time = 20 * 60  # 20 minutes in seconds
    avg_auto_time = np.mean(processing_times)
    
    methods = ['Manual Review', 'Automated Processing']
    times = [manual_time, avg_auto_time]
    colors = ['red', 'green']
    
    bars = plt.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time Comparison')
    plt.yscale('log')  # Log scale due to large difference
    
    # Add time labels
    for bar, time in zip(bars, times):
        if time > 60:
            label = f'{time/60:.1f} min'
        else:
            label = f'{time:.1f} s'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                label, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'processing_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'processing_time_analysis.pdf', bbox_inches='tight')
    plt.close()

def generate_baseline_comparison(data: Dict[str, Any], output_dir: Path):
    """Generate baseline comparison bar chart"""
    metrics = data.get("metrics", {})
    
    # Extract metrics for comparison
    methods = []
    gpa_maes = []
    decision_accs = []
    
    for key, metric_data in metrics.items():
        if isinstance(metric_data, dict) and 'gpa_mae' in metric_data:
            if 'baseline_random' in key:
                methods.append('Random')
            elif 'baseline_gpa_only' in key:
                methods.append('GPA-Only')
            elif 'main_pipeline' in key:
                methods.append('Proposed IDP')
            else:
                continue
            
            gpa_maes.append(metric_data.get('gpa_mae', 0))
            decision_accs.append(metric_data.get('decision_accuracy', 0))
    
    if not methods:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # GPA MAE comparison
    bars1 = ax1.bar(methods, gpa_maes, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('GPA Mean Absolute Error')
    ax1.set_title('GPA Prediction Accuracy')
    ax1.set_ylim(0, max(gpa_maes) * 1.2)
    
    # Add value labels
    for bar, mae in zip(bars1, gpa_maes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mae:.3f}', ha='center', va='bottom')
    
    # Decision accuracy comparison
    bars2 = ax2.bar(methods, decision_accs, alpha=0.7, edgecolor='black', color='orange')
    ax2.set_ylabel('Decision Accuracy')
    ax2.set_title('Academic Decision Accuracy')
    ax2.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, acc in zip(bars2, decision_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_comparison.pdf', bbox_inches='tight')
    plt.close()

def generate_ui_overview_figure(output_dir: Path):
    """Generate UI overview schematic"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw UI components as rectangles
    components = [
        {"name": "Upload Tab", "pos": (1, 7), "size": (2, 1), "color": "lightblue"},
        {"name": "Dashboard Tab", "pos": (4, 7), "size": (2, 1), "color": "lightgreen"},
        {"name": "Detail Tab", "pos": (7, 7), "size": (2, 1), "color": "lightyellow"},
        {"name": "Chat Bot", "pos": (10, 7), "size": (2, 1), "color": "lightcoral"},
        
        {"name": "File Upload", "pos": (0.5, 5), "size": (3, 1), "color": "aliceblue"},
        {"name": "Applications Table", "pos": (4.5, 5), "size": (3, 1), "color": "honeydew"},
        {"name": "Evidence Viewer", "pos": (8.5, 5), "size": (3, 1), "color": "lightyellow"},
        
        {"name": "OCR Engine", "pos": (1, 3), "size": (2, 0.8), "color": "lightgray"},
        {"name": "Parser", "pos": (4, 3), "size": (2, 0.8), "color": "lightgray"},
        {"name": "Decision Engine", "pos": (7, 3), "size": (2, 0.8), "color": "lightgray"},
        {"name": "Results Storage", "pos": (10, 3), "size": (2, 0.8), "color": "lightgray"},
        
        {"name": "Incoming Documents", "pos": (2, 1), "size": (3, 0.8), "color": "mistyrose"},
        {"name": "Processed Results", "pos": (7, 1), "size": (3, 0.8), "color": "lightsteelblue"},
    ]
    
    # Draw components
    for comp in components:
        rect = plt.Rectangle(comp["pos"], comp["size"][0], comp["size"][1], 
                           facecolor=comp["color"], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add text
        text_x = comp["pos"][0] + comp["size"][0] / 2
        text_y = comp["pos"][1] + comp["size"][1] / 2
        ax.text(text_x, text_y, comp["name"], ha='center', va='center', fontsize=9, weight='bold')
    
    # Draw arrows to show data flow
    arrow_props = dict(arrowstyle='->', lw=2, color='darkblue')
    
    # Upload to processing
    ax.annotate('', xy=(2, 3.8), xytext=(2, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 3.8), xytext=(5, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 3.8), xytext=(8, 4.5), arrowprops=arrow_props)
    
    # Processing pipeline
    ax.annotate('', xy=(4, 3.4), xytext=(3, 3.4), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 3.4), xytext=(6, 3.4), arrowprops=arrow_props)
    ax.annotate('', xy=(10, 3.4), xytext=(9, 3.4), arrowprops=arrow_props)
    
    # Storage to display
    ax.annotate('', xy=(8.5, 4.5), xytext=(11, 3.8), arrowprops=arrow_props)
    
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.set_title('IDP System UI Architecture Overview', fontsize=14, weight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ui_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ui_overview.pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Test plot generation
    output_dir = Path("test_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Generate UI overview
    generate_ui_overview_figure(output_dir)
    print(f"Generated test plots in {output_dir}")