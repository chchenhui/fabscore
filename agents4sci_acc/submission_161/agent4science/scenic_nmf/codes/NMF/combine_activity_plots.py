#!/usr/bin/env python
"""
Combine all activity_by_cell_type.png plots from k=10 to k=50 into a single PDF
"""

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

def combine_activity_plots_to_pdf(k_range=(10, 50), base_dir='nmf_results', output_file='activity_by_cell_type_all_k.pdf'):
    """
    Combine all activity_by_cell_type.png files into a single PDF
    
    Parameters:
    -----------
    k_range : tuple
        Range of k values (start, end inclusive)
    base_dir : str
        Base directory for results
    output_file : str
        Output PDF filename
    """
    
    print("="*60)
    print("COMBINING ACTIVITY BY CELL TYPE PLOTS")
    print("="*60)
    print(f"K range: {k_range[0]} to {k_range[1]}")
    print(f"Output: {output_file}")
    print("")
    
    # Collect all available plots
    available_plots = []
    missing_k = []
    
    for k in range(k_range[0], k_range[1] + 1):
        plot_path = os.path.join(f"{base_dir}_k{k}", "figures", "activity_by_cell_type.png")
        if os.path.exists(plot_path):
            available_plots.append((k, plot_path))
            print(f"  ✓ Found k={k}")
        else:
            missing_k.append(k)
            print(f"  ✗ Missing k={k}")
    
    if not available_plots:
        print("\nNo plots found! Make sure the analyses have completed.")
        return
    
    print(f"\nFound {len(available_plots)} plots out of {k_range[1] - k_range[0] + 1}")
    
    # Create PDF with all plots
    with PdfPages(output_file) as pdf:
        # Create a cover page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'NMF Analysis', ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, 'Activity by Cell Type', ha='center', fontsize=20)
        fig.text(0.5, 0.5, f'k = {k_range[0]} to {k_range[1]}', ha='center', fontsize=18)
        fig.text(0.5, 0.4, f'Total analyses: {len(available_plots)}', ha='center', fontsize=14)
        
        if missing_k:
            fig.text(0.5, 0.3, f'Missing k values: {", ".join(map(str, missing_k[:10]))}{"..." if len(missing_k) > 10 else ""}', 
                    ha='center', fontsize=10, color='red')
        
        fig.text(0.5, 0.1, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
                ha='center', fontsize=10, style='italic')
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Add each activity plot
        for k, plot_path in available_plots:
            # Load the image
            img = Image.open(plot_path)
            
            # Create figure with the image
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with k value
            fig.suptitle(f'k = {k}', fontsize=16, fontweight='bold', y=0.98)
            
            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"  Added k={k} to PDF")
        
        # Add summary page if we have all results
        if len(available_plots) == k_range[1] - k_range[0] + 1:
            create_summary_page(pdf, available_plots, base_dir)
    
    print(f"\n✓ PDF created successfully: {output_file}")
    print(f"  Total pages: {len(available_plots) + 1}")  # +1 for cover page
    
    return output_file

def create_summary_page(pdf, available_plots, base_dir):
    """
    Create a summary page showing trends across k values
    """
    import pandas as pd
    
    # Collect activity variance data
    k_values = []
    variances = []
    
    for k, _ in available_plots:
        activity_file = os.path.join(f"{base_dir}_k{k}", "activity_by_cell_type.csv")
        if os.path.exists(activity_file):
            df = pd.read_csv(activity_file, index_col=0)
            # Calculate variance for each program and take mean
            var = df.var(axis=0).mean()
            k_values.append(k)
            variances.append(var)
    
    if k_values:
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Variance trend
        ax1.plot(k_values, variances, 'o-', color='steelblue', markersize=6)
        ax1.set_xlabel('k (number of programs)', fontsize=12)
        ax1.set_ylabel('Mean variance across cell types', fontsize=12)
        ax1.set_title('Cell Type Specificity vs k', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(k_values, variances, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(k_values), max(k_values), 100)
        ax1.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.5, label='Trend')
        ax1.legend()
        
        # Plot 2: Bar chart of variance
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(k_values)))
        ax2.bar(k_values, variances, color=colors, alpha=0.7)
        ax2.set_xlabel('k (number of programs)', fontsize=12)
        ax2.set_ylabel('Mean variance', fontsize=12)
        ax2.set_title('Variance Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Mark optimal k values (highest variance)
        top_k_indices = np.argsort(variances)[-3:]  # Top 3
        for idx in top_k_indices:
            ax2.bar(k_values[idx], variances[idx], color='red', alpha=0.8)
        
        plt.suptitle('Summary: Cell Type Specificity Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def combine_all_plot_types(k_range=(10, 50), base_dir='nmf_results'):
    """
    Create PDFs for all three plot types: cell_type, disease, and tissue
    """
    plot_types = ['cell_type', 'disease', 'tissue']
    
    for plot_type in plot_types:
        output_file = f'activity_by_{plot_type}_all_k.pdf'
        print(f"\nCreating PDF for {plot_type}...")
        
        # Collect plots
        available_plots = []
        for k in range(k_range[0], k_range[1] + 1):
            plot_path = os.path.join(f"{base_dir}_k{k}", "figures", f"activity_by_{plot_type}.png")
            if os.path.exists(plot_path):
                available_plots.append((k, plot_path))
        
        if available_plots:
            with PdfPages(output_file) as pdf:
                # Cover page
                fig = plt.figure(figsize=(8.5, 11))
                fig.text(0.5, 0.7, 'NMF Analysis', ha='center', fontsize=24, fontweight='bold')
                fig.text(0.5, 0.6, f'Activity by {plot_type.replace("_", " ").title()}', ha='center', fontsize=20)
                fig.text(0.5, 0.5, f'k = {k_range[0]} to {k_range[1]}', ha='center', fontsize=18)
                fig.text(0.5, 0.4, f'Total analyses: {len(available_plots)}', ha='center', fontsize=14)
                plt.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Add plots
                for k, plot_path in available_plots:
                    img = Image.open(plot_path)
                    fig = plt.figure(figsize=(11, 8.5))
                    ax = fig.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')
                    fig.suptitle(f'k = {k}', fontsize=16, fontweight='bold', y=0.98)
                    pdf.savefig(fig, bbox_inches='tight', dpi=150)
                    plt.close()
            
            print(f"  ✓ Created: {output_file}")

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Combine activity plots into PDF')
    parser.add_argument('--k-start', type=int, default=10, help='Starting k value')
    parser.add_argument('--k-end', type=int, default=50, help='Ending k value')
    parser.add_argument('--base-dir', type=str, default='nmf_results', help='Base directory for results')
    parser.add_argument('--output', type=str, default='activity_by_cell_type_all_k.pdf', help='Output PDF filename')
    parser.add_argument('--all-types', action='store_true', help='Create PDFs for all plot types')
    
    args = parser.parse_args()
    
    if args.all_types:
        # Create PDFs for all three types
        combine_all_plot_types((args.k_start, args.k_end), args.base_dir)
    else:
        # Create PDF for cell_type only
        combine_activity_plots_to_pdf((args.k_start, args.k_end), args.base_dir, args.output)