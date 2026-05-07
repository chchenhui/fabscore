#!/usr/bin/env python
"""
Create PDFs for all requested plot types across k=10 to k=50
"""

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np
import pandas as pd

def create_pdf_for_plot_type(plot_type, k_range=(10, 50), base_dir='nmf_results'):
    """
    Create a PDF for a specific plot type across all k values
    
    Parameters:
    -----------
    plot_type : str
        Type of plot (e.g., 'activity_by_disease', 'top_genes_per_program')
    k_range : tuple
        Range of k values (start, end inclusive)
    base_dir : str
        Base directory for results
    """
    
    output_file = f'{plot_type}_k{k_range[0]}_to_k{k_range[1]}.pdf'
    
    print(f"\n{'='*60}")
    print(f"Creating PDF for: {plot_type}")
    print(f"K range: {k_range[0]} to {k_range[1]}")
    print(f"Output: {output_file}")
    print('='*60)
    
    # Collect all available plots
    available_plots = []
    missing_k = []
    
    for k in range(k_range[0], k_range[1] + 1):
        plot_path = os.path.join(f"{base_dir}_k{k}", "figures", f"{plot_type}.png")
        if os.path.exists(plot_path):
            available_plots.append((k, plot_path))
            print(f"  ✓ Found k={k}")
        else:
            missing_k.append(k)
            print(f"  ✗ Missing k={k}")
    
    if not available_plots:
        print(f"\n⚠ No plots found for {plot_type}!")
        return None
    
    print(f"\nFound {len(available_plots)} plots out of {k_range[1] - k_range[0] + 1}")
    
    # Create PDF with all plots
    with PdfPages(output_file) as pdf:
        # Create a cover page
        create_cover_page(pdf, plot_type, k_range, len(available_plots), missing_k)
        
        # Add each plot
        for idx, (k, plot_path) in enumerate(available_plots):
            try:
                # Load the image
                img = Image.open(plot_path)
                
                # Create figure with the image
                if 'enrichment' in plot_type or 'top_genes' in plot_type:
                    # Larger figures need different sizing
                    fig = plt.figure(figsize=(16, 12))
                else:
                    fig = plt.figure(figsize=(11, 8.5))
                
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                
                # Add title with k value and page number
                fig.suptitle(f'k = {k}', fontsize=16, fontweight='bold', y=0.98)
                fig.text(0.95, 0.02, f'Page {idx+2}/{len(available_plots)+1}', 
                        ha='right', fontsize=8, style='italic')
                
                # Save to PDF
                pdf.savefig(fig, bbox_inches='tight', dpi=150)
                plt.close()
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx+1}/{len(available_plots)} plots...")
                    
            except Exception as e:
                print(f"  ⚠ Error processing k={k}: {e}")
        
        # Add summary page if applicable
        if 'activity' in plot_type:
            create_activity_summary_page(pdf, plot_type, available_plots, base_dir)
        elif plot_type == 'top_genes_per_program':
            create_top_genes_summary_page(pdf, available_plots, base_dir)
    
    print(f"\n✓ PDF created successfully: {output_file}")
    print(f"  Total pages: {len(available_plots) + 1}")
    
    return output_file

def create_cover_page(pdf, plot_type, k_range, n_plots, missing_k):
    """
    Create a cover page for the PDF
    """
    fig = plt.figure(figsize=(8.5, 11))
    
    # Title
    fig.text(0.5, 0.75, 'NMF Analysis Results', ha='center', fontsize=24, fontweight='bold')
    
    # Plot type
    plot_title = plot_type.replace('_', ' ').title()
    if plot_type == 'all_programs_enrichment_combined':
        plot_title = 'Enrichment Analysis (All Programs)'
    elif plot_type == 'top_genes_per_program':
        plot_title = 'Top Genes per Program'
        
    fig.text(0.5, 0.65, plot_title, ha='center', fontsize=20)
    
    # K range
    fig.text(0.5, 0.55, f'k = {k_range[0]} to {k_range[1]}', ha='center', fontsize=18)
    
    # Statistics
    fig.text(0.5, 0.45, f'Total analyses included: {n_plots}', ha='center', fontsize=14)
    
    if missing_k:
        missing_text = f'Missing k values: {", ".join(map(str, missing_k[:10]))}'
        if len(missing_k) > 10:
            missing_text += f'... ({len(missing_k)} total)'
        fig.text(0.5, 0.35, missing_text, ha='center', fontsize=10, color='red')
    
    # Description based on plot type
    descriptions = {
        'activity_by_disease': 'Program activity patterns across different disease conditions',
        'activity_by_tissue': 'Program activity patterns across different tissue types',
        'activity_by_cell_type': 'Program activity patterns across different cell types',
        'top_genes_per_program': 'Top 20 genes for each NMF program',
        'all_programs_enrichment_combined': 'GO and KEGG enrichment analysis for all programs'
    }
    
    if plot_type in descriptions:
        fig.text(0.5, 0.25, descriptions[plot_type], ha='center', fontsize=12, style='italic')
    
    # Timestamp
    fig.text(0.5, 0.1, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 
            ha='center', fontsize=10, style='italic')
    
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_activity_summary_page(pdf, plot_type, available_plots, base_dir):
    """
    Create a summary page for activity plots showing trends
    """
    # Extract the metadata type (disease, tissue, cell_type)
    meta_type = plot_type.replace('activity_by_', '')
    
    # Collect variance data
    k_values = []
    variances = []
    n_groups = []
    
    for k, _ in available_plots:
        activity_file = os.path.join(f"{base_dir}_k{k}", f"{plot_type}.csv")
        if os.path.exists(activity_file):
            df = pd.read_csv(activity_file, index_col=0)
            # Calculate variance for each program and take mean
            var = df.var(axis=0).mean()
            k_values.append(k)
            variances.append(var)
            n_groups.append(len(df))
    
    if k_values:
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Variance trend
        ax1 = axes[0, 0]
        ax1.plot(k_values, variances, 'o-', color='steelblue', markersize=6)
        ax1.set_xlabel('k (number of programs)', fontsize=11)
        ax1.set_ylabel(f'Mean variance across {meta_type}', fontsize=11)
        ax1.set_title(f'{meta_type.title()} Specificity vs k', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add polynomial trend
        if len(k_values) > 3:
            z = np.polyfit(k_values, variances, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(k_values), max(k_values), 100)
            ax1.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.5, label='Trend')
            ax1.legend()
        
        # Plot 2: Variance distribution
        ax2 = axes[0, 1]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(k_values)))
        bars = ax2.bar(k_values, variances, color=colors, alpha=0.7)
        ax2.set_xlabel('k (number of programs)', fontsize=11)
        ax2.set_ylabel('Mean variance', fontsize=11)
        ax2.set_title('Variance Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Highlight top 3 k values
        if len(variances) > 3:
            top_indices = np.argsort(variances)[-3:]
            for idx in top_indices:
                bars[idx].set_color('red')
                bars[idx].set_alpha(0.9)
        
        # Plot 3: Optimal k suggestion
        ax3 = axes[1, 0]
        # Calculate score combining variance and k (prefer higher variance, moderate k)
        scores = np.array(variances) / np.max(variances) if np.max(variances) > 0 else variances
        ax3.plot(k_values, scores, 'o-', color='purple', markersize=6)
        ax3.set_xlabel('k (number of programs)', fontsize=11)
        ax3.set_ylabel('Normalized specificity score', fontsize=11)
        ax3.set_title('Specificity Score', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # Mark optimal k values
        optimal_k = k_values[np.argmax(scores)]
        ax3.scatter([optimal_k], [scores[k_values.index(optimal_k)]], 
                   color='red', s=100, zorder=5)
        ax3.annotate(f'Optimal k={optimal_k}', 
                    xy=(optimal_k, scores[k_values.index(optimal_k)]),
                    xytext=(optimal_k+2, scores[k_values.index(optimal_k)]+0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        # Plot 4: Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create statistics summary
        stats_text = f"Statistics Summary\n" + "="*30 + "\n"
        stats_text += f"K range analyzed: {min(k_values)} - {max(k_values)}\n"
        stats_text += f"Number of {meta_type} groups: {n_groups[0]}\n"
        stats_text += f"Highest variance k: {k_values[np.argmax(variances)]}\n"
        stats_text += f"Mean variance: {np.mean(variances):.4f}\n"
        stats_text += f"Variance range: {min(variances):.4f} - {max(variances):.4f}\n"
        
        # Add top 5 k values by variance
        top_5_idx = np.argsort(variances)[-5:][::-1]
        stats_text += f"\nTop 5 k values by specificity:\n"
        for i, idx in enumerate(top_5_idx, 1):
            stats_text += f"  {i}. k={k_values[idx]} (var={variances[idx]:.4f})\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Summary Analysis: {plot_type.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def create_top_genes_summary_page(pdf, available_plots, base_dir):
    """
    Create a summary page for top genes analysis
    """
    fig = plt.figure(figsize=(11, 8.5))
    
    # Collect information about number of programs
    k_values = [k for k, _ in available_plots]
    
    # Create text summary
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    summary_text = "Top Genes Analysis Summary\n" + "="*40 + "\n\n"
    summary_text += f"K values analyzed: {min(k_values)} - {max(k_values)}\n"
    summary_text += f"Total analyses: {len(k_values)}\n\n"
    
    summary_text += "Key Observations:\n"
    summary_text += "• Lower k values (10-20): Broader, more general programs\n"
    summary_text += "• Medium k values (20-35): Balanced specificity\n"
    summary_text += "• Higher k values (35-50): More specialized programs\n\n"
    
    summary_text += "Program Resolution:\n"
    for k_range, description in [
        ((10, 15), "Basic cell type programs"),
        ((15, 25), "Cell type + functional programs"),
        ((25, 35), "Detailed functional states"),
        ((35, 50), "Fine-grained subtypes and states")
    ]:
        k_in_range = [k for k in k_values if k_range[0] <= k <= k_range[1]]
        if k_in_range:
            summary_text += f"  k={k_range[0]}-{k_range[1]}: {description} ({len(k_in_range)} analyses)\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top')
    
    plt.title('Top Genes per Program: Summary', fontsize=16, fontweight='bold', pad=20)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_all_pdfs(k_range=(10, 50), base_dir='nmf_results'):
    """
    Create PDFs for all requested plot types
    """
    plot_types = [
        'activity_by_disease',
        'activity_by_tissue', 
        'top_genes_per_program',
        'all_programs_enrichment_combined'
    ]
    
    created_pdfs = []
    
    for plot_type in plot_types:
        pdf_file = create_pdf_for_plot_type(plot_type, k_range, base_dir)
        if pdf_file:
            created_pdfs.append(pdf_file)
    
    print("\n" + "="*60)
    print("ALL PDFs CREATED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    for pdf_file in created_pdfs:
        if os.path.exists(pdf_file):
            size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
            print(f"  ✓ {pdf_file} ({size_mb:.1f} MB)")
    
    return created_pdfs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create PDFs for NMF analysis plots')
    parser.add_argument('--k-start', type=int, default=10, help='Starting k value')
    parser.add_argument('--k-end', type=int, default=50, help='Ending k value')
    parser.add_argument('--base-dir', type=str, default='nmf_results', help='Base directory for results')
    parser.add_argument('--plot-type', type=str, help='Specific plot type to process')
    
    args = parser.parse_args()
    
    if args.plot_type:
        # Process specific plot type
        create_pdf_for_plot_type(args.plot_type, (args.k_start, args.k_end), args.base_dir)
    else:
        # Process all requested plot types
        create_all_pdfs((args.k_start, args.k_end), args.base_dir)