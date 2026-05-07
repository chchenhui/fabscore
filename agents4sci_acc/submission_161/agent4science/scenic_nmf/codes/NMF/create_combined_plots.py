#!/usr/bin/env python
"""
Create combined enrichment plots and fix metadata column names
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
import warnings
warnings.filterwarnings('ignore')
import gseapy as gp

def create_combined_enrichment_plot(output_dir='nmf_enrichment_results'):
    """
    Create a combined enrichment plot for all programs
    """
    enrich_dir = os.path.join(output_dir, 'enrichment')
    
    # Collect all enrichment results
    all_programs_data = []
    
    # Define the enrichment types
    enrichment_types = ['GO_Biological_Process', 'GO_Molecular_Function', 
                       'GO_Cellular_Component', 'KEGG']
    
    for prog_idx in range(1, 11):  # 10 programs
        prog_name = f'Program_{prog_idx}'
        
        for enrich_type in enrichment_types:
            csv_file = os.path.join(enrich_dir, f'{prog_name}_{enrich_type}_enrichment.csv')
            
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Get top 2 terms
                    top_terms = df.nsmallest(2, 'Adjusted P-value')
                    for _, row in top_terms.iterrows():
                        all_programs_data.append({
                            'Program': prog_name,
                            'Category': enrich_type.replace('_', ' '),
                            'Term': row['Term'][:40],  # Truncate long terms
                            'Adjusted_P_value': row['Adjusted P-value'],
                            'Odds_Ratio': row.get('Odds Ratio', row.get('Combined Score', 1)),
                            'neg_log_p': -np.log10(row['Adjusted P-value'] + 1e-100)
                        })
    
    if all_programs_data:
        combined_df = pd.DataFrame(all_programs_data)
        
        # Create the combined plot
        create_master_enrichment_plot(combined_df, output_dir)
        create_enrichment_heatmap(combined_df, output_dir)
        create_enrichment_dotplot(combined_df, output_dir)
        
        # Save combined data
        combined_df.to_csv(os.path.join(enrich_dir, 'all_programs_enrichment_combined.csv'), index=False)
        print("Created combined enrichment visualizations")
    else:
        print("No enrichment data found")

def create_master_enrichment_plot(combined_df, output_dir):
    """
    Create a master enrichment plot showing all programs
    """
    fig_dir = os.path.join(output_dir, 'figures')
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(24, 12))
    axes = axes.flatten()
    
    # Colors for categories
    colors = {'GO Biological Process': '#1f77b4',
             'GO Molecular Function': '#ff7f0e',
             'GO Cellular Component': '#2ca02c',
             'KEGG': '#d62728'}
    
    for idx in range(10):
        prog_name = f'Program_{idx+1}'
        ax = axes[idx]
        
        # Get data for this program
        prog_data = combined_df[combined_df['Program'] == prog_name].copy()
        
        if not prog_data.empty:
            # Sort by p-value
            prog_data = prog_data.sort_values('neg_log_p', ascending=True)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(prog_data))
            bar_colors = [colors[cat] for cat in prog_data['Category']]
            
            bars = ax.barh(y_pos, prog_data['neg_log_p'].values, color=bar_colors, alpha=0.7)
            
            # Add term labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(prog_data['Term'].values, fontsize=7)
            
            # Add significance line
            ax.axvline(x=-np.log10(0.05), color='black', linestyle='--', alpha=0.3, linewidth=0.5)
            
            ax.set_xlabel('-log10(adj. p-value)', fontsize=8)
            ax.set_title(prog_name, fontsize=10, fontweight='bold')
            ax.set_xlim(0, max(3, prog_data['neg_log_p'].max() * 1.1))
        else:
            ax.text(0.5, 0.5, 'No significant\nenrichment', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(prog_name, fontsize=10, fontweight='bold')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=cat) 
                      for cat, color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.suptitle('Enrichment Analysis: All Programs', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'all_programs_enrichment_combined.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()

def create_enrichment_heatmap(combined_df, output_dir):
    """
    Create a heatmap of enrichment p-values across programs
    """
    fig_dir = os.path.join(output_dir, 'figures')
    
    # Get top terms per program
    top_terms_per_program = {}
    for prog in combined_df['Program'].unique():
        prog_data = combined_df[combined_df['Program'] == prog]
        top_term = prog_data.nsmallest(1, 'Adjusted_P_value')
        if not top_term.empty:
            top_terms_per_program[prog] = {
                'Term': top_term.iloc[0]['Term'],
                'Category': top_term.iloc[0]['Category'],
                'neg_log_p': top_term.iloc[0]['neg_log_p']
            }
    
    # Create matrix for heatmap
    programs = [f'Program_{i}' for i in range(1, 11)]
    categories = ['GO Biological Process', 'GO Molecular Function', 
                 'GO Cellular Component', 'KEGG']
    
    heatmap_data = np.zeros((len(programs), len(categories)))
    annotations = [['' for _ in categories] for _ in programs]
    
    for i, prog in enumerate(programs):
        for j, cat in enumerate(categories):
            cat_data = combined_df[(combined_df['Program'] == prog) & 
                                  (combined_df['Category'] == cat)]
            if not cat_data.empty:
                top_val = cat_data['neg_log_p'].max()
                heatmap_data[i, j] = top_val
                top_term = cat_data.nlargest(1, 'neg_log_p')['Term'].iloc[0]
                annotations[i][j] = top_term[:15] + '...' if len(top_term) > 15 else top_term
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    # Create custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    ax = sns.heatmap(heatmap_data, 
                    xticklabels=categories,
                    yticklabels=programs,
                    cmap=cmap,
                    cbar_kws={'label': '-log10(adjusted p-value)'},
                    linewidths=0.5,
                    linecolor='gray',
                    vmin=0,
                    vmax=10)
    
    # Add significance markers
    for i in range(len(programs)):
        for j in range(len(categories)):
            if heatmap_data[i, j] > -np.log10(0.05):
                ax.text(j + 0.5, i + 0.5, '*', 
                       ha='center', va='center', color='white', fontsize=12)
    
    plt.xlabel('Enrichment Category', fontsize=12)
    plt.ylabel('Program', fontsize=12)
    plt.title('Enrichment Significance Heatmap\n(* indicates p < 0.05)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'enrichment_heatmap_all_programs.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()

def create_enrichment_dotplot(combined_df, output_dir):
    """
    Create a dot plot showing enrichment across all programs
    """
    fig_dir = os.path.join(output_dir, 'figures')
    
    # Select top 3 terms per program
    top_terms = []
    for prog in combined_df['Program'].unique():
        prog_top = combined_df[combined_df['Program'] == prog].nsmallest(3, 'Adjusted_P_value')
        top_terms.append(prog_top)
    
    if top_terms:
        plot_df = pd.concat(top_terms)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create scatter plot
        programs = plot_df['Program'].unique()
        terms = plot_df['Term'].unique()
        
        # Create position mappings
        prog_pos = {p: i for i, p in enumerate(programs)}
        term_pos = {t: i for i, t in enumerate(terms)}
        
        # Plot dots
        for _, row in plot_df.iterrows():
            x = prog_pos[row['Program']]
            y = term_pos[row['Term']]
            size = row['neg_log_p'] * 50  # Scale size
            
            # Color by category
            colors_map = {'GO Biological Process': '#1f77b4',
                         'GO Molecular Function': '#ff7f0e',
                         'GO Cellular Component': '#2ca02c',
                         'KEGG': '#d62728'}
            color = colors_map.get(row['Category'], 'gray')
            
            ax.scatter(x, y, s=size, c=[color], alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Set labels
        ax.set_xticks(range(len(programs)))
        ax.set_xticklabels(programs, rotation=45, ha='right')
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms, fontsize=8)
        
        ax.set_xlabel('Program', fontsize=12)
        ax.set_ylabel('Enriched Terms', fontsize=12)
        ax.set_title('Top Enriched Terms Across All Programs\n(dot size = -log10 p-value)', 
                    fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend for categories
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, alpha=0.6, label=cat) 
                         for cat, color in colors_map.items()]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'enrichment_dotplot_all_programs.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def fix_metadata_activity_plots(input_file='../sub_adata/strict_endocrine.h5ad'):
    """
    Recreate activity plots with correct metadata column names
    """
    print("\nRecreating activity plots with correct metadata columns...")
    
    # Load data
    adata = sc.read_h5ad(input_file)
    
    # Load usage matrix
    output_dir = 'nmf_enrichment_results'
    usage_df = pd.read_csv(os.path.join(output_dir, 'usage_matrix.csv'), index_col=0)
    
    # Normalize usage
    usage_norm = usage_df.div(usage_df.sum(axis=1), axis=0)
    
    fig_dir = os.path.join(output_dir, 'figures')
    
    # Define correct metadata columns
    metadata_mapping = {
        'cell_type': 'cell_type',
        'disease': 'disease', 
        'tissue': 'tissue'
    }
    
    for meta_type, meta_col in metadata_mapping.items():
        if meta_col in adata.obs.columns:
            print(f"  Creating activity plot for {meta_type} (column: {meta_col})...")
            
            # Get unique values
            unique_vals = adata.obs[meta_col].unique()
            print(f"    Found {len(unique_vals)} unique {meta_type} values")
            
            # Calculate mean activity per group
            activity_data = []
            group_names = []
            
            for group in unique_vals:
                cells_in_group = adata.obs[adata.obs[meta_col] == group].index
                cells_in_group = [c for c in cells_in_group if c in usage_norm.index]
                
                if cells_in_group:
                    mean_usage = usage_norm.loc[cells_in_group].mean()
                    activity_data.append(mean_usage.values)
                    group_names.append(str(group))
            
            if activity_data:
                activity_df = pd.DataFrame(activity_data, 
                                          columns=usage_df.columns,
                                          index=group_names)
                
                # Save corrected CSV
                activity_df.to_csv(os.path.join(output_dir, f'activity_by_{meta_type}_corrected.csv'))
                
                # Create heatmap
                plt.figure(figsize=(10, max(6, len(group_names)*0.3)))
                sns.heatmap(activity_df.T, 
                           annot=True, 
                           fmt='.2f',
                           cmap='RdBu_r',
                           center=0,
                           cbar_kws={'label': 'Mean Activity'},
                           linewidths=0.5,
                           linecolor='gray')
                plt.title(f'Program Activity by {meta_type.title()}', fontsize=14)
                plt.xlabel(meta_type.title(), fontsize=12)
                plt.ylabel('Program', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f'activity_by_{meta_type}_corrected.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"    Saved corrected plot for {meta_type}")
        else:
            print(f"  Warning: Column '{meta_col}' not found in adata.obs")
            print(f"    Available columns: {list(adata.obs.columns[:10])}")

if __name__ == "__main__":
    # Create combined enrichment plots
    print("Creating combined enrichment visualizations...")
    create_combined_enrichment_plot()
    
    # Fix metadata activity plots with correct column names
    fix_metadata_activity_plots()
    
    print("\nAll combined plots created successfully!")
    print("New files:")
    print("  - figures/all_programs_enrichment_combined.png")
    print("  - figures/enrichment_heatmap_all_programs.png")
    print("  - figures/enrichment_dotplot_all_programs.png")
    print("  - enrichment/all_programs_enrichment_combined.csv")
    print("  - activity_by_cell_type_corrected.png/csv")
    print("  - activity_by_disease_corrected.png/csv")
    print("  - activity_by_tissue_corrected.png/csv")