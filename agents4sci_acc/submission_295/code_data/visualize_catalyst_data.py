#!/usr/bin/env python3
"""
Visualization script for catalyst data
Creates comprehensive plots showing the distribution of catalyst candidates
across stability (mixing enthalpy) and reactivity (d-band center) space
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set style
plt.style.use('seaborn-v0_8-deep')
sns.set_palette("husl")

def load_and_analyze_data(filepath):
    """Load catalyst data and perform basic analysis"""
    df = pd.read_csv(filepath)
    
    print("Data Summary:")
    print(f"Total catalysts: {len(df)}")
    print("\nCatalyst types:")
    print(df['catalyst_type'].value_counts())
    
    print("\nProperty ranges:")
    print(f"Mixing enthalpy: [{df['mixing_enthalpy_ev_atom'].min():.3f}, {df['mixing_enthalpy_ev_atom'].max():.3f}] eV/atom")
    print(f"d-band center: [{df['d_band_center_ev'].min():.3f}, {df['d_band_center_ev'].max():.3f}] eV")
    
    return df

def create_scatter_plot(df, save_path='catalyst_scatter.png'):
    """Create main scatter plot with all catalyst types"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors and markers for each catalyst type
    colors = {
        'Known': '#1f77b4',
        'LLM_Generated_HEA': '#ff7f0e', 
        'LLM_Generated_DA': '#2ca02c'
    }
    
    markers = {
        'Known': 'o',
        'LLM_Generated_HEA': 's',
        'LLM_Generated_DA': '^'
    }
    
    labels = {
        'Known': 'Known Catalysts',
        'LLM_Generated_HEA': 'LLM-Generated HEAs',
        'LLM_Generated_DA': 'LLM-Generated Doped Alloys'
    }
    
    # Plot each catalyst type
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]
        ax.scatter(data['d_band_center_ev'], 
                  data['mixing_enthalpy_ev_atom'],
                  c=colors[catalyst_type],
                  marker=markers[catalyst_type],
                  s=80,
                  alpha=0.7,
                  edgecolors='black',
                  linewidth=0.5,
                  label=labels[catalyst_type])
    
    # Add axis labels and title
    ax.set_xlabel('d-band Center (eV)', fontsize=14)
    ax.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=14)
    ax.set_title('Catalyst Discovery Space: Stability vs Reactivity', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add annotations for regions
    ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(-4.2, -0.45, 'More Stable', fontsize=10, style='italic', color='red')
    ax.text(-4.2, -0.05, 'Less Stable', fontsize=10, style='italic', color='red')
    
    ax.axvline(x=-2.5, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(-2.4, -1.0, 'Higher\nActivity', fontsize=10, style='italic', color='blue', ha='left')
    ax.text(-2.6, -1.0, 'Lower\nActivity', fontsize=10, style='italic', color='blue', ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_density_plot(df, save_path='catalyst_density.png'):
    """Create 2D density plot showing concentration of catalysts"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create density plot for known catalysts
    known_data = df[df['catalyst_type'] == 'Known']
    x = known_data['d_band_center_ev'].values
    y = known_data['mixing_enthalpy_ev_atom'].values
    
    # Calculate point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Create scatter plot with density coloring
    scatter = ax.scatter(x, y, c=z, s=50, cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add contour lines
    xi = np.linspace(x.min()-0.5, x.max()+0.5, 100)
    yi = np.linspace(y.min()-0.1, y.max()+0.1, 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Evaluate density on grid
    positions = np.vstack([Xi.ravel(), Yi.ravel()])
    kernel = gaussian_kde(xy)
    Zi = np.reshape(kernel(positions).T, Xi.shape)
    
    # Add contour lines
    contours = ax.contour(Xi, Yi, Zi, levels=5, colors='black', alpha=0.3, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.2f')
    
    # Overlay LLM-generated catalysts
    llm_hea = df[df['catalyst_type'] == 'LLM_Generated_HEA']
    ax.scatter(llm_hea['d_band_center_ev'], llm_hea['mixing_enthalpy_ev_atom'],
              marker='s', s=100, c='red', alpha=0.8, edgecolors='darkred', 
              linewidth=1.5, label='LLM-Generated HEAs')
    
    llm_da = df[df['catalyst_type'] == 'LLM_Generated_DA']
    ax.scatter(llm_da['d_band_center_ev'], llm_da['mixing_enthalpy_ev_atom'],
              marker='^', s=100, c='orange', alpha=0.8, edgecolors='darkorange',
              linewidth=1.5, label='LLM-Generated Doped Alloys')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density of Known Catalysts', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('d-band Center (eV)', fontsize=14)
    ax.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=14)
    ax.set_title('Catalyst Density Map with LLM-Generated Candidates', fontsize=16, pad=20)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_distribution_plots(df, save_path='catalyst_distributions.png'):
    """Create distribution plots for each property by catalyst type"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mixing enthalpy distribution
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]['mixing_enthalpy_ev_atom']
        ax1.hist(data, bins=20, alpha=0.6, label=catalyst_type.replace('_', ' '), 
                density=True, edgecolor='black', linewidth=0.8)
        
        # Add KDE
        kde_data = np.linspace(data.min(), data.max(), 100)
        kde = gaussian_kde(data)
        ax1.plot(kde_data, kde(kde_data), linewidth=2)
    
    ax1.set_xlabel('Mixing Enthalpy (eV/atom)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Distribution of Mixing Enthalpy by Catalyst Type', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # d-band center distribution
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]['d_band_center_ev']
        ax2.hist(data, bins=20, alpha=0.6, label=catalyst_type.replace('_', ' '),
                density=True, edgecolor='black', linewidth=0.8)
        
        # Add KDE
        kde_data = np.linspace(data.min(), data.max(), 100)
        kde = gaussian_kde(data)
        ax2.plot(kde_data, kde(kde_data), linewidth=2)
    
    ax2.set_xlabel('d-band Center (eV)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Distribution of d-band Center by Catalyst Type', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_box_plots(df, save_path='catalyst_boxplots.png'):
    """Create box plots comparing properties across catalyst types"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data for box plots
    catalyst_types = ['Known', 'LLM_Generated_HEA', 'LLM_Generated_DA']
    labels = ['Known', 'LLM HEA', 'LLM DA']
    
    # Mixing enthalpy box plot
    data_mixing = [df[df['catalyst_type'] == ct]['mixing_enthalpy_ev_atom'].values 
                   for ct in catalyst_types]
    bp1 = ax1.boxplot(data_mixing, labels=labels, patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=12)
    ax1.set_title('Mixing Enthalpy Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # d-band center box plot
    data_dband = [df[df['catalyst_type'] == ct]['d_band_center_ev'].values 
                  for ct in catalyst_types]
    bp2 = ax2.boxplot(data_dband, labels=labels, patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('d-band Center (eV)', fontsize=12)
    ax2.set_title('d-band Center Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_confidence_ellipses(df, save_path='catalyst_ellipses.png'):
    """Create scatter plot with confidence ellipses for each catalyst type"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'Known': 'blue',
        'LLM_Generated_HEA': 'red',
        'LLM_Generated_DA': 'green'
    }
    
    # Plot each catalyst type with confidence ellipses
    for catalyst_type, color in colors.items():
        data = df[df['catalyst_type'] == catalyst_type]
        x = data['d_band_center_ev'].values
        y = data['mixing_enthalpy_ev_atom'].values
        
        # Plot points
        ax.scatter(x, y, c=color, alpha=0.6, s=50, edgecolors='black', 
                  linewidth=0.5, label=catalyst_type.replace('_', ' '))
        
        # Calculate and plot confidence ellipse (95%)
        if len(x) > 2:
            mean_x, mean_y = np.mean(x), np.mean(y)
            cov = np.cov(x, y)
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            
            # Calculate angle and dimensions
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues) * 2.447  # 95% confidence
            
            # Create ellipse
            ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                            facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
            ax.add_patch(ellipse)
            
            # Add center marker
            ax.plot(mean_x, mean_y, marker='*', markersize=15, color=color,
                   markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('d-band Center (eV)', fontsize=14)
    ax.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=14)
    ax.set_title('Catalyst Types with 95% Confidence Ellipses', fontsize=16, pad=20)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_property_correlation_plot(df, save_path='catalyst_correlation.png'):
    """Create correlation plot with regression lines"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'Known': 'blue',
        'LLM_Generated_HEA': 'red',
        'LLM_Generated_DA': 'green'
    }
    
    # Plot each catalyst type with regression line
    for catalyst_type, color in colors.items():
        data = df[df['catalyst_type'] == catalyst_type]
        x = data['d_band_center_ev'].values
        y = data['mixing_enthalpy_ev_atom'].values
        
        # Scatter plot
        ax.scatter(x, y, c=color, alpha=0.6, s=50, edgecolors='black',
                  linewidth=0.5, label=catalyst_type.replace('_', ' '))
        
        # Fit linear regression
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=2, linestyle='--', alpha=0.8)
            
            # Calculate correlation
            corr = np.corrcoef(x, y)[0, 1]
            ax.text(0.02, 0.98 - 0.05 * list(colors.keys()).index(catalyst_type), 
                   f'{catalyst_type}: r = {corr:.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('d-band Center (eV)', fontsize=14)
    ax.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=14)
    ax.set_title('Property Correlations by Catalyst Type', fontsize=16, pad=20)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics_plot(df, save_path='catalyst_summary_stats.png'):
    """Create a summary statistics visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Count by catalyst type
    catalyst_counts = df['catalyst_type'].value_counts()
    ax1.bar(range(len(catalyst_counts)), catalyst_counts.values, 
            color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_xticks(range(len(catalyst_counts)))
    ax1.set_xticklabels([ct.replace('_', '\n') for ct in catalyst_counts.index], rotation=0)
    ax1.set_ylabel('Count')
    ax1.set_title('Number of Catalysts by Type')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, v in enumerate(catalyst_counts.values):
        ax1.text(i, v + 1, str(v), ha='center', va='bottom')
    
    # Mean property values
    mean_data = df.groupby('catalyst_type')[['mixing_enthalpy_ev_atom', 'd_band_center_ev']].mean()
    x = np.arange(len(mean_data))
    width = 0.35
    
    ax2.bar(x - width/2, mean_data['mixing_enthalpy_ev_atom'], width, 
            label='Mixing Enthalpy', color='steelblue')
    ax2.bar(x + width/2, mean_data['d_band_center_ev'], width,
            label='d-band Center', color='darkorange')
    ax2.set_xticks(x)
    ax2.set_xticklabels([ct.replace('_', '\n') for ct in mean_data.index], rotation=0)
    ax2.set_ylabel('Mean Value (eV)')
    ax2.set_title('Mean Property Values by Catalyst Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Standard deviation comparison
    std_data = df.groupby('catalyst_type')[['mixing_enthalpy_ev_atom', 'd_band_center_ev']].std()
    ax3.bar(x - width/2, std_data['mixing_enthalpy_ev_atom'], width,
            label='Mixing Enthalpy', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, std_data['d_band_center_ev'], width,
            label='d-band Center', color='darkorange', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([ct.replace('_', '\n') for ct in std_data.index], rotation=0)
    ax3.set_ylabel('Standard Deviation (eV)')
    ax3.set_title('Property Variability by Catalyst Type')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Property ranges
    ranges_data = []
    for ct in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == ct]
        ranges_data.append({
            'type': ct,
            'mixing_range': data['mixing_enthalpy_ev_atom'].max() - data['mixing_enthalpy_ev_atom'].min(),
            'd_band_range': data['d_band_center_ev'].max() - data['d_band_center_ev'].min()
        })
    
    ranges_df = pd.DataFrame(ranges_data)
    x = np.arange(len(ranges_df))
    ax4.bar(x - width/2, ranges_df['mixing_range'], width,
            label='Mixing Enthalpy', color='steelblue', alpha=0.7)
    ax4.bar(x + width/2, ranges_df['d_band_range'], width,
            label='d-band Center', color='darkorange', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([ct.replace('_', '\n') for ct in ranges_df['type']], rotation=0)
    ax4.set_ylabel('Range (eV)')
    ax4.set_title('Property Ranges by Catalyst Type')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    # Load data
    df = load_and_analyze_data('fig1_catalyst_data.csv')
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    
    print("1. Creating scatter plot...")
    create_scatter_plot(df)
    
    print("2. Creating density plot...")
    create_density_plot(df)
    
    print("3. Creating distribution plots...")
    create_distribution_plots(df)
    
    print("4. Creating box plots...")
    create_box_plots(df)
    
    print("5. Creating confidence ellipses...")
    create_confidence_ellipses(df)
    
    print("6. Creating correlation plot...")
    create_property_correlation_plot(df)
    
    print("7. Creating summary statistics...")
    create_summary_statistics_plot(df)
    
    print("\nAll visualizations complete!")

if __name__ == "__main__":
    main()