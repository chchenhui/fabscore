#!/usr/bin/env python3
"""
Visualization script for candidate selection data
Creates volcano plots and other visualizations showing the computational validation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

def load_and_analyze_data(filepath):
    """Load candidate selection data and perform analysis"""
    df = pd.read_csv(filepath)
    
    print("Candidate Selection Data Summary:")
    print(f"Total candidates: {len(df)}")
    print(f"Catalyst type: {df['catalyst_type'].unique()[0]}")
    print("\nProperty ranges:")
    print(f"Mixing enthalpy: [{df['mixing_enthalpy_ev_atom'].min():.3f}, {df['mixing_enthalpy_ev_atom'].max():.3f}] eV/atom")
    print(f"d-band center: [{df['d_band_center_ev'].min():.3f}, {df['d_band_center_ev'].max():.3f}] eV")
    print(f"ΔE_NOH: [{df['delta_e_noh_ev'].min():.3f}, {df['delta_e_noh_ev'].max():.3f}] eV")
    print(f"Limiting potential: [{df['limiting_potential_v'].min():.3f}, {df['limiting_potential_v'].max():.3f}] V")
    
    return df

def create_volcano_plot(df, save_path='figures/volcano_plot.png'):
    """Create volcano plot showing activity vs adsorption energy"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(df['delta_e_noh_ev'], 
                        df['limiting_potential_v'],
                        c=df['mixing_enthalpy_ev_atom'],
                        cmap='viridis',
                        s=100,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mixing Enthalpy (eV/atom)', fontsize=12)
    
    # Fit volcano curve (quadratic)
    x_sort = np.sort(df['delta_e_noh_ev'])
    
    # Fit a quadratic function to approximate volcano shape
    coeffs = np.polyfit(df['delta_e_noh_ev'], df['limiting_potential_v'], 2)
    poly = np.poly1d(coeffs)
    y_fit = poly(x_sort)
    
    # Plot fitted curve
    ax.plot(x_sort, y_fit, 'r--', linewidth=2, alpha=0.8, label='Quadratic fit')
    
    # Find and mark the peak
    peak_x = -coeffs[1] / (2 * coeffs[0])
    peak_y = poly(peak_x)
    ax.plot(peak_x, peak_y, 'r*', markersize=20, label=f'Peak: ΔE = {peak_x:.2f} eV')
    
    # Add optimal region shading
    ax.axvspan(peak_x - 0.2, peak_x + 0.2, alpha=0.2, color='red', label='Optimal region')
    
    # Labels and title
    ax.set_xlabel('ΔE_NOH (eV)', fontsize=14)
    ax.set_ylabel('Limiting Potential (V)', fontsize=14)
    ax.set_title('Volcano Plot: Activity vs NOH Adsorption Energy', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add annotations for best catalysts
    top_5 = df.nlargest(5, 'limiting_potential_v')
    for idx, row in top_5.iterrows():
        ax.annotate(f'{idx}', 
                   (row['delta_e_noh_ev'], row['limiting_potential_v']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_stability_activity_plot(df, save_path='figures/stability_activity.png'):
    """Create plot showing relationship between stability and activity"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(df['mixing_enthalpy_ev_atom'], 
                        df['limiting_potential_v'],
                        c=df['delta_e_noh_ev'],
                        cmap='coolwarm',
                        s=100,
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ΔE_NOH (eV)', fontsize=12)
    
    # Add trend line
    z = np.polyfit(df['mixing_enthalpy_ev_atom'], df['limiting_potential_v'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['mixing_enthalpy_ev_atom'].min(), 
                         df['mixing_enthalpy_ev_atom'].max(), 100)
    ax.plot(x_trend, p(x_trend), 'k--', linewidth=2, alpha=0.8)
    
    # Calculate correlation
    corr = np.corrcoef(df['mixing_enthalpy_ev_atom'], df['limiting_potential_v'])[0, 1]
    ax.text(0.05, 0.95, f'Correlation: r = {corr:.3f}', 
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add threshold lines
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=-0.7, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Mixing Enthalpy (eV/atom)', fontsize=14)
    ax.set_ylabel('Limiting Potential (V)', fontsize=14)
    ax.set_title('Stability vs Activity Trade-off', fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add quadrant labels
    ax.text(-0.95, 0.8, 'Stable &\nActive', fontsize=10, ha='center', 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.text(-0.55, 0.8, 'Less Stable &\nActive', fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_property_correlations(df, save_path='figures/property_correlations.png'):
    """Create correlation matrix of all properties"""
    # Create figure with correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    properties = ['mixing_enthalpy_ev_atom', 'd_band_center_ev', 
                 'delta_e_noh_ev', 'limiting_potential_v']
    labels = ['Mixing Enthalpy\n(eV/atom)', 'd-band Center\n(eV)', 
              'ΔE_NOH (eV)', 'Limiting Potential\n(V)']
    
    # Calculate correlation matrix
    corr_matrix = df[properties].corr()
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(properties)))
    ax.set_yticks(np.arange(len(properties)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add correlation values
    for i in range(len(properties)):
        for j in range(len(properties)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', 
                          color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white',
                          fontsize=12, weight='bold')
    
    ax.set_title('Property Correlation Matrix for HEA Catalysts', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_3d_surface_plot(df, save_path='figures/3d_activity_surface.png'):
    """Create 3D surface plot of activity landscape"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create grid for surface
    x = df['delta_e_noh_ev']
    y = df['mixing_enthalpy_ev_atom']
    z = df['limiting_potential_v']
    
    # Create mesh grid
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate Z values
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    
    # Create surface plot
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.7, 
                          edgecolor='none', antialiased=True)
    
    # Add scatter points
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, 
                        edgecolors='black', linewidth=1, alpha=0.9)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, pad=0.1, label='Limiting Potential (V)')
    
    # Labels
    ax.set_xlabel('ΔE_NOH (eV)', fontsize=12, labelpad=10)
    ax.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=12, labelpad=10)
    ax.set_zlabel('Limiting Potential (V)', fontsize=12, labelpad=10)
    ax.set_title('3D Activity Landscape of HEA Catalysts', fontsize=14, pad=20)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_ranking(df, save_path='figures/performance_ranking.png'):
    """Create performance ranking visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sort by limiting potential
    df_sorted = df.sort_values('limiting_potential_v', ascending=False).reset_index(drop=True)
    
    # Top 10 catalysts bar chart
    top_10 = df_sorted.head(10)
    
    ax1.barh(range(10), top_10['limiting_potential_v'], 
             color=plt.cm.viridis(top_10['mixing_enthalpy_ev_atom'] / top_10['mixing_enthalpy_ev_atom'].min()))
    ax1.set_yticks(range(10))
    ax1.set_yticklabels([f'Catalyst {i+1}' for i in range(10)])
    ax1.set_xlabel('Limiting Potential (V)', fontsize=12)
    ax1.set_title('Top 10 Catalysts by Activity', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add stability info
    for i, (idx, row) in enumerate(top_10.iterrows()):
        ax1.text(row['limiting_potential_v'] + 0.02, i, 
                f'ΔH={row["mixing_enthalpy_ev_atom"]:.2f}',
                va='center', fontsize=8)
    
    # Activity vs Stability scatter with size based on d-band center
    scatter = ax2.scatter(df['mixing_enthalpy_ev_atom'], 
                         df['limiting_potential_v'],
                         s=100 * (df['d_band_center_ev'] - df['d_band_center_ev'].min()) / 
                           (df['d_band_center_ev'].max() - df['d_band_center_ev'].min()) + 50,
                         c=df['delta_e_noh_ev'],
                         cmap='coolwarm',
                         alpha=0.7,
                         edgecolors='black',
                         linewidth=1)
    
    # Highlight top 10
    ax2.scatter(top_10['mixing_enthalpy_ev_atom'], 
               top_10['limiting_potential_v'],
               s=200, facecolors='none', edgecolors='red', linewidths=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('ΔE_NOH (eV)', fontsize=10)
    
    ax2.set_xlabel('Mixing Enthalpy (eV/atom)', fontsize=12)
    ax2.set_ylabel('Limiting Potential (V)', fontsize=12)
    ax2.set_title('All Catalysts (Top 10 highlighted)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add note about size
    ax2.text(0.02, 0.02, 'Size ∝ d-band center', transform=ax2.transAxes,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_optimization_path_plot(df, save_path='figures/optimization_path.png'):
    """Create plot showing potential optimization paths"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot of activity
    x = df['delta_e_noh_ev']
    y = df['mixing_enthalpy_ev_atom']
    z = df['limiting_potential_v']
    
    # Create grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    
    # Contour plot
    contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap='viridis', alpha=0.7)
    contour_lines = ax.contour(Xi, Yi, Zi, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.2f')
    
    # Add scatter points
    scatter = ax.scatter(x, y, c=z, cmap='viridis', s=100, 
                        edgecolors='black', linewidth=1, zorder=10)
    
    # Find and mark best catalyst
    best_idx = df['limiting_potential_v'].idxmax()
    best = df.loc[best_idx]
    ax.plot(best['delta_e_noh_ev'], best['mixing_enthalpy_ev_atom'], 
           'r*', markersize=20, zorder=20, label='Best catalyst')
    
    # Draw optimization arrows from nearby points to best
    nearby = df.loc[(df.index != best_idx) & 
                   (np.sqrt((df['delta_e_noh_ev'] - best['delta_e_noh_ev'])**2 + 
                           (df['mixing_enthalpy_ev_atom'] - best['mixing_enthalpy_ev_atom'])**2) < 0.3)]
    
    for idx, row in nearby.iterrows():
        ax.annotate('', xy=(best['delta_e_noh_ev'], best['mixing_enthalpy_ev_atom']),
                   xytext=(row['delta_e_noh_ev'], row['mixing_enthalpy_ev_atom']),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.5, lw=1.5))
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Limiting Potential (V)', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('ΔE_NOH (eV)', fontsize=14)
    ax.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=14)
    ax.set_title('Activity Landscape and Optimization Paths', fontsize=16, pad=20)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(df, save_path='figures/summary_statistics.png'):
    """Create summary statistics visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distribution plots
    properties = ['mixing_enthalpy_ev_atom', 'd_band_center_ev', 
                 'delta_e_noh_ev', 'limiting_potential_v']
    labels = ['Mixing Enthalpy (eV/atom)', 'd-band Center (eV)', 
              'ΔE_NOH (eV)', 'Limiting Potential (V)']
    
    for ax, prop, label in zip([ax1, ax2, ax3, ax4], properties, labels):
        ax.hist(df[prop], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(df[prop].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[prop].mean():.3f}')
        ax.axvline(df[prop].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[prop].median():.3f}')
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Property Distributions for HEA Catalysts', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    # Load data
    df = load_and_analyze_data('candidate_selection_data.csv')
    
    print("\nGenerating visualizations...")
    
    print("1. Creating volcano plot...")
    create_volcano_plot(df)
    
    print("2. Creating stability-activity plot...")
    create_stability_activity_plot(df)
    
    print("3. Creating property correlations...")
    create_property_correlations(df)
    
    print("4. Creating 3D surface plot...")
    create_3d_surface_plot(df)
    
    print("5. Creating performance ranking...")
    create_performance_ranking(df)
    
    print("6. Creating optimization path plot...")
    create_optimization_path_plot(df)
    
    print("7. Creating summary statistics...")
    create_summary_report(df)
    
    print("\nAll visualizations saved to figures/ folder!")
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS:")
    print("="*50)
    
    best_catalyst = df.loc[df['limiting_potential_v'].idxmax()]
    print(f"\nBest catalyst (highest activity):")
    print(f"  Limiting potential: {best_catalyst['limiting_potential_v']:.3f} V")
    print(f"  ΔE_NOH: {best_catalyst['delta_e_noh_ev']:.3f} eV")
    print(f"  Mixing enthalpy: {best_catalyst['mixing_enthalpy_ev_atom']:.3f} eV/atom")
    
    most_stable = df.loc[df['mixing_enthalpy_ev_atom'].idxmin()]
    print(f"\nMost stable catalyst:")
    print(f"  Mixing enthalpy: {most_stable['mixing_enthalpy_ev_atom']:.3f} eV/atom")
    print(f"  Limiting potential: {most_stable['limiting_potential_v']:.3f} V")
    
    # Find optimal trade-off (high activity and good stability)
    df['score'] = df['limiting_potential_v'] - 0.3 * df['mixing_enthalpy_ev_atom']
    optimal = df.loc[df['score'].idxmax()]
    print(f"\nOptimal trade-off catalyst:")
    print(f"  Limiting potential: {optimal['limiting_potential_v']:.3f} V")
    print(f"  Mixing enthalpy: {optimal['mixing_enthalpy_ev_atom']:.3f} eV/atom")

if __name__ == "__main__":
    main()