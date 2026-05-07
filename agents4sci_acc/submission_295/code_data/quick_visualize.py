#!/usr/bin/env python3
"""
Quick visualization script for catalyst data
Creates the most important plots with minimal dependencies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    # Load data
    df = pd.read_csv('fig1_catalyst_data.csv')
    
    # Print summary
    print("Catalyst Data Summary:")
    print(f"Total catalysts: {len(df)}")
    print("\nCatalyst types:")
    print(df['catalyst_type'].value_counts())
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Main scatter plot
    ax1 = plt.subplot(2, 2, 1)
    colors = {'Known': '#1f77b4', 'LLM_Generated_HEA': '#ff7f0e', 'LLM_Generated_DA': '#2ca02c'}
    markers = {'Known': 'o', 'LLM_Generated_HEA': 's', 'LLM_Generated_DA': '^'}
    
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]
        ax1.scatter(data['d_band_center_ev'], data['mixing_enthalpy_ev_atom'],
                   c=colors[catalyst_type], marker=markers[catalyst_type],
                   s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=catalyst_type.replace('_', ' '))
    
    ax1.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=-2.5, color='blue', linestyle='--', alpha=0.5)
    ax1.set_xlabel('d-band Center (eV)', fontsize=12)
    ax1.set_ylabel('Mixing Enthalpy (eV/atom)', fontsize=12)
    ax1.set_title('Catalyst Property Space', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Mixing enthalpy distribution
    ax2 = plt.subplot(2, 2, 2)
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]['mixing_enthalpy_ev_atom']
        ax2.hist(data, bins=15, alpha=0.6, label=catalyst_type.replace('_', ' '),
                density=True, edgecolor='black', linewidth=0.8)
    
    ax2.set_xlabel('Mixing Enthalpy (eV/atom)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Mixing Enthalpy Distribution', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. d-band center distribution
    ax3 = plt.subplot(2, 2, 3)
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]['d_band_center_ev']
        ax3.hist(data, bins=15, alpha=0.6, label=catalyst_type.replace('_', ' '),
                density=True, edgecolor='black', linewidth=0.8)
    
    ax3.set_xlabel('d-band Center (eV)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('d-band Center Distribution', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate statistics
    stats_text = "Summary Statistics:\n\n"
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]
        stats_text += f"{catalyst_type}:\n"
        stats_text += f"  Count: {len(data)}\n"
        stats_text += f"  Mixing Enthalpy: {data['mixing_enthalpy_ev_atom'].mean():.3f} ± {data['mixing_enthalpy_ev_atom'].std():.3f} eV/atom\n"
        stats_text += f"  d-band Center: {data['d_band_center_ev'].mean():.3f} ± {data['d_band_center_ev'].std():.3f} eV\n\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Catalyst Data Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig('catalyst_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second figure for box plots
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot for mixing enthalpy
    catalyst_types = ['Known', 'LLM_Generated_HEA', 'LLM_Generated_DA']
    labels = ['Known', 'LLM HEA', 'LLM DA']
    data_mixing = [df[df['catalyst_type'] == ct]['mixing_enthalpy_ev_atom'].values for ct in catalyst_types]
    
    bp1 = ax5.boxplot(data_mixing, labels=labels, patch_artist=True, showmeans=True)
    colors_box = ['lightblue', 'lightcoral', 'lightgreen']
    for patch, color in zip(bp1['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax5.set_ylabel('Mixing Enthalpy (eV/atom)')
    ax5.set_title('Mixing Enthalpy by Catalyst Type')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Box plot for d-band center
    data_dband = [df[df['catalyst_type'] == ct]['d_band_center_ev'].values for ct in catalyst_types]
    bp2 = ax6.boxplot(data_dband, labels=labels, patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp2['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax6.set_ylabel('d-band Center (eV)')
    ax6.set_title('d-band Center by Catalyst Type')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('catalyst_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()