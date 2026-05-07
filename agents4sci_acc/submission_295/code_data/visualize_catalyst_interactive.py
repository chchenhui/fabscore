#!/usr/bin/env python3
"""
Interactive visualization script for catalyst data using Plotly
Creates interactive plots that can be explored in a web browser
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

def create_interactive_scatter(df, save_path='catalyst_interactive.html'):
    """Create interactive scatter plot with hover information"""
    
    # Create figure
    fig = go.Figure()
    
    # Define colors and symbols
    colors = {
        'Known': 'blue',
        'LLM_Generated_HEA': 'red',
        'LLM_Generated_DA': 'green'
    }
    
    symbols = {
        'Known': 'circle',
        'LLM_Generated_HEA': 'square',
        'LLM_Generated_DA': 'diamond'
    }
    
    labels = {
        'Known': 'Known Catalysts',
        'LLM_Generated_HEA': 'LLM-Generated HEAs',
        'LLM_Generated_DA': 'LLM-Generated Doped Alloys'
    }
    
    # Add traces for each catalyst type
    for catalyst_type in df['catalyst_type'].unique():
        data = df[df['catalyst_type'] == catalyst_type]
        
        # Create hover text
        hover_text = []
        for idx, row in data.iterrows():
            text = f"Type: {catalyst_type}<br>"
            text += f"Mixing Enthalpy: {row['mixing_enthalpy_ev_atom']:.3f} eV/atom<br>"
            text += f"d-band Center: {row['d_band_center_ev']:.3f} eV<br>"
            text += f"Index: {idx}"
            hover_text.append(text)
        
        fig.add_trace(go.Scatter(
            x=data['d_band_center_ev'],
            y=data['mixing_enthalpy_ev_atom'],
            mode='markers',
            name=labels[catalyst_type],
            marker=dict(
                color=colors[catalyst_type],
                size=10,
                symbol=symbols[catalyst_type],
                line=dict(width=1, color='black')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Add reference lines
    fig.add_hline(y=-0.5, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="Stability Threshold", annotation_position="right")
    fig.add_vline(x=-2.5, line_dash="dash", line_color="blue", opacity=0.5,
                  annotation_text="Activity Threshold", annotation_position="top")
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Catalyst Discovery Space',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='d-band Center (eV)',
        yaxis_title='Mixing Enthalpy (eV/atom)',
        hovermode='closest',
        width=1000,
        height=700,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add grid
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Save
    fig.write_html(save_path)
    fig.show()
    
    return fig

def create_3d_visualization(df, save_path='catalyst_3d.html'):
    """Create 3D visualization with an additional dimension"""
    
    # Add a third dimension based on distance from origin in property space
    df['distance_from_origin'] = np.sqrt(
        df['mixing_enthalpy_ev_atom']**2 + df['d_band_center_ev']**2
    )
    
    fig = px.scatter_3d(
        df, 
        x='d_band_center_ev', 
        y='mixing_enthalpy_ev_atom', 
        z='distance_from_origin',
        color='catalyst_type',
        symbol='catalyst_type',
        title='3D Catalyst Property Space',
        labels={
            'd_band_center_ev': 'd-band Center (eV)',
            'mixing_enthalpy_ev_atom': 'Mixing Enthalpy (eV/atom)',
            'distance_from_origin': 'Distance from Origin',
            'catalyst_type': 'Catalyst Type'
        },
        color_discrete_map={
            'Known': 'blue',
            'LLM_Generated_HEA': 'red',
            'LLM_Generated_DA': 'green'
        },
        symbol_map={
            'Known': 'circle',
            'LLM_Generated_HEA': 'square',
            'LLM_Generated_DA': 'diamond'
        }
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
    
    fig.update_layout(
        width=1000,
        height=800,
        scene=dict(
            xaxis_title='d-band Center (eV)',
            yaxis_title='Mixing Enthalpy (eV/atom)',
            zaxis_title='Distance from Origin'
        )
    )
    
    fig.write_html(save_path)
    fig.show()
    
    return fig

def create_parallel_coordinates(df, save_path='catalyst_parallel.html'):
    """Create parallel coordinates plot"""
    
    # Normalize the data for better visualization
    df_norm = df.copy()
    df_norm['mixing_enthalpy_norm'] = (df['mixing_enthalpy_ev_atom'] - df['mixing_enthalpy_ev_atom'].min()) / \
                                      (df['mixing_enthalpy_ev_atom'].max() - df['mixing_enthalpy_ev_atom'].min())
    df_norm['d_band_center_norm'] = (df['d_band_center_ev'] - df['d_band_center_ev'].min()) / \
                                    (df['d_band_center_ev'].max() - df['d_band_center_ev'].min())
    
    # Create color map
    color_map = {'Known': 0, 'LLM_Generated_HEA': 1, 'LLM_Generated_DA': 2}
    df_norm['color'] = df_norm['catalyst_type'].map(color_map)
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_norm['color'],
                colorscale=[[0, 'blue'], [0.5, 'red'], [1, 'green']],
                showscale=True,
                colorbar=dict(
                    title='Catalyst Type',
                    tickvals=[0, 1, 2],
                    ticktext=['Known', 'LLM HEA', 'LLM DA']
                )
            ),
            dimensions=[
                dict(range=[df['mixing_enthalpy_ev_atom'].min(), df['mixing_enthalpy_ev_atom'].max()],
                     label='Mixing Enthalpy (eV/atom)',
                     values=df['mixing_enthalpy_ev_atom']),
                dict(range=[df['d_band_center_ev'].min(), df['d_band_center_ev'].max()],
                     label='d-band Center (eV)',
                     values=df['d_band_center_ev']),
                dict(range=[0, 2],
                     label='Catalyst Type',
                     values=df_norm['color'],
                     tickvals=[0, 1, 2],
                     ticktext=['Known', 'LLM HEA', 'LLM DA'])
            ]
        )
    )
    
    fig.update_layout(
        title='Parallel Coordinates: Catalyst Properties',
        width=1000,
        height=600
    )
    
    fig.write_html(save_path)
    fig.show()
    
    return fig

def create_density_heatmap(df, save_path='catalyst_heatmap.html'):
    """Create 2D density heatmap"""
    
    # Create 2D histogram
    fig = go.Figure()
    
    # Add heatmap for known catalysts
    known_data = df[df['catalyst_type'] == 'Known']
    
    fig.add_trace(go.Histogram2d(
        x=known_data['d_band_center_ev'],
        y=known_data['mixing_enthalpy_ev_atom'],
        colorscale='Blues',
        nbinsx=20,
        nbinsy=20,
        showscale=True,
        colorbar=dict(title='Known Catalyst Density', x=1.1)
    ))
    
    # Overlay LLM-generated catalysts
    for catalyst_type, color, symbol in [
        ('LLM_Generated_HEA', 'red', 'square'),
        ('LLM_Generated_DA', 'green', 'diamond')
    ]:
        data = df[df['catalyst_type'] == catalyst_type]
        fig.add_trace(go.Scatter(
            x=data['d_band_center_ev'],
            y=data['mixing_enthalpy_ev_atom'],
            mode='markers',
            name=catalyst_type.replace('_', ' '),
            marker=dict(
                color=color,
                size=10,
                symbol=symbol,
                line=dict(width=1, color='black')
            )
        ))
    
    fig.update_layout(
        title='Catalyst Density Heatmap with LLM-Generated Candidates',
        xaxis_title='d-band Center (eV)',
        yaxis_title='Mixing Enthalpy (eV/atom)',
        width=1000,
        height=700,
        template='plotly_white'
    )
    
    fig.write_html(save_path)
    fig.show()
    
    return fig

def create_violin_plots(df, save_path='catalyst_violins.html'):
    """Create violin plots for property distributions"""
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Mixing Enthalpy Distribution', 
                                      'd-band Center Distribution'))
    
    # Mixing enthalpy violin plots
    for i, catalyst_type in enumerate(df['catalyst_type'].unique()):
        data = df[df['catalyst_type'] == catalyst_type]['mixing_enthalpy_ev_atom']
        fig.add_trace(
            go.Violin(x=[catalyst_type.replace('_', ' ')]*len(data), y=data,
                     name=catalyst_type.replace('_', ' '),
                     box_visible=True,
                     meanline_visible=True,
                     showlegend=False),
            row=1, col=1
        )
    
    # d-band center violin plots
    for i, catalyst_type in enumerate(df['catalyst_type'].unique()):
        data = df[df['catalyst_type'] == catalyst_type]['d_band_center_ev']
        fig.add_trace(
            go.Violin(x=[catalyst_type.replace('_', ' ')]*len(data), y=data,
                     name=catalyst_type.replace('_', ' '),
                     box_visible=True,
                     meanline_visible=True,
                     showlegend=False),
            row=1, col=2
        )
    
    fig.update_layout(
        title='Property Distributions by Catalyst Type',
        height=600,
        width=1200,
        showlegend=False
    )
    
    fig.update_yaxes(title_text='Mixing Enthalpy (eV/atom)', row=1, col=1)
    fig.update_yaxes(title_text='d-band Center (eV)', row=1, col=2)
    
    fig.write_html(save_path)
    fig.show()
    
    return fig

def create_animated_scatter(df, save_path='catalyst_animated.html'):
    """Create animated scatter plot showing progression"""
    
    # Add a synthetic 'generation' column for animation
    df['generation'] = 0
    df.loc[df['catalyst_type'] == 'LLM_Generated_HEA', 'generation'] = 1
    df.loc[df['catalyst_type'] == 'LLM_Generated_DA', 'generation'] = 2
    
    # Create animated figure
    fig = px.scatter(
        df, 
        x='d_band_center_ev', 
        y='mixing_enthalpy_ev_atom',
        color='catalyst_type',
        symbol='catalyst_type',
        animation_frame='generation',
        title='Evolution of Catalyst Discovery',
        labels={
            'd_band_center_ev': 'd-band Center (eV)',
            'mixing_enthalpy_ev_atom': 'Mixing Enthalpy (eV/atom)',
            'catalyst_type': 'Catalyst Type',
            'generation': 'Generation'
        },
        color_discrete_map={
            'Known': 'blue',
            'LLM_Generated_HEA': 'red',
            'LLM_Generated_DA': 'green'
        },
        range_x=[df['d_band_center_ev'].min()-0.5, df['d_band_center_ev'].max()+0.5],
        range_y=[df['mixing_enthalpy_ev_atom'].min()-0.1, df['mixing_enthalpy_ev_atom'].max()+0.1]
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
    
    fig.update_layout(
        width=1000,
        height=700,
        template='plotly_white'
    )
    
    # Update animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
    
    fig.write_html(save_path)
    fig.show()
    
    return fig

def create_dashboard(df, save_path='catalyst_dashboard.html'):
    """Create a comprehensive dashboard with multiple visualizations"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Catalyst Distribution', 'Property Correlations',
                       'Statistical Summary', 'Density Contours'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'bar'}, {'type': 'scatter'}]],
        row_heights=[0.5, 0.5]
    )
    
    # 1. Main scatter plot
    colors = {'Known': 'blue', 'LLM_Generated_HEA': 'red', 'LLM_Generated_DA': 'green'}
    for catalyst_type, color in colors.items():
        data = df[df['catalyst_type'] == catalyst_type]
        fig.add_trace(
            go.Scatter(
                x=data['d_band_center_ev'],
                y=data['mixing_enthalpy_ev_atom'],
                mode='markers',
                name=catalyst_type.replace('_', ' '),
                marker=dict(color=color, size=8),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # 2. Correlation plot with trend lines
    for catalyst_type, color in colors.items():
        data = df[df['catalyst_type'] == catalyst_type]
        x = data['d_band_center_ev'].values
        y = data['mixing_enthalpy_ev_atom'].values
        
        # Add scatter
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color=color, size=6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            x_trend = np.linspace(x.min(), x.max(), 100)
            y_trend = z[0] * x_trend + z[1]
            fig.add_trace(
                go.Scatter(
                    x=x_trend, y=y_trend,
                    mode='lines',
                    line=dict(color=color, dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # 3. Count statistics
    counts = df['catalyst_type'].value_counts()
    fig.add_trace(
        go.Bar(
            x=[ct.replace('_', ' ') for ct in counts.index],
            y=counts.values,
            marker_color=['blue', 'red', 'green'],
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Density contours for known catalysts
    known_data = df[df['catalyst_type'] == 'Known']
    fig.add_trace(
        go.Histogram2dContour(
            x=known_data['d_band_center_ev'],
            y=known_data['mixing_enthalpy_ev_atom'],
            colorscale='Blues',
            showscale=False
        ),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='d-band Center (eV)', row=1, col=1)
    fig.update_yaxes(title_text='Mixing Enthalpy (eV/atom)', row=1, col=1)
    fig.update_xaxes(title_text='d-band Center (eV)', row=1, col=2)
    fig.update_yaxes(title_text='Mixing Enthalpy (eV/atom)', row=1, col=2)
    fig.update_xaxes(title_text='Catalyst Type', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    fig.update_xaxes(title_text='d-band Center (eV)', row=2, col=2)
    fig.update_yaxes(title_text='Mixing Enthalpy (eV/atom)', row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Catalyst Discovery Dashboard',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1000,
        width=1400,
        showlegend=True,
        template='plotly_white'
    )
    
    fig.write_html(save_path)
    fig.show()
    
    return fig

def main():
    """Main execution function"""
    # Load data
    df = pd.read_csv('fig1_catalyst_data.csv')
    
    print("Creating interactive visualizations...")
    
    # Create all interactive plots
    print("1. Creating interactive scatter plot...")
    create_interactive_scatter(df)
    
    print("2. Creating 3D visualization...")
    create_3d_visualization(df)
    
    print("3. Creating parallel coordinates plot...")
    create_parallel_coordinates(df)
    
    print("4. Creating density heatmap...")
    create_density_heatmap(df)
    
    print("5. Creating violin plots...")
    create_violin_plots(df)
    
    print("6. Creating animated scatter plot...")
    create_animated_scatter(df)
    
    print("7. Creating comprehensive dashboard...")
    create_dashboard(df)
    
    print("\nAll interactive visualizations complete!")
    print("Open the HTML files in your web browser to explore the interactive plots.")

if __name__ == "__main__":
    main()