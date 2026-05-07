#!/usr/bin/env python3
"""
Create comprehensive visualizations for clustering results from multi-agent surveys
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_cluster_data(directory):
    """Load all cluster JSON files from the directory"""
    cluster_files = {
        'In-Context Learning': 'icl_clusters.json',
        'RLHF Alignment': 'rlhf_alignment_clusters.json',
        'Multimodal RL': 'multimodal_rl_clusters.json',
        'Synthetic Data': 'synthetic_data_clusters.json'
    }
    
    data = {}
    for topic, filename in cluster_files.items():
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                topic_data = json.load(f)
                data[topic] = topic_data
                
                # Handle different data structures
                if 'metadata' in topic_data:
                    if 'num_papers' in topic_data['metadata']:
                        num_papers = topic_data['metadata']['num_papers']
                    else:
                        num_papers = len(topic_data.get('papers', []))
                    num_clusters = topic_data['metadata'].get('num_clusters', len(topic_data.get('clusters', [])))
                else:
                    num_papers = len(topic_data.get('papers', []))
                    num_clusters = len(topic_data.get('clusters', []))
                    
                print(f"Loaded {topic}: {num_papers} papers, {num_clusters} clusters")
    
    return data

def create_cluster_size_heatmap(data, output_dir):
    """Create a heatmap showing cluster sizes across different topics"""
    
    # Prepare data for heatmap
    topics = list(data.keys())
    max_clusters = max(len(d['clusters']) for d in data.values())
    
    # Create matrix for heatmap
    matrix = np.zeros((len(topics), max_clusters))
    
    for i, topic in enumerate(topics):
        clusters = data[topic]['clusters']
        for j, cluster in enumerate(clusters[:max_clusters]):
            matrix[i, j] = cluster['size']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create heatmap
    sns.heatmap(matrix, 
                annot=True, 
                fmt='.0f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Papers'},
                xticklabels=[f'Cluster {i+1}' for i in range(max_clusters)],
                yticklabels=topics,
                ax=ax,
                vmin=0,
                linewidths=0.5,
                linecolor='gray')
    
    ax.set_title('Cluster Size Distribution Across Survey Topics', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Survey Topic', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_size_heatmap.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'cluster_size_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved cluster size heatmap")

def create_keyword_frequency_plot(data, output_dir):
    """Create a bar plot showing top keywords across all clusters"""
    
    # Extract keywords from all clusters
    all_keywords = Counter()
    topic_keywords = defaultdict(Counter)
    
    for topic, topic_data in data.items():
        for cluster in topic_data['clusters']:
            # Extract words from cluster name
            words = re.findall(r'\b[a-z]+\b', cluster['name'].lower())
            for word in words:
                if len(word) > 3 and word not in ['this', 'with', 'from', 'that', 'have', 'most']:
                    all_keywords[word] += cluster['size']
                    topic_keywords[topic][word] += cluster['size']
    
    # Get top 20 keywords
    top_keywords = dict(all_keywords.most_common(20))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Overall keyword frequency
    keywords = list(top_keywords.keys())
    values = list(top_keywords.values())
    
    bars = ax1.bar(keywords, values, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Keywords', fontsize=12)
    ax1.set_ylabel('Weighted Frequency (by cluster size)', fontsize=12)
    ax1.set_title('Top 20 Keywords Across All Survey Clusters', fontsize=14, fontweight='bold')
    ax1.set_xticklabels(keywords, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)
    
    # Stacked bar chart for keywords by topic
    topics = list(data.keys())
    keyword_matrix = np.zeros((len(topics), len(keywords)))
    
    for i, topic in enumerate(topics):
        for j, keyword in enumerate(keywords):
            keyword_matrix[i, j] = topic_keywords[topic].get(keyword, 0)
    
    # Create stacked bars
    bottom = np.zeros(len(keywords))
    colors = plt.cm.Set3(np.linspace(0, 1, len(topics)))
    
    for i, topic in enumerate(topics):
        ax2.bar(keywords, keyword_matrix[i], bottom=bottom, 
                label=topic, color=colors[i], alpha=0.8)
        bottom += keyword_matrix[i]
    
    ax2.set_xlabel('Keywords', fontsize=12)
    ax2.set_ylabel('Weighted Frequency (by cluster size)', fontsize=12)
    ax2.set_title('Keyword Distribution by Survey Topic', fontsize=14, fontweight='bold')
    ax2.set_xticklabels(keywords, rotation=45, ha='right')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'keyword_frequency.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'keyword_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved keyword frequency plots")

def create_cluster_statistics_plot(data, output_dir):
    """Create comprehensive statistics plots for clustering results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Papers per topic
    ax = axes[0, 0]
    topics = list(data.keys())
    paper_counts = []
    cluster_counts = []
    
    for d in data.values():
        if 'metadata' in d and 'num_papers' in d['metadata']:
            paper_counts.append(d['metadata']['num_papers'])
        else:
            paper_counts.append(len(d.get('papers', [])))
            
        if 'metadata' in d and 'num_clusters' in d['metadata']:
            cluster_counts.append(d['metadata']['num_clusters'])
        else:
            cluster_counts.append(len(d.get('clusters', [])))
    
    x = np.arange(len(topics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, paper_counts, width, label='Total Papers', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, [c*100 for c in cluster_counts], width, label='Clusters (×100)', color='coral', edgecolor='black')
    
    ax.set_xlabel('Survey Topic', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Papers and Clusters by Topic', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 2. Average cluster size
    ax = axes[0, 1]
    avg_sizes = []
    for topic in topics:
        clusters = data[topic]['clusters']
        avg_size = np.mean([c['size'] for c in clusters])
        avg_sizes.append(avg_size)
    
    bars = ax.bar(topics, avg_sizes, color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Survey Topic', fontsize=11)
    ax.set_ylabel('Average Papers per Cluster', fontsize=11)
    ax.set_title('Average Cluster Size by Topic', fontsize=12, fontweight='bold')
    ax.set_xticklabels(topics, rotation=45, ha='right')
    
    for bar, val in zip(bars, avg_sizes):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Cluster size distribution (violin plot)
    ax = axes[1, 0]
    size_data = []
    labels = []
    for topic in topics:
        sizes = [c['size'] for c in data[topic]['clusters']]
        size_data.extend(sizes)
        labels.extend([topic] * len(sizes))
    
    df = pd.DataFrame({'Topic': labels, 'Cluster Size': size_data})
    sns.violinplot(data=df, x='Topic', y='Cluster Size', ax=ax, inner='box')
    ax.set_xlabel('Survey Topic', fontsize=11)
    ax.set_ylabel('Cluster Size (papers)', fontsize=11)
    ax.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Silhouette scores comparison
    ax = axes[1, 1]
    silhouette_scores = []
    for topic in topics:
        topic_data = data[topic]
        if 'metadata' in topic_data and 'optimization_metrics' in topic_data['metadata']:
            metrics = topic_data['metadata']['optimization_metrics']
            # Get the score for the selected number of clusters
            if 'num_clusters' in topic_data['metadata']:
                n_clusters = str(topic_data['metadata']['num_clusters'])
            else:
                n_clusters = str(len(topic_data.get('clusters', [])))
                
            if n_clusters in metrics:
                silhouette_scores.append(metrics[n_clusters]['silhouette'])
            else:
                # Use the first available score
                first_key = list(metrics.keys())[0]
                silhouette_scores.append(metrics[first_key]['silhouette'])
        else:
            silhouette_scores.append(0)
    
    bars = ax.bar(topics, silhouette_scores, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Survey Topic', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Clustering Quality (Silhouette Score)', fontsize=12, fontweight='bold')
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Threshold')
    
    for bar, val in zip(bars, silhouette_scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.legend()
    
    plt.suptitle('Clustering Statistics Across Survey Topics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_statistics.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'clustering_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved clustering statistics plots")

def create_temporal_distribution_plot(data, output_dir):
    """Create a plot showing temporal distribution of papers in clusters"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (topic, topic_data) in enumerate(data.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # Extract temporal data
        cluster_names = []
        cluster_years = []
        
        for cluster in topic_data['clusters'][:8]:  # Limit to 8 clusters for visibility
            if 'statistics' in cluster and 'avg_year' in cluster['statistics']:
                cluster_names.append(cluster['name'][:20] + '...' if len(cluster['name']) > 20 else cluster['name'])
                cluster_years.append(cluster['statistics']['avg_year'])
        
        if cluster_years:
            # Create bar plot
            bars = ax.barh(range(len(cluster_names)), cluster_years, color='teal', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(cluster_names)))
            ax.set_yticklabels(cluster_names, fontsize=9)
            ax.set_xlabel('Average Publication Year', fontsize=10)
            ax.set_title(f'{topic}', fontsize=11, fontweight='bold')
            ax.set_xlim(2018, 2025)
            
            # Add value labels
            for bar, year in zip(bars, cluster_years):
                ax.text(year, bar.get_y() + bar.get_height()/2.,
                       f'{year:.1f}', ha='left', va='center', fontsize=8)
    
    plt.suptitle('Temporal Distribution of Papers Across Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'temporal_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved temporal distribution plot")

def main():
    # Setup directories
    input_dir = '/path/to/project/output-multi-agent-v2'
    output_dir = '/path/to/project/LLM-Surveying-LLMs-An-Agentic-Pipeline-for-Autonomous-Scientific-Paper-Surveying/paper/figures'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading clustering data...")
    data = load_cluster_data(input_dir)
    
    if not data:
        print("No clustering data found!")
        return
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_cluster_size_heatmap(data, output_dir)
    create_keyword_frequency_plot(data, output_dir)
    create_cluster_statistics_plot(data, output_dir)
    create_temporal_distribution_plot(data, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")
    print("\nGenerated files:")
    print("  - cluster_size_heatmap.pdf/png")
    print("  - keyword_frequency.pdf/png") 
    print("  - clustering_statistics.pdf/png")
    print("  - temporal_distribution.pdf/png")

if __name__ == "__main__":
    main()