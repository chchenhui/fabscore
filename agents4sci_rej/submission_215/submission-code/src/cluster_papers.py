#!/usr/bin/env python3
"""
Cluster LLM RLHF alignment papers into meaningful research themes.
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Visualization imports removed - not needed for clustering
from collections import Counter, defaultdict
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_papers(filepath):
    """Load papers from JSON file."""
    print(f"Loading papers from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['papers']

def prepare_text_for_embedding(paper):
    """Prepare paper text for embedding generation."""
    # Combine title and abstract for richer representation
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    
    # Add year context if available
    year = paper.get('publication_year', '')
    year_str = f" ({year})" if year else ""
    
    # Combine with weight on title
    text = f"{title}{year_str}. {title}. {abstract}"
    
    # Truncate if too long (model has max length)
    max_length = 512
    if len(text) > max_length:
        text = text[:max_length]
    
    return text

def generate_embeddings(papers, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for papers using sentence-transformers."""
    print(f"\nGenerating embeddings using {model_name}...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    
    # Prepare texts
    texts = [prepare_text_for_embedding(paper) for paper in papers]
    
    # Generate embeddings with progress indicator
    print(f"Embedding {len(texts)} papers...")
    embeddings = model.encode(texts, 
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            batch_size=32 if device == 'cuda' else 16)
    
    return embeddings

def find_optimal_clusters(embeddings, min_clusters=5, max_clusters=15):
    """Find optimal number of clusters using multiple metrics."""
    print(f"\nFinding optimal number of clusters ({min_clusters}-{max_clusters})...")
    
    scores = {
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    K = range(min_clusters, max_clusters + 1)
    
    for k in K:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        scores['silhouette'].append(silhouette_score(embeddings, labels))
        scores['davies_bouldin'].append(davies_bouldin_score(embeddings, labels))
        scores['calinski_harabasz'].append(calinski_harabasz_score(embeddings, labels))
    
    # Find optimal k (maximize silhouette and calinski, minimize davies-bouldin)
    silhouette_optimal = min_clusters + np.argmax(scores['silhouette'])
    davies_optimal = min_clusters + np.argmin(scores['davies_bouldin'])
    calinski_optimal = min_clusters + np.argmax(scores['calinski_harabasz'])
    
    # Use majority vote or silhouette if no consensus
    optimal_k = max(set([silhouette_optimal, davies_optimal, calinski_optimal]), 
                   key=[silhouette_optimal, davies_optimal, calinski_optimal].count)
    
    print(f"\nOptimal clusters by metric:")
    print(f"  Silhouette: {silhouette_optimal}")
    print(f"  Davies-Bouldin: {davies_optimal}")
    print(f"  Calinski-Harabasz: {calinski_optimal}")
    print(f"  Selected: {optimal_k}")
    
    return optimal_k, scores

def extract_key_terms(papers, cluster_labels, n_terms=5):
    """Extract key terms for each cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    cluster_texts = defaultdict(list)
    for paper, label in zip(papers, cluster_labels):
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        cluster_texts[label].append(text)
    
    key_terms = {}
    for cluster_id, texts in cluster_texts.items():
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', 
                                    ngram_range=(1, 2))
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms based on mean TF-IDF
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
            key_terms[cluster_id] = [feature_names[i] for i in top_indices]
        except:
            key_terms[cluster_id] = []
    
    return key_terms

def name_clusters(papers, cluster_labels, key_terms):
    """Generate meaningful names for clusters based on content analysis."""
    cluster_names = {}
    cluster_papers = defaultdict(list)
    
    for paper, label in zip(papers, cluster_labels):
        cluster_papers[label].append(paper)
    
    # Analyze each cluster to generate names
    for cluster_id, cluster_paper_list in cluster_papers.items():
        # Analyze titles for common patterns
        titles = [p.get('title', '').lower() for p in cluster_paper_list]
        
        # Common RLHF-related patterns
        patterns = {
            'constitutional': 'Constitutional AI and Self-Supervision',
            'dpo': 'Direct Preference Optimization',
            'preference optimization': 'Preference Optimization Methods',
            'reward model': 'Reward Modeling and Design',
            'safety': 'Safety and Alignment Techniques',
            'evaluation': 'Evaluation and Benchmarking',
            'human feedback': 'Human Feedback Integration',
            'ppo': 'PPO and Policy Optimization',
            'instruction': 'Instruction Following and Tuning',
            'harmless': 'Harmlessness and Safety',
            'helpful': 'Helpfulness Optimization',
            'fine-tun': 'Fine-tuning Strategies',
            'dataset': 'Datasets and Data Collection',
            'benchmark': 'Benchmarks and Evaluation',
            'llama': 'Open Model Alignment',
            'gpt': 'GPT Model Alignment',
            'claude': 'Claude and Anthropic Methods',
            'chat': 'Conversational AI Alignment',
            'reasoning': 'Reasoning and Chain-of-Thought',
            'multimodal': 'Multimodal Alignment',
            'efficient': 'Efficient Training Methods',
            'scalab': 'Scalable Alignment Approaches',
            'theoret': 'Theoretical Foundations',
            'survey': 'Surveys and Reviews',
            'critic': 'Self-Critique and Reflection',
            'debate': 'AI Debate and Argumentation',
            'jailbreak': 'Adversarial Robustness',
            'attack': 'Attack and Defense Methods',
            'toxic': 'Toxicity and Bias Mitigation',
            'bias': 'Bias Detection and Mitigation'
        }
        
        # Count pattern matches
        pattern_counts = Counter()
        for pattern, name in patterns.items():
            count = sum(1 for title in titles if pattern in title)
            if count > 0:
                pattern_counts[name] = count
        
        # Use key terms if no strong pattern
        if pattern_counts:
            # Get top pattern
            top_pattern = pattern_counts.most_common(1)[0][0]
            cluster_names[cluster_id] = top_pattern
        else:
            # Generate from key terms
            terms = key_terms.get(cluster_id, [])
            if terms:
                # Clean and format terms
                clean_terms = [t.replace('_', ' ').title() for t in terms[:3]]
                cluster_names[cluster_id] = f"{' '.join(clean_terms)} Methods"
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}"
    
    return cluster_names

def analyze_clusters(papers, cluster_labels, cluster_names):
    """Analyze cluster characteristics."""
    cluster_stats = {}
    
    for cluster_id in set(cluster_labels):
        cluster_papers = [p for p, l in zip(papers, cluster_labels) if l == cluster_id]
        
        # Calculate statistics
        years = [p.get('publication_year', 0) for p in cluster_papers if p.get('publication_year')]
        citations = [p.get('citation_count', 0) for p in cluster_papers if p.get('citation_count') is not None]
        
        stats = {
            'name': cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
            'paper_count': len(cluster_papers),
            'avg_year': np.mean(years) if years else 0,
            'avg_citations': np.mean(citations) if citations else 0,
            'max_citations': max(citations) if citations else 0,
            'paper_ids': [p['id'] for p in cluster_papers]
        }
        
        cluster_stats[cluster_id] = stats
    
    return cluster_stats

def identify_relationships(embeddings, cluster_labels):
    """Identify relationships between clusters."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    n_clusters = len(set(cluster_labels))
    relationships = []
    
    # Calculate cluster centroids
    centroids = []
    for i in range(n_clusters):
        cluster_embeddings = embeddings[cluster_labels == i]
        centroid = cluster_embeddings.mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(centroids)
    
    # Find strong relationships (similarity > threshold)
    threshold = 0.5
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            sim = similarities[i, j]
            if sim > threshold:
                relationship = {
                    'cluster_1': i,
                    'cluster_2': j,
                    'relationship_type': 'overlapping' if sim > 0.7 else 'complementary',
                    'strength': float(sim)
                }
                relationships.append(relationship)
    
    return relationships

def perform_clustering(papers, n_clusters):
    """Perform the actual clustering."""
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    
    # Generate embeddings
    embeddings = generate_embeddings(papers)
    
    # Normalize embeddings
    embeddings = StandardScaler().fit_transform(embeddings)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate quality metrics
    silhouette = silhouette_score(embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
    
    print(f"\nClustering Quality Metrics:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.1f}")
    
    return embeddings, cluster_labels, {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }

def main():
    """Main clustering pipeline."""
    # Load papers
    papers = load_papers('/path/to/project/output/llm_rlhf_alignment/output/papers.json')
    print(f"Loaded {len(papers)} papers")
    
    # Generate initial embeddings for optimization
    embeddings = generate_embeddings(papers)
    embeddings = StandardScaler().fit_transform(embeddings)
    
    # Find optimal number of clusters
    optimal_k, scores = find_optimal_clusters(embeddings, min_clusters=5, max_clusters=15)
    
    # Perform final clustering with optimal k
    embeddings, cluster_labels, quality_metrics = perform_clustering(papers, optimal_k)
    
    # Extract key terms for each cluster
    key_terms = extract_key_terms(papers, cluster_labels)
    
    # Generate cluster names
    cluster_names = name_clusters(papers, cluster_labels, key_terms)
    
    # Analyze clusters
    cluster_stats = analyze_clusters(papers, cluster_labels, cluster_names)
    
    # Identify relationships
    relationships = identify_relationships(embeddings, cluster_labels)
    
    # Prepare output
    clusters = []
    for cluster_id, stats in cluster_stats.items():
        cluster_data = {
            'id': int(cluster_id),
            'name': stats['name'],
            'description': f"Research cluster focusing on {stats['name'].lower()}",
            'paper_count': int(stats['paper_count']),
            'papers': stats['paper_ids'],
            'key_terms': [str(term) for term in key_terms.get(cluster_id, [])],
            'avg_year': float(stats['avg_year']) if stats['avg_year'] else 0,
            'avg_citations': float(stats['avg_citations']) if stats['avg_citations'] else 0,
            'confidence': float(0.8 + (quality_metrics['silhouette'] * 0.2))  # Confidence based on quality
        }
        clusters.append(cluster_data)
    
    # Sort clusters by paper count
    clusters.sort(key=lambda x: x['paper_count'], reverse=True)
    
    # Identify outliers (papers in smallest clusters)
    outliers = []
    min_cluster_size = 3
    for cluster in clusters:
        if cluster['paper_count'] < min_cluster_size:
            for paper_id in cluster['papers']:
                outliers.append({
                    'paper_id': paper_id,
                    'reason': f"In small cluster ({cluster['paper_count']} papers)"
                })
    
    # Prepare final output
    output = {
        'clusters': clusters,
        'relationships': [{
            'cluster_1': int(r['cluster_1']),
            'cluster_2': int(r['cluster_2']),
            'relationship_type': r['relationship_type'],
            'strength': float(r['strength'])
        } for r in relationships],
        'clustering_metadata': {
            'total_papers': int(len(papers)),
            'num_clusters': int(optimal_k),
            'unclustered_papers': int(len(outliers)),
            'avg_cluster_size': float(len(papers) / optimal_k),
            'clustering_quality': float(quality_metrics['silhouette'])
        },
        'outliers': outliers
    }
    
    # Save results
    output_path = '/path/to/project/output/llm_rlhf_alignment/output/clusters.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nClusters saved to {output_path}")
    
    # Generate report
    generate_report(output, quality_metrics, scores)
    
    return output

def generate_report(clustering_output, quality_metrics, optimization_scores):
    """Generate a detailed clustering report."""
    report = []
    report.append("# LLM RLHF Alignment Papers - Clustering Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overview
    report.append("\n## Overview")
    meta = clustering_output['clustering_metadata']
    report.append(f"- **Total Papers**: {meta['total_papers']}")
    report.append(f"- **Number of Clusters**: {meta['num_clusters']}")
    report.append(f"- **Average Cluster Size**: {meta['avg_cluster_size']:.1f}")
    report.append(f"- **Clustering Quality (Silhouette)**: {meta['clustering_quality']:.3f}")
    
    # Quality Metrics
    report.append("\n## Clustering Quality Metrics")
    report.append(f"- **Silhouette Score**: {quality_metrics['silhouette']:.3f} (Higher is better, range: -1 to 1)")
    report.append(f"- **Davies-Bouldin Score**: {quality_metrics['davies_bouldin']:.3f} (Lower is better)")
    report.append(f"- **Calinski-Harabasz Score**: {quality_metrics['calinski_harabasz']:.1f} (Higher is better)")
    
    # Clusters
    report.append("\n## Research Clusters")
    for cluster in clustering_output['clusters']:
        report.append(f"\n### {cluster['id'] + 1}. {cluster['name']}")
        report.append(f"- **Papers**: {cluster['paper_count']}")
        report.append(f"- **Average Year**: {cluster['avg_year']:.1f}")
        report.append(f"- **Average Citations**: {cluster['avg_citations']:.1f}")
        report.append(f"- **Key Terms**: {', '.join(cluster['key_terms'][:5])}")
        report.append(f"- **Confidence**: {cluster['confidence']:.2f}")
    
    # Relationships
    if clustering_output['relationships']:
        report.append("\n## Inter-Cluster Relationships")
        for rel in clustering_output['relationships']:
            c1_name = clustering_output['clusters'][rel['cluster_1']]['name']
            c2_name = clustering_output['clusters'][rel['cluster_2']]['name']
            report.append(f"- **{c1_name}** ↔ **{c2_name}**: {rel['relationship_type']} (strength: {rel['strength']:.2f})")
    
    # Outliers
    if clustering_output['outliers']:
        report.append(f"\n## Outliers")
        report.append(f"Found {len(clustering_output['outliers'])} papers in very small clusters")
    
    # Save report
    report_path = '/path/to/project/output/llm_rlhf_alignment/output/cluster_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()