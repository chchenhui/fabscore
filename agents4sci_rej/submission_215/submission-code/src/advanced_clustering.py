#!/usr/bin/env python3
"""
Advanced clustering for LLM RLHF alignment papers with better cluster identification.
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
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
    """Prepare paper text for embedding generation with better feature extraction."""
    # Get all fields
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    
    # Extract key phrases from title
    title_lower = title.lower()
    
    # Weight important keywords higher
    important_keywords = []
    if 'dpo' in title_lower or 'direct preference' in title_lower:
        important_keywords.append('direct preference optimization')
    if 'constitutional' in title_lower:
        important_keywords.append('constitutional ai')
    if 'ppo' in title_lower or 'proximal policy' in title_lower:
        important_keywords.append('proximal policy optimization')
    if 'reward model' in title_lower:
        important_keywords.append('reward modeling')
    if 'safety' in title_lower or 'safe' in title_lower:
        important_keywords.append('safety alignment')
    if 'evaluation' in title_lower or 'benchmark' in title_lower:
        important_keywords.append('evaluation benchmarking')
    if 'instruction' in title_lower:
        important_keywords.append('instruction tuning')
    if 'human feedback' in title_lower or 'rlhf' in title_lower:
        important_keywords.append('reinforcement learning human feedback')
    if 'preference' in title_lower:
        important_keywords.append('preference learning')
    if 'harmless' in title_lower or 'helpful' in title_lower or 'honest' in title_lower:
        important_keywords.append('harmless helpful honest')
    if 'attack' in title_lower or 'jailbreak' in title_lower or 'adversarial' in title_lower:
        important_keywords.append('adversarial robustness attacks')
    if 'toxic' in title_lower or 'bias' in title_lower:
        important_keywords.append('toxicity bias mitigation')
    if 'multimodal' in title_lower or 'vision' in title_lower or 'image' in title_lower:
        important_keywords.append('multimodal alignment')
    if 'reasoning' in title_lower or 'chain' in title_lower or 'thought' in title_lower:
        important_keywords.append('reasoning chain of thought')
    if 'efficient' in title_lower or 'scalab' in title_lower:
        important_keywords.append('efficient scalable methods')
    if 'dataset' in title_lower or 'data' in title_lower:
        important_keywords.append('dataset creation')
    if 'theory' in title_lower or 'theoret' in title_lower:
        important_keywords.append('theoretical foundations')
    if 'survey' in title_lower or 'review' in title_lower:
        important_keywords.append('survey review')
    if 'llama' in title_lower or 'alpaca' in title_lower or 'vicuna' in title_lower:
        important_keywords.append('open source models')
    if 'gpt' in title_lower or 'chatgpt' in title_lower:
        important_keywords.append('gpt models')
    if 'claude' in title_lower or 'anthropic' in title_lower:
        important_keywords.append('anthropic claude')
    if 'debate' in title_lower or 'self-critique' in title_lower or 'critique' in title_lower:
        important_keywords.append('self critique debate')
    
    # Combine with emphasis on keywords
    keyword_string = ' '.join(important_keywords * 3)  # Triple weight
    text = f"{keyword_string} {title} {title} {abstract}"  # Double weight title
    
    # Truncate if too long
    max_length = 512
    if len(text) > max_length:
        text = text[:max_length]
    
    return text

def generate_embeddings(papers, model_name='sentence-transformers/all-mpnet-base-v2'):
    """Generate embeddings using a better model."""
    print(f"\nGenerating embeddings using {model_name}...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Use a more powerful model for better clustering
    model = SentenceTransformer(model_name, device=device)
    
    # Prepare texts with better feature extraction
    texts = [prepare_text_for_embedding(paper) for paper in papers]
    
    # Generate embeddings
    print(f"Embedding {len(texts)} papers...")
    embeddings = model.encode(texts, 
                            show_progress_bar=True,
                            convert_to_numpy=True,
                            batch_size=32 if device == 'cuda' else 16)
    
    return embeddings

def find_optimal_clusters_advanced(embeddings, min_clusters=8, max_clusters=15):
    """Find optimal clusters with better heuristics for RLHF papers."""
    print(f"\nFinding optimal number of clusters ({min_clusters}-{max_clusters})...")
    
    # Apply PCA for noise reduction
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(embeddings)
    
    scores = {
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': [],
        'inertia': []
    }
    
    K = range(min_clusters, max_clusters + 1)
    
    for k in K:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = kmeans.fit_predict(embeddings_pca)
        
        # Calculate metrics
        scores['silhouette'].append(silhouette_score(embeddings_pca, labels))
        scores['davies_bouldin'].append(davies_bouldin_score(embeddings_pca, labels))
        scores['calinski_harabasz'].append(calinski_harabasz_score(embeddings_pca, labels))
        scores['inertia'].append(kmeans.inertia_)
    
    # Elbow method for inertia
    inertia_diffs = np.diff(scores['inertia'])
    inertia_diffs_2nd = np.diff(inertia_diffs)
    elbow_k = min_clusters + 2 + np.argmax(inertia_diffs_2nd)
    
    # Get best k from each metric
    silhouette_optimal = min_clusters + np.argmax(scores['silhouette'])
    davies_optimal = min_clusters + np.argmin(scores['davies_bouldin'])
    calinski_optimal = min_clusters + np.argmax(scores['calinski_harabasz'])
    
    # For RLHF papers, we expect around 10-12 major themes
    # Weight towards middle range
    candidates = [silhouette_optimal, davies_optimal, calinski_optimal, elbow_k]
    weights = []
    for k in candidates:
        if 9 <= k <= 12:
            weights.append(2.0)  # Prefer 9-12 clusters
        else:
            weights.append(1.0)
    
    # Weighted vote
    optimal_k = int(np.average(candidates, weights=weights))
    
    print(f"\nOptimal clusters by metric:")
    print(f"  Silhouette: {silhouette_optimal}")
    print(f"  Davies-Bouldin: {davies_optimal}")
    print(f"  Calinski-Harabasz: {calinski_optimal}")
    print(f"  Elbow method: {elbow_k}")
    print(f"  Selected (weighted): {optimal_k}")
    
    return optimal_k, scores, embeddings_pca

def extract_detailed_key_terms(papers, cluster_labels, n_terms=10):
    """Extract more detailed key terms for each cluster."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    cluster_texts = defaultdict(list)
    for paper, label in zip(papers, cluster_labels):
        # Combine title and abstract with more weight on title
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        text = f"{title} {title} {abstract}"  # Double weight title
        cluster_texts[label].append(text)
    
    key_terms = {}
    for cluster_id, texts in cluster_texts.items():
        # Use TF-IDF with better parameters
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 3),  # Include bigrams and trigrams
            min_df=2,  # Term must appear in at least 2 documents
            max_df=0.8  # Exclude terms that appear in >80% of documents
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms based on mean TF-IDF
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-n_terms:][::-1]
            
            # Filter out generic terms
            generic_terms = {'model', 'models', 'language', 'large', 'learning', 'using', 'based'}
            terms = []
            for i in top_indices:
                term = feature_names[i]
                if not any(word in generic_terms for word in term.split()):
                    terms.append(term)
            
            key_terms[cluster_id] = terms[:n_terms]
        except:
            key_terms[cluster_id] = []
    
    return key_terms

def name_clusters_advanced(papers, cluster_labels, key_terms):
    """Generate more specific and meaningful cluster names."""
    cluster_names = {}
    cluster_papers = defaultdict(list)
    
    for paper, label in zip(papers, cluster_labels):
        cluster_papers[label].append(paper)
    
    for cluster_id, cluster_paper_list in cluster_papers.items():
        # Analyze titles in detail
        titles = [p.get('title', '').lower() for p in cluster_paper_list]
        abstracts = [p.get('abstract', '').lower() for p in cluster_paper_list]
        
        # Count specific technique mentions
        technique_counts = Counter()
        
        # Check for specific methodologies and topics
        patterns = {
            # Core RLHF methods
            ('rlhf', 'reinforcement learning', 'human feedback'): 'RLHF Core Methods',
            ('dpo', 'direct preference'): 'Direct Preference Optimization (DPO)',
            ('ppo', 'proximal policy'): 'PPO-based Alignment',
            ('constitutional', 'cai', 'self-supervision'): 'Constitutional AI',
            
            # Reward and preference modeling
            ('reward model', 'reward function', 'reward design'): 'Reward Model Design',
            ('preference', 'ranking', 'pairwise'): 'Preference Learning',
            
            # Safety and alignment
            ('safety', 'safe', 'harm', 'risk'): 'Safety and Risk Mitigation',
            ('alignment', 'value alignment', 'goal alignment'): 'Value Alignment Theory',
            ('helpful', 'harmless', 'honest', 'hhh'): 'HHH Alignment Framework',
            
            # Evaluation and benchmarking
            ('evaluation', 'benchmark', 'metric', 'measure'): 'Evaluation and Benchmarks',
            ('dataset', 'data collection', 'annotation'): 'Datasets and Annotation',
            
            # Specific applications
            ('instruction', 'instruct', 'following'): 'Instruction Following',
            ('chat', 'dialogue', 'conversation'): 'Conversational Alignment',
            ('reasoning', 'chain of thought', 'cot'): 'Reasoning Enhancement',
            ('multimodal', 'vision', 'image', 'visual'): 'Multimodal Alignment',
            
            # Efficiency and scaling
            ('efficient', 'parameter-efficient', 'lora', 'adapter'): 'Efficient Training Methods',
            ('scalable', 'scaling', 'large-scale'): 'Scalable Alignment',
            
            # Adversarial and robustness
            ('jailbreak', 'attack', 'adversarial', 'robust'): 'Adversarial Robustness',
            ('toxic', 'bias', 'fairness', 'stereotype'): 'Bias and Toxicity Mitigation',
            
            # Theoretical work
            ('theory', 'theoretical', 'analysis', 'proof'): 'Theoretical Foundations',
            ('survey', 'review', 'overview', 'systematic'): 'Surveys and Reviews',
            
            # Model-specific
            ('gpt', 'chatgpt', 'openai'): 'GPT-based Alignment',
            ('llama', 'alpaca', 'vicuna', 'open'): 'Open Model Alignment',
            ('claude', 'anthropic'): 'Anthropic Methods',
            
            # Advanced techniques
            ('debate', 'self-critique', 'critique', 'reflection'): 'Self-Critique and Debate',
            ('rlaif', 'ai feedback', 'self-training'): 'AI Feedback (RLAIF)',
            ('red team', 'red-team', 'adversarial training'): 'Red Teaming',
        }
        
        # Score each pattern
        pattern_scores = {}
        for pattern_tuple, name in patterns.items():
            score = 0
            for pattern in pattern_tuple:
                # Check titles (weighted more)
                title_matches = sum(1 for t in titles if pattern in t)
                # Check abstracts
                abstract_matches = sum(1 for a in abstracts if pattern in a) * 0.5
                score += title_matches * 2 + abstract_matches
            
            if score > 0:
                pattern_scores[name] = score
        
        # Select best matching pattern
        if pattern_scores:
            best_name = max(pattern_scores, key=pattern_scores.get)
            # Add paper count context if it's a major cluster
            if len(cluster_paper_list) > 30:
                cluster_names[cluster_id] = f"{best_name} (Major Theme)"
            else:
                cluster_names[cluster_id] = best_name
        else:
            # Fallback to key terms
            terms = key_terms.get(cluster_id, [])
            if terms:
                # Use first few meaningful terms
                meaningful_terms = [t for t in terms[:3] if len(t.split()) <= 3]
                if meaningful_terms:
                    cluster_names[cluster_id] = ' '.join(meaningful_terms).title()
                else:
                    cluster_names[cluster_id] = f"Cluster {cluster_id + 1}"
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}"
    
    return cluster_names

def perform_advanced_clustering(papers, n_clusters):
    """Perform clustering with advanced techniques."""
    print(f"\nPerforming advanced clustering with {n_clusters} clusters...")
    
    # Generate embeddings with better model
    embeddings = generate_embeddings(papers)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca = pca.fit_transform(StandardScaler().fit_transform(embeddings))
    
    # Use multiple clustering attempts and select best
    best_score = -1
    best_labels = None
    
    for attempt in range(5):
        # Try both KMeans and Agglomerative
        kmeans = KMeans(n_clusters=n_clusters, random_state=42 + attempt, n_init=30, max_iter=500)
        kmeans_labels = kmeans.fit_predict(embeddings_pca)
        kmeans_score = silhouette_score(embeddings_pca, kmeans_labels)
        
        if kmeans_score > best_score:
            best_score = kmeans_score
            best_labels = kmeans_labels
    
    # Try agglomerative as well
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    agglo_labels = agglo.fit_predict(embeddings_pca)
    agglo_score = silhouette_score(embeddings_pca, agglo_labels)
    
    if agglo_score > best_score:
        best_score = agglo_score
        best_labels = agglo_labels
        print("Using Agglomerative clustering (better score)")
    else:
        print("Using KMeans clustering")
    
    # Calculate final metrics
    silhouette = silhouette_score(embeddings_pca, best_labels)
    davies_bouldin = davies_bouldin_score(embeddings_pca, best_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings_pca, best_labels)
    
    print(f"\nFinal Clustering Quality Metrics:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.1f}")
    
    return embeddings_pca, best_labels, {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }

def main():
    """Main clustering pipeline with advanced techniques."""
    # Load papers
    papers = load_papers('/path/to/project/output/llm_rlhf_alignment/output/papers.json')
    print(f"Loaded {len(papers)} papers")
    
    # Generate initial embeddings for optimization
    embeddings = generate_embeddings(papers)
    embeddings_scaled = StandardScaler().fit_transform(embeddings)
    
    # Find optimal number of clusters with better range
    optimal_k, scores, embeddings_pca = find_optimal_clusters_advanced(
        embeddings_scaled, min_clusters=8, max_clusters=15
    )
    
    # Override if too few clusters for 443 papers
    if optimal_k < 10:
        print(f"Overriding optimal k={optimal_k} to k=10 for better granularity")
        optimal_k = 10
    
    # Perform final clustering
    embeddings_final, cluster_labels, quality_metrics = perform_advanced_clustering(papers, optimal_k)
    
    # Extract detailed key terms
    key_terms = extract_detailed_key_terms(papers, cluster_labels, n_terms=10)
    
    # Generate meaningful cluster names
    cluster_names = name_clusters_advanced(papers, cluster_labels, key_terms)
    
    # Analyze clusters
    cluster_stats = {}
    for cluster_id in range(optimal_k):
        cluster_papers = [p for p, l in zip(papers, cluster_labels) if l == cluster_id]
        
        # Calculate statistics
        years = [p.get('publication_year', 0) for p in cluster_papers if p.get('publication_year')]
        citations = [p.get('citation_count', 0) for p in cluster_papers if p.get('citation_count') is not None]
        
        # Get sample titles for verification
        sample_titles = [p.get('title', '') for p in cluster_papers[:5]]
        
        stats = {
            'name': cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
            'paper_count': len(cluster_papers),
            'avg_year': np.mean(years) if years else 0,
            'avg_citations': np.mean(citations) if citations else 0,
            'max_citations': max(citations) if citations else 0,
            'paper_ids': [p['id'] for p in cluster_papers],
            'sample_titles': sample_titles,
            'description': f"Research focusing on {cluster_names.get(cluster_id, 'unspecified topic').lower()}"
        }
        
        cluster_stats[cluster_id] = stats
    
    # Identify relationships between clusters
    from sklearn.metrics.pairwise import cosine_similarity
    
    relationships = []
    # Calculate cluster centroids
    centroids = []
    for i in range(optimal_k):
        cluster_embeddings = embeddings_final[cluster_labels == i]
        if len(cluster_embeddings) > 0:
            centroid = cluster_embeddings.mean(axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(embeddings_final.shape[1]))
    
    centroids = np.array(centroids)
    similarities = cosine_similarity(centroids)
    
    # Find strong relationships
    for i in range(optimal_k):
        for j in range(i+1, optimal_k):
            sim = similarities[i, j]
            if sim > 0.3:  # Lower threshold for better relationship detection
                rel_type = 'overlapping' if sim > 0.6 else 'complementary'
                relationships.append({
                    'cluster_1': i,
                    'cluster_2': j,
                    'relationship_type': rel_type,
                    'strength': float(sim)
                })
    
    # Sort relationships by strength
    relationships.sort(key=lambda x: x['strength'], reverse=True)
    
    # Prepare final output
    clusters = []
    for cluster_id, stats in cluster_stats.items():
        cluster_data = {
            'id': int(cluster_id),
            'name': stats['name'],
            'description': stats['description'],
            'paper_count': int(stats['paper_count']),
            'papers': stats['paper_ids'],
            'key_terms': [str(term) for term in key_terms.get(cluster_id, [])[:7]],
            'avg_year': float(stats['avg_year']) if stats['avg_year'] else 0,
            'avg_citations': float(stats['avg_citations']) if stats['avg_citations'] else 0,
            'confidence': float(0.7 + min(0.3, quality_metrics['silhouette']))
        }
        clusters.append(cluster_data)
    
    # Sort clusters by paper count
    clusters.sort(key=lambda x: x['paper_count'], reverse=True)
    
    # Identify outliers
    outliers = []
    min_cluster_size = 5
    for cluster in clusters:
        if cluster['paper_count'] < min_cluster_size:
            for paper_id in cluster['papers']:
                outliers.append({
                    'paper_id': paper_id,
                    'reason': f"In small cluster ({cluster['paper_count']} papers)"
                })
    
    # Final output
    output = {
        'clusters': clusters,
        'relationships': [{
            'cluster_1': int(r['cluster_1']),
            'cluster_2': int(r['cluster_2']),
            'relationship_type': r['relationship_type'],
            'strength': float(r['strength'])
        } for r in relationships[:20]],  # Top 20 relationships
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
    
    # Generate detailed report
    generate_detailed_report(output, quality_metrics, cluster_stats)
    
    return output

def generate_detailed_report(clustering_output, quality_metrics, cluster_stats):
    """Generate a comprehensive clustering report."""
    report = []
    report.append("# LLM RLHF Alignment Papers - Advanced Clustering Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overview
    report.append("\n## Executive Summary")
    meta = clustering_output['clustering_metadata']
    report.append(f"Successfully clustered **{meta['total_papers']} papers** on LLM RLHF alignment into **{meta['num_clusters']} distinct research themes**.")
    
    report.append("\n## Clustering Statistics")
    report.append(f"- **Total Papers**: {meta['total_papers']}")
    report.append(f"- **Number of Clusters**: {meta['num_clusters']}")
    report.append(f"- **Average Cluster Size**: {meta['avg_cluster_size']:.1f} papers")
    report.append(f"- **Clustering Quality Score**: {meta['clustering_quality']:.3f}")
    
    # Quality Assessment
    report.append("\n## Quality Assessment")
    quality_interpretation = "Poor" if meta['clustering_quality'] < 0.25 else "Fair" if meta['clustering_quality'] < 0.5 else "Good" if meta['clustering_quality'] < 0.75 else "Excellent"
    report.append(f"- **Overall Quality**: {quality_interpretation}")
    report.append(f"- **Silhouette Score**: {quality_metrics['silhouette']:.3f} (measures cluster cohesion)")
    report.append(f"- **Davies-Bouldin Score**: {quality_metrics['davies_bouldin']:.3f} (lower is better)")
    report.append(f"- **Calinski-Harabasz Score**: {quality_metrics['calinski_harabasz']:.1f} (higher indicates better defined clusters)")
    
    # Research Themes
    report.append("\n## Identified Research Themes")
    report.append("\nThe following major research themes were identified in the LLM RLHF alignment literature:")
    
    for i, cluster in enumerate(clustering_output['clusters'], 1):
        report.append(f"\n### {i}. {cluster['name']}")
        report.append(f"**{cluster['paper_count']} papers** | Avg. year: {cluster['avg_year']:.1f} | Avg. citations: {cluster['avg_citations']:.0f}")
        report.append(f"\n{cluster['description']}")
        
        if cluster['key_terms']:
            report.append(f"\n**Key concepts**: {', '.join(cluster['key_terms'][:5])}")
        
        # Add sample papers if available
        if cluster['paper_count'] > 0:
            # Get actual paper titles
            cluster_id = cluster['id']
            if cluster_id in cluster_stats:
                sample_titles = cluster_stats[cluster_id].get('sample_titles', [])
                if sample_titles:
                    report.append("\n**Representative papers**:")
                    for j, title in enumerate(sample_titles[:3], 1):
                        if title:
                            report.append(f"  {j}. {title}")
    
    # Cluster Relationships
    if clustering_output['relationships']:
        report.append("\n## Inter-Theme Relationships")
        report.append("\nSignificant relationships between research themes:")
        
        # Group by relationship type
        overlapping = [r for r in clustering_output['relationships'] if r['relationship_type'] == 'overlapping']
        complementary = [r for r in clustering_output['relationships'] if r['relationship_type'] == 'complementary']
        
        if overlapping:
            report.append("\n### Overlapping Themes")
            for rel in overlapping[:5]:
                c1_name = clustering_output['clusters'][rel['cluster_1']]['name']
                c2_name = clustering_output['clusters'][rel['cluster_2']]['name']
                report.append(f"- **{c1_name}** ↔ **{c2_name}** (similarity: {rel['strength']:.2f})")
        
        if complementary:
            report.append("\n### Complementary Themes")
            for rel in complementary[:5]:
                c1_name = clustering_output['clusters'][rel['cluster_1']]['name']
                c2_name = clustering_output['clusters'][rel['cluster_2']]['name']
                report.append(f"- **{c1_name}** ↔ **{c2_name}** (similarity: {rel['strength']:.2f})")
    
    # Analysis Insights
    report.append("\n## Key Insights")
    
    # Find dominant themes
    largest_clusters = clustering_output['clusters'][:3]
    report.append("\n### Dominant Research Areas")
    for cluster in largest_clusters:
        percentage = (cluster['paper_count'] / meta['total_papers']) * 100
        report.append(f"- **{cluster['name']}**: {percentage:.1f}% of all papers")
    
    # Emerging vs established
    report.append("\n### Research Maturity")
    high_citation_clusters = [c for c in clustering_output['clusters'] if c['avg_citations'] > 50]
    low_citation_clusters = [c for c in clustering_output['clusters'] if c['avg_citations'] < 10]
    
    if high_citation_clusters:
        report.append("\n**Established areas** (high average citations):")
        for cluster in high_citation_clusters[:3]:
            report.append(f"- {cluster['name']} ({cluster['avg_citations']:.0f} avg citations)")
    
    if low_citation_clusters:
        report.append("\n**Emerging areas** (recent/low citations):")
        for cluster in low_citation_clusters[:3]:
            report.append(f"- {cluster['name']} ({cluster['avg_citations']:.0f} avg citations)")
    
    # Outliers
    if clustering_output['outliers']:
        report.append(f"\n## Outliers")
        report.append(f"Identified {len(clustering_output['outliers'])} papers that don't fit well into major themes.")
        report.append("These may represent:")
        report.append("- Novel research directions")
        report.append("- Interdisciplinary work")
        report.append("- Highly specialized subtopics")
    
    # Recommendations
    report.append("\n## Recommendations for Survey Writing")
    report.append("\nBased on the clustering analysis:")
    report.append("1. **Structure**: Organize the survey around the major themes identified")
    report.append("2. **Coverage**: Ensure representation from all major clusters")
    report.append("3. **Emphasis**: Allocate space proportional to cluster sizes")
    report.append("4. **Connections**: Highlight relationships between overlapping themes")
    report.append("5. **Future Work**: Consider emerging areas and outliers for future directions section")
    
    # Save report
    report_path = '/path/to/project/output/llm_rlhf_alignment/output/cluster_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Detailed report saved to {report_path}")

if __name__ == "__main__":
    main()