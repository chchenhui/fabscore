#!/usr/bin/env python3
"""
Improved Topic Mining & Clustering for Multimodal RL Papers
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict, Counter
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load papers
print("\n1. Loading papers...")
with open('output-multi-agent-v2/multimodal_rl_papers.json', 'r') as f:
    data = json.load(f)

papers = data['papers']
print(f"Loaded {len(papers)} papers")

# Prepare texts for embedding - more comprehensive
print("\n2. Preparing texts for embedding...")
def prepare_text(paper):
    """Combine title, abstract, and venue for richer embeddings"""
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    venue = paper.get('venue', '')
    # Give more weight to title
    return f"{title}. {title}. {abstract}. Published at {venue}"

texts = [prepare_text(p) for p in papers]

# Generate embeddings using a better model
print("\n3. Generating embeddings with sentence-transformers...")
# Use a more powerful model for better embeddings
model = SentenceTransformer('all-mpnet-base-v2', device=device)

# Process in batches
batch_size = 16 if device == 'cuda' else 8
embeddings = model.encode(
    texts,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True  # L2 normalization
)

print(f"Embeddings shape: {embeddings.shape}")

# Apply PCA for dimensionality reduction (can help with clustering)
print("\n4. Applying dimensionality reduction...")
pca = PCA(n_components=100, random_state=42)
embeddings_reduced = pca.fit_transform(embeddings)
print(f"Reduced embeddings shape: {embeddings_reduced.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")

# Normalize after PCA
embeddings_normalized = StandardScaler().fit_transform(embeddings_reduced)

# More sophisticated cluster naming function
def extract_domain_themes(cluster_papers):
    """Extract domain-specific themes from papers"""
    
    # Combine titles and abstracts
    texts = []
    titles = []
    for p in cluster_papers:
        texts.append(p.get('abstract', ''))
        titles.append(p.get('title', '').lower())
    
    combined_text = ' '.join(texts).lower()
    combined_titles = ' '.join(titles)
    
    # Define domain-specific patterns and their cluster names
    theme_patterns = [
        # Robotics and Embodied AI
        (['robot', 'manipulat', 'grasp', 'pick', 'place', 'arm', 'gripper'], 
         "Robotic Manipulation and Control"),
        (['navigation', 'navigate', 'path', 'explore', 'slam', 'mapping'], 
         "Navigation and Spatial Reasoning"),
        (['embodied', 'embodi', 'physical', 'real-world', 'sensor'], 
         "Embodied AI and Physical Interaction"),
        
        # Vision-Language
        (['vision-language', 'vision language', 'vln', 'visual language', 'image text', 'clip'], 
         "Vision-Language Models and Understanding"),
        (['instruction', 'command', 'follow', 'task completion', 'directive'], 
         "Instruction Following and Task Execution"),
        (['caption', 'describe', 'visual question', 'vqa', 'visual dialog'], 
         "Visual Question Answering and Captioning"),
        
        # Learning Paradigms
        (['rlhf', 'human feedback', 'preference', 'human-in-the-loop', 'alignment'], 
         "RLHF and Human-Aligned Learning"),
        (['reward model', 'reward learn', 'reward shaping', 'reward design'], 
         "Reward Modeling and Design"),
        (['imitation', 'demonstration', 'behavior clon', 'expert'], 
         "Imitation Learning and Demonstrations"),
        (['offline', 'batch rl', 'dataset', 'logged'], 
         "Offline RL and Dataset-based Learning"),
        
        # Applications
        (['game', 'gaming', 'video game', 'atari', 'minecraft', 'starcraft'], 
         "Game Playing and Virtual Environments"),
        (['medical', 'clinical', 'healthcare', 'diagnosis', 'patient', 'treatment'], 
         "Medical and Healthcare Applications"),
        (['autonomous', 'driving', 'vehicle', 'traffic', 'road'], 
         "Autonomous Driving and Transportation"),
        (['dialogue', 'conversation', 'chat', 'assistant', 'dialog'], 
         "Dialogue Systems and Conversational AI"),
        
        # Technical Approaches
        (['diffusion', 'denois', 'generative', 'synthesis', 'generate'], 
         "Diffusion Models and Generation"),
        (['transformer', 'attention', 'bert', 'gpt', 'llm', 'large language'], 
         "Transformer-based Multimodal Models"),
        (['reasoning', 'reason', 'logic', 'chain-of-thought', 'cot', 'planning'], 
         "Reasoning and Planning Capabilities"),
        (['multi-agent', 'multiagent', 'cooperative', 'collaboration'], 
         "Multi-Agent Systems and Cooperation"),
        (['meta', 'few-shot', 'zero-shot', 'adaptation', 'transfer'], 
         "Meta-Learning and Few-Shot Adaptation"),
        
        # Optimization
        (['policy gradient', 'ppo', 'a3c', 'actor-critic', 'policy optim'], 
         "Policy Gradient Methods"),
        (['q-learning', 'dqn', 'value', 'bellman', 'td-learning'], 
         "Value-Based RL Methods"),
        (['exploration', 'curiosity', 'intrinsic', 'novelty'], 
         "Exploration and Intrinsic Motivation"),
    ]
    
    # Count pattern matches
    theme_scores = {}
    for patterns, theme_name in theme_patterns:
        score = 0
        for pattern in patterns:
            # Check in combined text
            score += len(re.findall(pattern, combined_text)) * 0.5
            # Give extra weight to title matches
            score += len(re.findall(pattern, combined_titles)) * 2
        
        if score > 0:
            theme_scores[theme_name] = score
    
    # Return the theme with highest score
    if theme_scores:
        return max(theme_scores, key=theme_scores.get)
    
    return None

def generate_improved_cluster_name(cluster_papers, cluster_id):
    """Generate more specific and descriptive cluster names"""
    
    # First try domain-specific themes
    theme_name = extract_domain_themes(cluster_papers)
    if theme_name:
        return theme_name
    
    # Fallback to TF-IDF based naming
    texts = [p.get('abstract', '') for p in cluster_papers]
    
    if not texts or all(not t for t in texts):
        return f"Multimodal RL Cluster {cluster_id}"
    
    try:
        # Use bigrams and trigrams for more specific terms
        vectorizer = TfidfVectorizer(
            max_features=30,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=0.15,  # Term must appear in at least 15% of cluster docs
            max_df=0.8    # But not in more than 80%
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores
        scores = tfidf_matrix.mean(axis=0).A1
        top_indices = scores.argsort()[-15:][::-1]
        
        # Filter and select best terms
        top_terms = []
        for idx in top_indices:
            term = terms[idx]
            # Skip generic terms
            if term not in ['model', 'learning', 'using', 'based', 'method', 'approach', 'paper', 'work']:
                top_terms.append(term)
        
        if len(top_terms) >= 2:
            # Combine top distinctive terms
            return f"{top_terms[0].title()} and {top_terms[1].title()}"
        elif top_terms:
            return top_terms[0].title()
        else:
            return f"Multimodal RL Applications {cluster_id}"
            
    except Exception as e:
        print(f"Error in TF-IDF naming: {e}")
        return f"Multimodal RL Cluster {cluster_id}"

# Try multiple clustering approaches
print("\n5. Testing multiple clustering approaches...")
k_range = range(8, 11)
best_score = -1
best_labels = None
best_k = None
best_method = None

# Try KMeans
print("\n   Testing KMeans...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(embeddings_normalized)
    score = silhouette_score(embeddings_normalized, labels)
    print(f"   K={k}: Silhouette={score:.4f}")
    
    if score > best_score:
        best_score = score
        best_labels = labels
        best_k = k
        best_method = "KMeans"

# Try Agglomerative Clustering
print("\n   Testing Agglomerative Clustering...")
for k in k_range:
    agglo = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = agglo.fit_predict(embeddings_normalized)
    score = silhouette_score(embeddings_normalized, labels)
    print(f"   K={k}: Silhouette={score:.4f}")
    
    if score > best_score:
        best_score = score
        best_labels = labels
        best_k = k
        best_method = "Agglomerative"

print(f"\nBest method: {best_method} with K={best_k}, Silhouette={best_score:.4f}")

# Use best clustering
cluster_labels = best_labels
optimal_k = best_k

# Calculate final metrics
final_silhouette = silhouette_score(embeddings_normalized, cluster_labels)
final_db = davies_bouldin_score(embeddings_normalized, cluster_labels)
final_ch = calinski_harabasz_score(embeddings_normalized, cluster_labels)

print(f"\n6. Final clustering metrics:")
print(f"   Silhouette Score: {final_silhouette:.4f}")
print(f"   Davies-Bouldin Score: {final_db:.4f}")
print(f"   Calinski-Harabasz Score: {final_ch:.4f}")

# Analyze clusters
print("\n7. Analyzing clusters...")
clusters = defaultdict(list)

for idx, label in enumerate(cluster_labels):
    clusters[label].append(idx)

# Build detailed cluster information
cluster_info = []

for cluster_id in sorted(clusters.keys()):
    paper_indices = clusters[cluster_id]
    cluster_papers = [papers[i] for i in paper_indices]
    
    # Generate improved cluster name
    cluster_name = generate_improved_cluster_name(cluster_papers, cluster_id)
    
    # Calculate statistics
    years = [p.get('year', 0) for p in cluster_papers if p.get('year')]
    citations = [p.get('citations', 0) for p in cluster_papers if p.get('citations') is not None]
    
    avg_year = np.mean(years) if years else 0
    avg_citations = np.mean(citations) if citations else 0
    median_citations = np.median(citations) if citations else 0
    
    # Find representative papers (closest to centroid in original embedding space)
    cluster_embeddings = embeddings[paper_indices]
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    representative_indices = np.argsort(distances)[:5]
    
    representative_papers = []
    for idx in representative_indices:
        paper = cluster_papers[idx]
        representative_papers.append({
            'title': paper.get('title', 'Unknown'),
            'year': paper.get('year', 'N/A'),
            'citations': paper.get('citations', 0),
            'venue': paper.get('venue', 'N/A'),
            'relevance_to_cluster': float(1.0 - distances[idx] / distances.max())
        })
    
    # Extract key terms
    cluster_texts = [p.get('abstract', '') + ' ' + p.get('title', '') for p in cluster_papers]
    if cluster_texts and any(cluster_texts):
        try:
            vec = TfidfVectorizer(
                max_features=15, 
                stop_words='english', 
                ngram_range=(1, 2),
                min_df=2
            )
            tfidf = vec.fit_transform(cluster_texts)
            key_terms = vec.get_feature_names_out().tolist()
            
            # Filter out generic terms
            key_terms = [t for t in key_terms if t not in ['model', 'learning', 'using', 'based', 'paper']][:8]
        except:
            key_terms = []
    else:
        key_terms = []
    
    # Generate detailed description
    description = f"This cluster contains {len(paper_indices)} papers focusing on {cluster_name.lower()}. "
    if key_terms:
        description += f"Key research areas include {', '.join(key_terms[:3])}."
    
    cluster_info.append({
        'id': int(cluster_id),
        'name': cluster_name,
        'description': description,
        'size': len(paper_indices),
        'paper_indices': paper_indices,
        'key_terms': key_terms,
        'representative_papers': representative_papers,
        'avg_year': float(avg_year),
        'avg_citations': float(avg_citations),
        'median_citations': float(median_citations),
        'year_range': [min(years), max(years)] if years else [0, 0]
    })
    
    print(f"\nCluster {cluster_id}: {cluster_name}")
    print(f"  Size: {len(paper_indices)} papers")
    print(f"  Year range: {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}")
    print(f"  Avg Citations: {avg_citations:.1f} (median: {median_citations:.1f})")

# Sort clusters by size
cluster_info.sort(key=lambda x: x['size'], reverse=True)

# Identify outliers
print("\n8. Identifying outliers...")
outliers = []
for cluster_id in clusters.keys():
    paper_indices = clusters[cluster_id]
    if len(paper_indices) < 3:
        continue  # Skip very small clusters
        
    cluster_embeddings = embeddings_normalized[paper_indices]
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    
    # Papers more than 2 std deviations from centroid
    threshold = distances.mean() + 2 * distances.std()
    outlier_mask = distances > threshold
    
    for idx, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            paper_idx = paper_indices[idx]
            outliers.append({
                'paper_index': paper_idx,
                'paper_title': papers[paper_idx].get('title', 'Unknown'),
                'cluster': int(cluster_id),
                'cluster_name': cluster_info[cluster_id]['name'],
                'distance_from_centroid': float(distances[idx]),
                'reason': 'Low similarity to cluster centroid (>2 std dev)'
            })

print(f"Found {len(outliers)} outliers")

# Analyze cluster relationships
print("\n9. Analyzing cluster relationships...")
relationships = []
cluster_centroids = []

for cluster_id in sorted(clusters.keys()):
    paper_indices = clusters[cluster_id]
    cluster_embeddings = embeddings[paper_indices]  # Use original embeddings
    centroid = cluster_embeddings.mean(axis=0)
    cluster_centroids.append(centroid)

# Calculate pairwise similarities
for i in range(len(cluster_centroids)):
    for j in range(i+1, len(cluster_centroids)):
        # Cosine similarity
        similarity = np.dot(cluster_centroids[i], cluster_centroids[j])
        
        if similarity > 0.4:  # Threshold for significant relationship
            rel_type = 'strongly_related' if similarity > 0.6 else 'related' if similarity > 0.5 else 'weakly_related'
            relationships.append({
                'cluster_1': i,
                'cluster_1_name': cluster_info[i]['name'],
                'cluster_2': j,
                'cluster_2_name': cluster_info[j]['name'],
                'relationship': rel_type,
                'strength': float(similarity)
            })

relationships.sort(key=lambda x: x['strength'], reverse=True)
print(f"Found {len(relationships)} cluster relationships")

# Create comprehensive output
output = {
    'metadata': {
        'num_papers': len(papers),
        'num_clusters': optimal_k,
        'clustering_method': best_method,
        'embedding_model': 'all-mpnet-base-v2',
        'dimensionality_reduction': 'PCA-100',
        'timestamp': datetime.now().isoformat(),
        'device_used': device
    },
    'clusters': cluster_info,
    'relationships': relationships[:10],  # Top 10 relationships
    'outliers': sorted(outliers, key=lambda x: x['distance_from_centroid'], reverse=True)[:10],
    'quality_metrics': {
        'silhouette_score': float(final_silhouette),
        'davies_bouldin': float(final_db),
        'calinski_harabasz': float(final_ch)
    }
}

# Save results
print("\n10. Saving results...")
output_path = 'output-multi-agent-v2/multimodal_rl_clusters.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Results saved to: {output_path}")

# Generate comprehensive markdown report
print("\n11. Generating detailed report...")
report = []
report.append("# Multimodal RL Papers Clustering Report\n")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

report.append("\n## Executive Summary\n")
report.append(f"- **Total Papers**: {len(papers)}\n")
report.append(f"- **Number of Clusters**: {optimal_k}\n")
report.append(f"- **Clustering Method**: {best_method}\n")
report.append(f"- **Embedding Model**: all-mpnet-base-v2\n")
report.append(f"- **Overall Quality (Silhouette)**: {final_silhouette:.4f}\n")
report.append(f"- **Outliers Detected**: {len(outliers)}\n")

# Quality interpretation
report.append("\n### Quality Assessment\n")
if final_silhouette > 0.5:
    quality = "Excellent"
    emoji = "🟢"
elif final_silhouette > 0.3:
    quality = "Good"
    emoji = "🟡"
elif final_silhouette > 0.1:
    quality = "Moderate"
    emoji = "🟠"
else:
    quality = "Poor - Consider manual refinement"
    emoji = "🔴"

report.append(f"{emoji} **Clustering Quality: {quality}**\n\n")
report.append(f"- Silhouette Score: {final_silhouette:.4f} (range: -1 to 1, higher is better)\n")
report.append(f"- Davies-Bouldin Score: {final_db:.4f} (lower is better)\n")
report.append(f"- Calinski-Harabasz Score: {final_ch:.4f} (higher is better)\n")

# Detailed cluster descriptions
report.append("\n## Detailed Cluster Analysis\n")

for i, cluster in enumerate(cluster_info):
    report.append(f"\n### Cluster {i+1}: {cluster['name']}\n")
    report.append(f"**Size**: {cluster['size']} papers ({cluster['size']/len(papers)*100:.1f}% of total)\n\n")
    report.append(f"**Description**: {cluster['description']}\n\n")
    report.append(f"**Temporal Coverage**: {cluster['year_range'][0]} - {cluster['year_range'][1]}\n\n")
    report.append(f"**Citation Statistics**:\n")
    report.append(f"- Average: {cluster['avg_citations']:.1f} citations\n")
    report.append(f"- Median: {cluster['median_citations']:.1f} citations\n\n")
    
    if cluster['key_terms']:
        report.append(f"**Key Research Terms**: {', '.join(cluster['key_terms'][:6])}\n\n")
    
    report.append("**Most Representative Papers**:\n\n")
    for j, paper in enumerate(cluster['representative_papers'][:3], 1):
        report.append(f"{j}. **{paper['title']}**\n")
        report.append(f"   - Year: {paper['year']}, Citations: {paper['citations']}\n")
        report.append(f"   - Venue: {paper['venue']}\n")
        report.append(f"   - Cluster Relevance: {paper['relevance_to_cluster']:.2%}\n\n")

# Cluster balance analysis
report.append("\n## Cluster Balance and Distribution\n")
sizes = [c['size'] for c in cluster_info]
report.append(f"- **Largest cluster**: {max(sizes)} papers ({max(sizes)/len(papers)*100:.1f}%)\n")
report.append(f"- **Smallest cluster**: {min(sizes)} papers ({min(sizes)/len(papers)*100:.1f}%)\n")
report.append(f"- **Average cluster size**: {np.mean(sizes):.1f} papers\n")
report.append(f"- **Size standard deviation**: {np.std(sizes):.1f}\n\n")

if max(sizes) > len(papers) * 0.4:
    report.append("⚠️ **Warning**: Largest cluster contains >40% of papers. ")
    report.append("This might indicate that the topics are too broad or that more clusters are needed.\n\n")
elif np.std(sizes) / np.mean(sizes) > 0.5:
    report.append("⚠️ **Note**: High variation in cluster sizes detected. ")
    report.append("Some topics might be over- or under-represented.\n\n")
else:
    report.append("✅ **Good cluster balance**: Clusters are relatively well-balanced in size.\n\n")

# Inter-cluster relationships
if relationships:
    report.append("\n## Inter-Cluster Relationships\n")
    report.append("Strong relationships between clusters indicate overlapping research areas:\n\n")
    
    for rel in relationships[:7]:
        report.append(f"- **{rel['cluster_1_name']}** ↔ **{rel['cluster_2_name']}**\n")
        report.append(f"  - Relationship: {rel['relationship'].replace('_', ' ').title()}\n")
        report.append(f"  - Similarity: {rel['strength']:.3f}\n\n")

# Outliers section
if outliers:
    report.append("\n## Notable Outliers\n")
    report.append("Papers that don't fit well into any cluster (potential emerging topics):\n\n")
    
    for i, outlier in enumerate(outliers[:5], 1):
        report.append(f"{i}. **{outlier['paper_title']}**\n")
        report.append(f"   - Assigned to: {outlier['cluster_name']}\n")
        report.append(f"   - Reason: {outlier['reason']}\n\n")

# Research trends
report.append("\n## Research Trends and Insights\n\n")

# Find fastest growing clusters
growth_clusters = []
for cluster in cluster_info:
    if cluster['year_range'][0] and cluster['year_range'][1]:
        recent_papers = sum(1 for idx in cluster['paper_indices'] 
                          if papers[idx].get('year', 0) >= 2023)
        growth_rate = recent_papers / cluster['size']
        if growth_rate > 0.5:
            growth_clusters.append((cluster['name'], growth_rate))

if growth_clusters:
    growth_clusters.sort(key=lambda x: x[1], reverse=True)
    report.append("### Emerging Research Areas (>50% papers from 2023+)\n")
    for name, rate in growth_clusters[:3]:
        report.append(f"- **{name}**: {rate:.1%} recent papers\n")
    report.append("\n")

# High-impact clusters
impact_clusters = sorted(cluster_info, key=lambda x: x['avg_citations'], reverse=True)[:3]
report.append("### High-Impact Research Areas (by average citations)\n")
for cluster in impact_clusters:
    report.append(f"- **{cluster['name']}**: {cluster['avg_citations']:.1f} avg citations\n")

# Recommendations
report.append("\n## Recommendations\n\n")
if final_silhouette < 0.1:
    report.append("1. **Improve Clustering**: Consider using different embedding models or clustering algorithms\n")
    report.append("2. **Manual Refinement**: Review cluster assignments for outliers and edge cases\n")
elif final_silhouette < 0.3:
    report.append("1. **Refine Clusters**: Some clusters might benefit from subdivision or merging\n")
    report.append("2. **Feature Engineering**: Consider adding additional features beyond text embeddings\n")
else:
    report.append("1. **Cluster Quality**: The clustering quality is satisfactory for survey structure\n")
    report.append("2. **Next Steps**: Proceed with detailed analysis of each cluster for the survey\n")

report.append("\n---\n")
report.append(f"*Report generated using {device.upper()} acceleration*\n")

# Save report
report_path = 'output-multi-agent-v2/multimodal_rl_cluster_report.md'
with open(report_path, 'w') as f:
    f.write(''.join(report))

print(f"Report saved to: {report_path}")

print("\n" + "="*60)
print("IMPROVED CLUSTERING COMPLETE!")
print("="*60)
print(f"✅ Created {optimal_k} distinct clusters from {len(papers)} papers")
print(f"✅ Clustering quality: {quality}")
print(f"✅ Results saved to: {output_path}")
print(f"✅ Detailed report saved to: {report_path}")
print("\nCluster Summary:")
for i, cluster in enumerate(cluster_info[:optimal_k], 1):
    print(f"  {i}. {cluster['name']} ({cluster['size']} papers)")