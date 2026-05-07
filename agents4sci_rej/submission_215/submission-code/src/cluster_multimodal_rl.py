#!/usr/bin/env python3
"""
Topic Mining & Clustering for Multimodal RL Papers
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict, Counter
from datetime import datetime
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

papers = data['papers']  # It's already a list
print(f"Loaded {len(papers)} papers")

# Prepare texts for embedding
print("\n2. Preparing texts for embedding...")
def prepare_text(paper):
    """Combine title and abstract for better embeddings"""
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    return f"{title}. {abstract}"

texts = [prepare_text(p) for p in papers]

# Generate embeddings using sentence-transformers
print("\n3. Generating embeddings with sentence-transformers...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Process in batches to avoid memory issues
batch_size = 32
embeddings = model.encode(
    texts,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"Embeddings shape: {embeddings.shape}")

# Normalize embeddings
embeddings = StandardScaler().fit_transform(embeddings)

# Find optimal number of clusters
print("\n4. Finding optimal number of clusters...")
k_range = range(8, 11)  # Test K between 8-10 as requested
silhouette_scores = []
db_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    sil_score = silhouette_score(embeddings, labels)
    db_score = davies_bouldin_score(embeddings, labels)
    
    silhouette_scores.append(sil_score)
    db_scores.append(db_score)
    
    print(f"K={k}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")

# Select optimal K based on silhouette score
optimal_idx = np.argmax(silhouette_scores)
optimal_k = list(k_range)[optimal_idx]
print(f"\nOptimal K={optimal_k} with Silhouette score={silhouette_scores[optimal_idx]:.4f}")

# Perform final clustering with optimal K
print(f"\n5. Performing final clustering with K={optimal_k}...")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = final_kmeans.fit_predict(embeddings)

# Calculate final metrics
final_silhouette = silhouette_score(embeddings, cluster_labels)
final_db = davies_bouldin_score(embeddings, cluster_labels)
final_ch = calinski_harabasz_score(embeddings, cluster_labels)

print(f"Final Silhouette Score: {final_silhouette:.4f}")
print(f"Final Davies-Bouldin Score: {final_db:.4f}")
print(f"Final Calinski-Harabasz Score: {final_ch:.4f}")

# Analyze clusters
print("\n6. Analyzing clusters...")
clusters = defaultdict(list)

for idx, label in enumerate(cluster_labels):
    clusters[label].append(idx)

# Generate cluster names using TF-IDF on abstracts
print("\n7. Generating cluster names...")
def generate_cluster_name(cluster_papers, all_papers):
    """Generate descriptive name for cluster using TF-IDF"""
    # Get abstracts for this cluster
    texts = [p.get('abstract', '') for p in cluster_papers]
    
    if not texts or all(not t for t in texts):
        return "Uncategorized Papers"
    
    # Use TF-IDF to find key terms
    try:
        vectorizer = TfidfVectorizer(
            max_features=20,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=0.2  # Term must appear in at least 20% of cluster docs
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores
        scores = tfidf_matrix.mean(axis=0).A1
        top_indices = scores.argsort()[-10:][::-1]
        
        # Get top terms
        top_terms = [terms[i] for i in top_indices]
        
        # Manually curate cluster names based on top terms and paper analysis
        # We'll also look at titles for better context
        titles = [p.get('title', '').lower() for p in cluster_papers]
        title_text = ' '.join(titles)
        
        # Common patterns in multimodal RL
        if any(term in ' '.join(top_terms).lower() for term in ['robot', 'embodied', 'navigation', 'manipulation']):
            return "Embodied AI and Robotics with RL"
        elif any(term in ' '.join(top_terms).lower() for term in ['rlhf', 'human feedback', 'preference', 'alignment']):
            return "RLHF for Multimodal Model Alignment"
        elif any(term in ' '.join(top_terms).lower() for term in ['vision language', 'vln', 'instruction', 'following']):
            return "Vision-Language Instruction Following"
        elif any(term in ' '.join(top_terms).lower() for term in ['reward', 'reward model', 'reward learning']):
            return "Cross-Modal Reward Modeling"
        elif any(term in ' '.join(top_terms).lower() for term in ['reasoning', 'visual reasoning', 'chain thought']):
            return "Visual Reasoning with Reinforcement Learning"
        elif any(term in ' '.join(top_terms).lower() for term in ['policy', 'policy learning', 'policy optimization']):
            return "Multimodal Policy Learning"
        elif any(term in ' '.join(top_terms).lower() for term in ['game', 'gaming', 'video game']):
            return "RL in Games and Interactive Environments"
        elif any(term in ' '.join(top_terms).lower() for term in ['medical', 'clinical', 'healthcare', 'diagnosis']):
            return "Medical and Healthcare Applications"
        elif any(term in ' '.join(top_terms).lower() for term in ['diffusion', 'generation', 'text image']):
            return "Generative Models with RL Optimization"
        else:
            # Use top 2-3 most distinctive terms
            key_terms = []
            for term in top_terms[:3]:
                if len(term.split()) <= 3 and term not in ['model', 'learning', 'using', 'based']:
                    key_terms.append(term.title())
            
            if key_terms:
                return ' '.join(key_terms[:2])
            else:
                return f"Multimodal RL Cluster {len(cluster_papers)} papers"
                
    except Exception as e:
        print(f"Error generating name: {e}")
        return f"Cluster with {len(cluster_papers)} papers"

# Build cluster information
cluster_info = []

for cluster_id in sorted(clusters.keys()):
    paper_indices = clusters[cluster_id]
    cluster_papers = [papers[i] for i in paper_indices]
    
    # Generate cluster name
    cluster_name = generate_cluster_name(cluster_papers, papers)
    
    # Calculate statistics
    years = [p.get('year', 0) for p in cluster_papers if p.get('year')]
    citations = [p.get('citations', 0) for p in cluster_papers if p.get('citations') is not None]
    
    avg_year = np.mean(years) if years else 0
    avg_citations = np.mean(citations) if citations else 0
    
    # Find representative papers (closest to centroid)
    cluster_embeddings = embeddings[paper_indices]
    centroid = cluster_embeddings.mean(axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    representative_indices = np.argsort(distances)[:5]  # Top 5 most representative
    
    representative_papers = []
    for idx in representative_indices:
        paper = cluster_papers[idx]
        representative_papers.append({
            'title': paper.get('title', 'Unknown'),
            'year': paper.get('year', 'N/A'),
            'citations': paper.get('citations', 0),
            'relevance_to_cluster': float(1.0 - distances[idx] / distances.max())
        })
    
    # Extract key terms using TF-IDF
    cluster_texts = [p.get('abstract', '') for p in cluster_papers]
    if cluster_texts and any(cluster_texts):
        try:
            vec = TfidfVectorizer(max_features=10, stop_words='english', ngram_range=(1, 2))
            tfidf = vec.fit_transform(cluster_texts)
            key_terms = vec.get_feature_names_out().tolist()
        except:
            key_terms = []
    else:
        key_terms = []
    
    cluster_info.append({
        'id': int(cluster_id),
        'name': cluster_name,
        'description': f"Cluster focusing on {cluster_name.lower()} with {len(paper_indices)} papers",
        'size': len(paper_indices),
        'paper_indices': paper_indices,
        'key_terms': key_terms[:5],  # Top 5 key terms
        'representative_papers': representative_papers,
        'avg_year': float(avg_year),
        'avg_citations': float(avg_citations)
    })
    
    print(f"\nCluster {cluster_id}: {cluster_name}")
    print(f"  Size: {len(paper_indices)} papers")
    print(f"  Avg Year: {avg_year:.1f}")
    print(f"  Avg Citations: {avg_citations:.1f}")

# Sort clusters by size
cluster_info.sort(key=lambda x: x['size'], reverse=True)

# Identify outliers (papers far from their cluster centers)
print("\n8. Identifying outliers...")
outliers = []
for cluster_id in clusters.keys():
    paper_indices = clusters[cluster_id]
    cluster_embeddings = embeddings[paper_indices]
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
                'distance_from_centroid': float(distances[idx]),
                'reason': 'Low similarity to cluster centroid'
            })

print(f"Found {len(outliers)} outliers")

# Analyze cluster relationships
print("\n9. Analyzing cluster relationships...")
relationships = []
cluster_centroids = []

for cluster_id in sorted(clusters.keys()):
    paper_indices = clusters[cluster_id]
    cluster_embeddings = embeddings[paper_indices]
    centroid = cluster_embeddings.mean(axis=0)
    cluster_centroids.append(centroid)

# Calculate pairwise similarities between clusters
for i in range(len(cluster_centroids)):
    for j in range(i+1, len(cluster_centroids)):
        similarity = np.dot(cluster_centroids[i], cluster_centroids[j])
        if similarity > 0.3:  # Threshold for significant relationship
            relationships.append({
                'cluster_1': i,
                'cluster_2': j,
                'relationship': 'related' if similarity > 0.5 else 'weakly_related',
                'strength': float(similarity)
            })

print(f"Found {len(relationships)} cluster relationships")

# Create output JSON
output = {
    'metadata': {
        'num_papers': len(papers),
        'num_clusters': optimal_k,
        'clustering_method': 'kmeans',
        'embedding_model': 'all-MiniLM-L6-v2',
        'timestamp': datetime.now().isoformat(),
        'device_used': device
    },
    'clusters': cluster_info,
    'relationships': relationships,
    'outliers': outliers[:10],  # Top 10 outliers
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

print(f"\nResults saved to: {output_path}")

# Generate markdown report
print("\n11. Generating report...")
report = []
report.append("# Multimodal RL Papers Clustering Report\n")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append("\n## Overview\n")
report.append(f"- **Papers analyzed**: {len(papers)}\n")
report.append(f"- **Clusters created**: {optimal_k}\n")
report.append(f"- **Outliers identified**: {len(outliers)}\n")
report.append(f"- **Quality score (silhouette)**: {final_silhouette:.4f}\n")
report.append(f"- **Davies-Bouldin score**: {final_db:.4f} (lower is better)\n")
report.append(f"- **Device used**: {device}\n")

report.append("\n## Cluster Descriptions\n")

for cluster in cluster_info:
    report.append(f"\n### Cluster {cluster['id']}: {cluster['name']} ({cluster['size']} papers)\n")
    report.append(f"**Description**: {cluster['description']}\n")
    report.append(f"**Average Year**: {cluster['avg_year']:.1f}\n")
    report.append(f"**Average Citations**: {cluster['avg_citations']:.1f}\n")
    
    if cluster['key_terms']:
        report.append(f"**Key Terms**: {', '.join(cluster['key_terms'][:5])}\n")
    
    report.append("\n**Most Representative Papers**:\n")
    for i, paper in enumerate(cluster['representative_papers'][:3], 1):
        report.append(f"{i}. {paper['title']} ({paper['year']}, {paper['citations']} citations)\n")

# Cluster balance analysis
report.append("\n## Cluster Balance Analysis\n")
sizes = [c['size'] for c in cluster_info]
report.append(f"- **Largest cluster**: {max(sizes)} papers ({max(sizes)/len(papers)*100:.1f}%)\n")
report.append(f"- **Smallest cluster**: {min(sizes)} papers ({min(sizes)/len(papers)*100:.1f}%)\n")
report.append(f"- **Size std deviation**: {np.std(sizes):.1f}\n")

if max(sizes) > len(papers) * 0.4:
    report.append("\n⚠️ Warning: Largest cluster contains >40% of papers, consider re-clustering\n")

# Relationships
if relationships:
    report.append("\n## Cluster Relationships\n")
    for rel in relationships[:5]:  # Top 5 relationships
        c1_name = cluster_info[rel['cluster_1']]['name']
        c2_name = cluster_info[rel['cluster_2']]['name']
        report.append(f"- **{c1_name}** & **{c2_name}**: {rel['relationship']} (strength: {rel['strength']:.3f})\n")

# Quality analysis
report.append("\n## Quality Analysis\n")
if final_silhouette > 0.5:
    report.append("✅ **Excellent clustering quality** (silhouette > 0.5)\n")
elif final_silhouette > 0.3:
    report.append("✅ **Good clustering quality** (silhouette > 0.3)\n")
else:
    report.append("⚠️ **Moderate clustering quality** (silhouette < 0.3)\n")

if final_db < 2.0:
    report.append("✅ **Excellent cluster separation** (Davies-Bouldin < 2.0)\n")
elif final_db < 3.0:
    report.append("✅ **Good cluster separation** (Davies-Bouldin < 3.0)\n")
else:
    report.append("⚠️ **Moderate cluster separation** (Davies-Bouldin > 3.0)\n")

# Well-separated clusters
well_separated = []
overlapping = []
for i, cluster in enumerate(cluster_info):
    cluster_id = cluster['id']
    # Check if this cluster has strong relationships with others
    strong_rels = [r for r in relationships if (r['cluster_1'] == cluster_id or r['cluster_2'] == cluster_id) and r['strength'] > 0.5]
    if not strong_rels:
        well_separated.append(cluster['name'])
    else:
        overlapping.append(cluster['name'])

if well_separated:
    report.append(f"\n**Well-separated clusters**: {', '.join(well_separated[:3])}\n")
if overlapping:
    report.append(f"**Overlapping clusters**: {', '.join(overlapping[:3])}\n")

# Save report
report_path = 'output-multi-agent-v2/multimodal_rl_cluster_report.md'
with open(report_path, 'w') as f:
    f.write(''.join(report))

print(f"Report saved to: {report_path}")

print("\n" + "="*60)
print("CLUSTERING COMPLETE!")
print("="*60)
print(f"✅ Created {optimal_k} clusters from {len(papers)} papers")
print(f"✅ Silhouette score: {final_silhouette:.4f}")
print(f"✅ Results saved to: {output_path}")
print(f"✅ Report saved to: {report_path}")