#!/usr/bin/env python3
"""
Cluster LLM agents papers into research topics
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict
from datetime import datetime
import sys

# Add project root to path
sys.path.append('/path/to/project')

from scripts.core.embeddings import EmbeddingGenerator
from scripts.core.clustering import PaperClusterer
from scripts.core.utils import load_config, save_json, load_json


def analyze_paper_themes(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze paper themes and extract key research directions
    """
    # Extract common terms from titles and abstracts
    all_text = []
    years = []
    citations = []
    
    for paper in papers:
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        all_text.append(f"{title} {abstract}")
        years.append(paper.get('year', 2024))
        citations.append(paper.get('citations', 0))
    
    # Common themes in LLM agents research
    theme_keywords = {
        'Multi-Agent Systems': ['multi-agent', 'multi agent', 'agent collaboration', 'cooperative', 'collective', 'swarm', 'distributed agents'],
        'Tool Use & Function Calling': ['tool', 'function calling', 'api', 'plugin', 'external tools', 'tool learning', 'tool use'],
        'Reasoning & Planning': ['reasoning', 'planning', 'chain of thought', 'cot', 'react', 'reflection', 'self-reflection', 'deliberation'],
        'Code Generation & Software Engineering': ['code generation', 'software', 'programming', 'coding', 'development', 'debugging', 'software engineering'],
        'Retrieval & RAG': ['retrieval', 'rag', 'retrieval augmented', 'information retrieval', 'knowledge retrieval', 'document retrieval'],
        'Evaluation & Benchmarks': ['benchmark', 'evaluation', 'metrics', 'performance', 'assessment', 'test suite', 'dataset'],
        'Memory & State Management': ['memory', 'state management', 'context', 'long-term memory', 'working memory', 'episodic memory'],
        'Human-AI Interaction': ['human', 'interaction', 'collaboration', 'human-ai', 'user study', 'interface', 'usability'],
        'Agent Frameworks & Architecture': ['framework', 'architecture', 'platform', 'infrastructure', 'system design', 'agent design'],
        'Safety & Alignment': ['safety', 'alignment', 'trustworthy', 'reliable', 'robustness', 'adversarial', 'security'],
        'Domain-Specific Applications': ['medical', 'healthcare', 'finance', 'education', 'scientific', 'legal', 'business'],
        'Embodied & Robotic Agents': ['embodied', 'robot', 'physical', 'environment', 'simulation', 'virtual world', 'game'],
        'Autonomous Agents': ['autonomous', 'self-directed', 'goal-oriented', 'self-improving', 'adaptive', 'learning agents'],
        'Communication & Dialog': ['dialog', 'dialogue', 'conversation', 'communication', 'chat', 'interactive', 'question answering']
    }
    
    # Count theme occurrences
    theme_counts = defaultdict(int)
    paper_themes = defaultdict(list)
    
    for i, text in enumerate(all_text):
        for theme, keywords in theme_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    theme_counts[theme] += 1
                    paper_themes[i].append(theme)
                    break  # Count each theme once per paper
    
    return {
        'total_papers': len(papers),
        'year_range': f"{min(years)}-{max(years)}",
        'avg_citations': np.mean(citations),
        'median_citations': np.median(citations),
        'identified_themes': dict(theme_counts),
        'paper_themes': dict(paper_themes)
    }


def assign_detailed_cluster_info(clusters: List[Dict[str, Any]], 
                                papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign detailed information to each cluster
    """
    # Define research theme names based on common topics in LLM agents
    theme_templates = [
        "Multi-Agent Collaboration and Coordination",
        "Tool Learning and Function Calling",
        "Reasoning and Planning Strategies", 
        "Code Generation and Software Development",
        "Retrieval-Augmented Generation Systems",
        "Agent Evaluation and Benchmarking",
        "Memory and Context Management",
        "Human-AI Collaborative Systems",
        "Agent Frameworks and Architectures",
        "Safety and Alignment in Autonomous Agents",
        "Domain-Specific Agent Applications",
        "Embodied and Interactive Agents",
        "Autonomous Learning Agents",
        "Conversational and Dialog Agents",
        "Knowledge-Grounded Agents"
    ]
    
    for i, cluster in enumerate(clusters):
        # Get papers in this cluster
        cluster_papers = [papers[idx] for idx in cluster['paper_indices']]
        
        if not cluster_papers:
            continue
        
        # Extract years and citations
        years = [p.get('year', 2024) for p in cluster_papers]
        citations = [p.get('citations', 0) for p in cluster_papers]
        
        # Extract key terms from titles and abstracts
        texts = []
        titles = []
        for paper in cluster_papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            texts.append(f"{title} {abstract}")
            titles.append(title)
        
        # Identify dominant research theme
        theme_scores = {}
        theme_keywords = {
            'multi-agent': ['multi-agent', 'multi agent', 'collaborative', 'coordination', 'distributed'],
            'tool-use': ['tool', 'function', 'api', 'plugin', 'external'],
            'reasoning': ['reasoning', 'planning', 'chain of thought', 'cot', 'react', 'reflection'],
            'code-generation': ['code', 'programming', 'software', 'development', 'debugging'],
            'retrieval': ['retrieval', 'rag', 'augmented', 'knowledge', 'information'],
            'evaluation': ['benchmark', 'evaluation', 'metric', 'performance', 'test'],
            'memory': ['memory', 'context', 'state', 'history', 'episodic'],
            'human-ai': ['human', 'user', 'interaction', 'collaboration', 'interface'],
            'framework': ['framework', 'architecture', 'platform', 'system', 'infrastructure'],
            'safety': ['safety', 'alignment', 'trustworthy', 'robust', 'secure'],
            'domain': ['medical', 'healthcare', 'finance', 'education', 'scientific', 'legal'],
            'embodied': ['embodied', 'robot', 'physical', 'environment', 'simulation'],
            'autonomous': ['autonomous', 'self', 'adaptive', 'learning', 'goal'],
            'dialog': ['dialog', 'dialogue', 'conversation', 'chat', 'interactive']
        }
        
        # Score each theme based on keyword occurrence
        combined_text = ' '.join(texts).lower()
        for theme, keywords in theme_keywords.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            theme_scores[theme] = score
        
        # Select best matching theme
        best_theme = max(theme_scores.items(), key=lambda x: x[1])[0] if theme_scores else 'general'
        
        # Map to descriptive name
        theme_map = {
            'multi-agent': "Multi-Agent Collaboration and Coordination",
            'tool-use': "Tool Learning and Function Calling",
            'reasoning': "Reasoning and Planning Strategies",
            'code-generation': "Code Generation and Software Development",
            'retrieval': "Retrieval-Augmented Generation",
            'evaluation': "Agent Evaluation and Benchmarking",
            'memory': "Memory and Context Management",
            'human-ai': "Human-AI Collaborative Systems",
            'framework': "Agent Frameworks and Architectures",
            'safety': "Safety and Alignment",
            'domain': "Domain-Specific Applications",
            'embodied': "Embodied and Interactive Agents",
            'autonomous': "Autonomous Learning Agents",
            'dialog': "Conversational Dialog Systems",
            'general': f"LLM Agent Research Cluster {i+1}"
        }
        
        cluster_name = theme_map.get(best_theme, f"Research Cluster {i+1}")
        
        # Generate description based on papers
        if best_theme == 'multi-agent':
            description = "Research on multi-agent systems, coordination mechanisms, and collaborative problem-solving with multiple LLM agents"
        elif best_theme == 'tool-use':
            description = "Studies on enabling LLMs to use external tools, APIs, and functions for enhanced capabilities"
        elif best_theme == 'reasoning':
            description = "Advances in reasoning, planning, and decision-making strategies for LLM-based agents"
        elif best_theme == 'code-generation':
            description = "Applications of LLM agents in software engineering, code generation, and automated programming"
        elif best_theme == 'retrieval':
            description = "Integration of retrieval mechanisms and knowledge bases with LLM agents for grounded generation"
        elif best_theme == 'evaluation':
            description = "Benchmarks, metrics, and evaluation frameworks for assessing LLM agent performance"
        elif best_theme == 'memory':
            description = "Memory architectures and context management strategies for maintaining agent state"
        elif best_theme == 'human-ai':
            description = "Human-AI interaction, collaboration interfaces, and user studies with LLM agents"
        elif best_theme == 'framework':
            description = "Architectural designs, frameworks, and platforms for building LLM agent systems"
        elif best_theme == 'safety':
            description = "Safety, alignment, and robustness considerations for deploying LLM agents"
        elif best_theme == 'domain':
            description = "Domain-specific applications of LLM agents in specialized fields"
        elif best_theme == 'embodied':
            description = "Embodied agents, robotics integration, and interaction with physical/virtual environments"
        elif best_theme == 'autonomous':
            description = "Autonomous, self-directed agents with learning and adaptation capabilities"
        elif best_theme == 'dialog':
            description = "Conversational agents, dialog systems, and interactive communication"
        else:
            description = "Collection of papers exploring various aspects of LLM-based agent systems"
        
        # Extract most common terms for keywords
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            vectorizer = TfidfVectorizer(max_features=10, stop_words='english', ngram_range=(1,2))
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-5:][::-1]
            key_terms = [feature_names[i] for i in top_indices]
        except:
            key_terms = ['llm', 'agent', 'model', 'learning', 'system']
        
        # Update cluster information
        cluster.update({
            'name': cluster_name,
            'description': description,
            'paper_count': len(cluster_papers),
            'papers': [p.get('id', f'paper_{idx}') for idx, p in enumerate(cluster_papers)],
            'key_terms': key_terms,
            'avg_year': float(np.mean(years)),
            'year_range': f"{min(years)}-{max(years)}",
            'avg_citations': float(np.mean(citations)),
            'total_citations': sum(citations),
            'confidence': 0.85 + (0.15 * min(len(cluster_papers) / 20, 1.0))  # Higher confidence for larger clusters
        })
    
    return clusters


def identify_cluster_relationships(clusters: List[Dict[str, Any]], 
                                  embeddings: np.ndarray) -> List[Dict[str, Any]]:
    """
    Identify relationships between clusters
    """
    relationships = []
    
    if len(clusters) < 2:
        return relationships
    
    # Calculate cluster center similarities
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            cluster1 = clusters[i]
            cluster2 = clusters[j]
            
            # Calculate similarity based on cluster centers
            if cluster1.get('center') and cluster2.get('center'):
                center1 = np.array(cluster1['center'])
                center2 = np.array(cluster2['center'])
                similarity = float(np.dot(center1, center2))
                
                # Determine relationship type based on similarity and themes
                if similarity > 0.7:
                    rel_type = 'overlapping'
                elif similarity > 0.5:
                    rel_type = 'complementary'
                else:
                    rel_type = 'distinct'
                
                # Check for hierarchical relationships based on cluster names
                name1 = cluster1.get('name', '').lower()
                name2 = cluster2.get('name', '').lower()
                
                if 'framework' in name1 and 'application' in name2:
                    rel_type = 'hierarchical'
                elif 'evaluation' in name1 or 'benchmark' in name1:
                    rel_type = 'complementary'  # Evaluation relates to all
                
                relationships.append({
                    'cluster_1': cluster1['id'],
                    'cluster_2': cluster2['id'],
                    'relationship_type': rel_type,
                    'strength': similarity
                })
    
    return relationships


def main():
    """
    Main clustering pipeline for LLM agents papers
    """
    # Load configuration
    config = load_config()
    
    # Set paths
    input_path = Path('/path/to/project/output/llm_agents/data/papers/papers.json')
    output_dir = Path('/path/to/project/output/llm_agents/data/clusters')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load papers
    print("Loading papers...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    papers = data['papers']
    print(f"Loaded {len(papers)} papers")
    
    # Analyze paper themes
    print("\nAnalyzing paper themes...")
    theme_analysis = analyze_paper_themes(papers)
    print(f"Identified themes: {list(theme_analysis['identified_themes'].keys())}")
    
    # Initialize components
    print("\nInitializing embedding generator...")
    embedding_gen = EmbeddingGenerator(config)
    
    print("Initializing paper clusterer...")
    clusterer = PaperClusterer(config)
    
    # Generate embeddings
    print("\nGenerating paper embeddings...")
    embeddings = embedding_gen.generate_paper_embeddings(
        papers, 
        use_title=True, 
        use_abstract=True
    )
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Perform clustering
    print("\nPerforming clustering...")
    clustering_result = clusterer.perform_clustering(
        embeddings,
        n_clusters=None,  # Auto-optimize
        method='kmeans'
    )
    
    n_clusters = clustering_result['n_clusters']
    print(f"Found {n_clusters} clusters")
    print(f"Clustering metrics: {clustering_result['metrics']}")
    
    # Assign cluster names and details
    print("\nAssigning cluster names and descriptions...")
    clusters = clustering_result['clusters']
    clusters = clusterer.assign_cluster_names(clusters, papers)
    clusters = assign_detailed_cluster_info(clusters, papers)
    
    # Identify relationships
    print("\nIdentifying cluster relationships...")
    relationships = identify_cluster_relationships(clusters, embeddings)
    
    # Calculate clustering quality
    labels = clustering_result['labels']
    unclustered = sum(1 for l in labels if l == -1)
    avg_cluster_size = len(papers) / n_clusters if n_clusters > 0 else 0
    
    # Identify outliers
    outliers = []
    for idx in clustering_result.get('outliers', []):
        if idx < len(papers):
            outliers.append({
                'paper_id': papers[idx].get('id', f'paper_{idx}'),
                'reason': 'Statistical outlier - far from cluster center'
            })
    
    # Prepare final output
    output_data = {
        'clusters': clusters,
        'relationships': relationships,
        'clustering_metadata': {
            'total_papers': len(papers),
            'num_clusters': n_clusters,
            'unclustered_papers': unclustered,
            'avg_cluster_size': avg_cluster_size,
            'clustering_quality': clustering_result['metrics'].get('silhouette', 0.0),
            'method': clustering_result['method'],
            'timestamp': datetime.now().isoformat(),
            'theme_analysis': theme_analysis
        },
        'outliers': outliers
    }
    
    # Save results
    output_path = output_dir / 'clusters.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nSaved clustering results to {output_path}")
    
    # Print summary
    print("\n=== Clustering Summary ===")
    print(f"Total papers: {len(papers)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Average cluster size: {avg_cluster_size:.1f}")
    print(f"Clustering quality (silhouette): {clustering_result['metrics'].get('silhouette', 0):.3f}")
    print(f"Outliers identified: {len(outliers)}")
    
    print("\n=== Cluster Details ===")
    for cluster in sorted(clusters, key=lambda x: x['paper_count'], reverse=True):
        print(f"\nCluster {cluster['id']}: {cluster['name']}")
        print(f"  Description: {cluster['description']}")
        print(f"  Papers: {cluster['paper_count']}")
        print(f"  Avg year: {cluster['avg_year']:.1f}")
        print(f"  Avg citations: {cluster['avg_citations']:.1f}")
        print(f"  Key terms: {', '.join(cluster['key_terms'][:3])}")
    
    print("\n=== Top Cluster Relationships ===")
    for rel in sorted(relationships, key=lambda x: x['strength'], reverse=True)[:5]:
        c1_name = next(c['name'] for c in clusters if c['id'] == rel['cluster_1'])
        c2_name = next(c['name'] for c in clusters if c['id'] == rel['cluster_2'])
        print(f"{c1_name} <-> {c2_name}")
        print(f"  Type: {rel['relationship_type']}, Strength: {rel['strength']:.3f}")


if __name__ == "__main__":
    main()