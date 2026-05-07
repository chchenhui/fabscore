#!/usr/bin/env python3
import json
import sys

# Read the papers and clusters
with open('/path/to/project/output/llm_rlhf_alignment/output/papers.json', 'r') as f:
    papers_data = json.load(f)

with open('/path/to/project/output/llm_rlhf_alignment/output/clusters.json', 'r') as f:
    clusters_data = json.load(f)

# Create a dictionary of papers by ID for quick lookup
papers_dict = {paper['id']: paper for paper in papers_data['papers']}

# Print statistics
print(f"Total papers: {len(papers_data['papers'])}")
print(f"Total clusters: {len(clusters_data['clusters'])}")
print()

# Print cluster information with sample papers
for cluster in clusters_data['clusters']:
    print(f"\nCluster {cluster['id']}: {cluster['name']}")
    print(f"Description: {cluster['description']}")
    print(f"Paper count: {cluster['paper_count']}")
    print(f"Key terms: {', '.join(cluster['key_terms'][:5])}")
    
    # Print first 3 papers in this cluster as examples
    print("Sample papers:")
    for i, paper_id in enumerate(cluster['papers'][:3]):
        if paper_id in papers_dict:
            paper = papers_dict[paper_id]
            print(f"  - {paper['title']} ({paper['year']})")
            if paper['authors']:
                print(f"    Authors: {', '.join(paper['authors'][:3])}")
    print()

# Count unique authors and years
all_authors = set()
years = []
for paper in papers_data['papers']:
    if paper['authors']:
        all_authors.update(paper['authors'])
    if paper['year']:
        years.append(paper['year'])

print(f"\nTotal unique authors: {len(all_authors)}")
if years:
    print(f"Year range: {min(years)} - {max(years)}")
    print(f"Average year: {sum(years)/len(years):.1f}")