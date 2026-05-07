#!/usr/bin/env python3
"""Build FAISS embeddings for chunked PASA database"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from pathlib import Path

print("Building embeddings for chunked PASA database...")
print("-" * 50)

# Load the database metadata
db_path = Path('./database_pasa_full_chunked')
chunks_dir = db_path / 'chunks'

with open(db_path / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Total papers: {metadata['total_papers']}")
print(f"Number of chunks: {metadata['num_chunks']}")

# Initialize embedding model (Force CPU due to B200 compatibility)
device = 'cpu'
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
model.to(torch.device(device))
print(f"Using device: {device}")

# Process embeddings parameters
batch_size = 100  # Process 100 papers at a time for memory efficiency
dim = 768  # nomic-embed dimension

# Create empty indices
title_index = faiss.IndexFlatL2(dim)
abs_index = faiss.IndexFlatL2(dim)

# Store all paper IDs in order
all_ids = []

# Process each chunk file
for chunk_idx in tqdm(range(metadata['num_chunks']), desc="Processing chunks"):
    chunk_file = chunks_dir / f"chunk_{chunk_idx:04d}.json"
    
    # Load chunk
    with open(chunk_file, 'r') as f:
        papers = json.load(f)
    
    # Process papers in batches within the chunk
    for i in tqdm(range(0, len(papers), batch_size), 
                  desc=f"  Chunk {chunk_idx}", leave=False):
        batch = papers[i:i+batch_size]
        
        # Extract data
        batch_ids = [p['id'] for p in batch]
        batch_titles = ['search_document: ' + p['title'] for p in batch]
        batch_abs = ['search_document: ' + p['abs'] for p in batch]
        
        # Generate embeddings
        title_emb = model.encode(batch_titles, show_progress_bar=False)
        abs_emb = model.encode(batch_abs, show_progress_bar=False)
        
        # Add to indices
        title_index.add(title_emb.astype('float32'))
        abs_index.add(abs_emb.astype('float32'))
        
        all_ids.extend(batch_ids)

print(f"\nTotal embeddings created: {len(all_ids)}")

# Save indices
print("Saving FAISS indices...")
faiss.write_index(title_index, str(db_path / 'faiss_paper_title_embeddings.bin'))
faiss.write_index(abs_index, str(db_path / 'faiss_paper_abs_embeddings.bin'))

# Create ID mappings
print("Creating ID mappings...")
id_to_index = {paper_id: i for i, paper_id in enumerate(all_ids)}
with open(db_path / 'arxivid_to_index_abs.json', 'w') as f:
    json.dump(id_to_index, f)

# Also save the ordered list of IDs for quick index-to-id lookup
with open(db_path / 'index_to_arxivid.json', 'w') as f:
    json.dump(all_ids, f)

print(f"\n✓ Embeddings built for {len(all_ids)} papers")
print(f"  Title embeddings: {db_path / 'faiss_paper_title_embeddings.bin'}")
print(f"  Abstract embeddings: {db_path / 'faiss_paper_abs_embeddings.bin'}")
print(f"  ID mappings: {db_path / 'arxivid_to_index_abs.json'}")
print(f"\nDatabase ready at: {db_path}")