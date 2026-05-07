#!/usr/bin/env python3
"""Build FAISS embeddings for the PASA database"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from pathlib import Path
from tinydb import TinyDB

print("Building embeddings for PASA database...")
print("-" * 50)

# Load the database
db_path = Path("./database_pasa")
db = TinyDB(str(db_path / "arxiv_paper_db.json"))
table = db.table('cs_paper_info')

# Get all papers
papers = table.all()
print(f"Loaded {len(papers)} papers from database")

# Initialize embedding model (Force CPU due to B200 compatibility)
print("Loading embedding model...")
device = 'cpu'  # Force CPU due to B200 CUDA compatibility issues
print(f"Using device: {device}")

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
model.to(torch.device(device))

# Extract titles and abstracts
titles = [p['title'] for p in papers]
abstracts = [p['abs'] for p in papers]
ids = [p['id'] for p in papers]

print(f"Processing {len(papers)} papers")

# Generate embeddings in batches
batch_size = 100
print("\nGenerating title embeddings...")
title_texts = ['search_document: ' + t for t in titles]
title_embeddings = []
for i in tqdm(range(0, len(title_texts), batch_size)):
    batch = title_texts[i:i+batch_size]
    embeddings = model.encode(batch, show_progress_bar=False)
    title_embeddings.append(embeddings)
title_embeddings = np.vstack(title_embeddings)

print("\nGenerating abstract embeddings...")
abs_texts = ['search_document: ' + a for a in abstracts]
abs_embeddings = []
for i in tqdm(range(0, len(abs_texts), batch_size)):
    batch = abs_texts[i:i+batch_size]
    embeddings = model.encode(batch, show_progress_bar=False)
    abs_embeddings.append(embeddings)
abs_embeddings = np.vstack(abs_embeddings)

# Create FAISS indices
print("\nCreating FAISS indices...")
dim = title_embeddings.shape[1]

# Title index
title_index = faiss.IndexFlatL2(dim)
title_index.add(title_embeddings.astype('float32'))

# Abstract index
abs_index = faiss.IndexFlatL2(dim)
abs_index.add(abs_embeddings.astype('float32'))

# Save indices
print("\nSaving indices...")
faiss.write_index(title_index, str(db_path / "faiss_paper_title_embeddings.bin"))
faiss.write_index(abs_index, str(db_path / "faiss_paper_abs_embeddings.bin"))

# Create ID mappings
id_to_index = {id: i for i, id in enumerate(ids)}
with open(db_path / "arxivid_to_index_abs.json", 'w') as f:
    json.dump(id_to_index, f)

print("\n✓ Embeddings built successfully!")
print(f"  Papers processed: {len(papers)}")
print(f"  Title embeddings: {db_path}/faiss_paper_title_embeddings.bin")
print(f"  Abstract embeddings: {db_path}/faiss_paper_abs_embeddings.bin")
print(f"  ID mappings: {db_path}/arxivid_to_index_abs.json")
print("\nDatabase is ready to use with AutoSurvey!")
print("Run: python run_local_survey.py --db-path ./database_pasa")