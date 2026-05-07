#!/usr/bin/env python3
"""Convert PASA dataset to AutoSurvey format"""

import json
import zipfile
import os
from pathlib import Path
from tinydb import TinyDB
from tqdm import tqdm

print("Converting PASA dataset to AutoSurvey format...")
print("-" * 50)

# Load id2paper mapping
with open('id2paper.json', 'r') as f:
    id2paper = json.load(f)

print(f"Loaded {len(id2paper)} paper ID->title mappings")

# Prepare database directory
db_dir = Path("./database_pasa")
db_dir.mkdir(exist_ok=True)

# Extract paper abstracts from zip file
papers_data = []
errors = 0

print("\nExtracting paper content from zip file...")
with zipfile.ZipFile('cs_paper_2nd.zip', 'r') as zf:
    filenames = zf.namelist()
    print(f"Found {len(filenames)} files in zip")
    
    for i, filename in enumerate(tqdm(filenames[:10000])):  # Process first 10k for testing
        try:
            # Read paper content
            with zf.open(filename) as f:
                content = f.read().decode('utf-8', errors='ignore')
            
            # Try to match filename to paper ID
            # The filename seems to be a sanitized version of the title
            # Let's find the matching ID by checking if title starts match
            paper_id = None
            paper_title = None
            
            # Search for matching title in id2paper
            for pid, title in id2paper.items():
                # Create a sanitized version of title for comparison
                sanitized_title = ''.join(c.lower() for c in title if c.isalnum())[:50]
                if sanitized_title.startswith(filename[:20]):
                    paper_id = pid
                    paper_title = title
                    break
            
            if not paper_id:
                # Use filename as fallback ID
                paper_id = f"pasa_{i:06d}"
                paper_title = filename.replace('_', ' ').title()
            
            # Create paper entry
            paper_entry = {
                "id": paper_id,
                "title": paper_title,
                "abs": content[:2000],  # Use first 2000 chars as abstract
                "authors": ["Unknown"],
                "categories": ["cs.CL"],
                "year": 2024,
                "date": "2024-01-01"
            }
            papers_data.append(paper_entry)
            
        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"Error processing {filename}: {e}")

print(f"\nProcessed {len(papers_data)} papers successfully")
print(f"Errors: {errors}")

# Save as TinyDB format for AutoSurvey
db_file = db_dir / "arxiv_paper_db.json"
db = TinyDB(str(db_file))
table = db.table('cs_paper_info')

print("\nSaving to TinyDB format...")
for paper in tqdm(papers_data):
    table.insert(paper)

print(f"\n✓ Database created with {len(papers_data)} papers")
print(f"  Saved to: {db_file}")
print("\nNext steps:")
print("  1. Run: python build_embeddings_pasa.py")
print("  2. Use --db-path ./database_pasa when running AutoSurvey")