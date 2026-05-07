"""
Chunked database adapter for AutoSurvey - handles large databases split into chunks
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

class ChunkedDatabase:
    """
    A database adapter that reads from chunked JSON files instead of TinyDB.
    Compatible with AutoSurvey's Database interface.
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.chunks_dir = self.db_path / 'chunks'
        
        # Load metadata
        with open(self.db_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load ID to index mapping for quick lookups
        with open(self.db_path / 'arxivid_to_index_abs.json', 'r') as f:
            self.id_to_index = json.load(f)
        
        # Cache for loaded chunks (keep last few chunks in memory)
        self.chunk_cache = {}
        self.max_cache_size = 5
        
        print(f"Loaded chunked database with {self.metadata['total_papers']} papers")
    
    def _load_chunk(self, chunk_idx: int) -> List[Dict[str, Any]]:
        """Load a chunk file, using cache if available"""
        if chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        
        chunk_file = self.chunks_dir / f"chunk_{chunk_idx:04d}.json"
        with open(chunk_file, 'r') as f:
            data = json.load(f)
        
        # Add to cache and manage cache size
        self.chunk_cache[chunk_idx] = data
        if len(self.chunk_cache) > self.max_cache_size:
            # Remove oldest chunk from cache
            oldest = min(self.chunk_cache.keys())
            del self.chunk_cache[oldest]
        
        return data
    
    def get_paper_by_id(self, paper_id: str) -> Dict[str, Any]:
        """Get a paper by its ID"""
        if paper_id not in self.id_to_index:
            return None
        
        # Calculate which chunk contains this paper
        global_idx = self.id_to_index[paper_id]
        chunk_idx = global_idx // self.metadata['chunk_size']
        local_idx = global_idx % self.metadata['chunk_size']
        
        # Load the chunk and get the paper
        chunk_data = self._load_chunk(chunk_idx)
        if local_idx < len(chunk_data):
            return chunk_data[local_idx]
        return None
    
    def get_papers_by_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple papers by their IDs"""
        papers = []
        for paper_id in paper_ids:
            paper = self.get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
        return papers
    
    def search(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Basic search functionality - for compatibility.
        Real search should be done through FAISS embeddings.
        """
        # This is a placeholder - AutoSurvey uses FAISS for actual search
        return []
    
    def all(self) -> List[Dict[str, Any]]:
        """
        Return all papers - WARNING: This loads entire database into memory.
        Should only be used for small operations or testing.
        """
        all_papers = []
        for chunk_idx in range(self.metadata['num_chunks']):
            chunk_data = self._load_chunk(chunk_idx)
            all_papers.extend(chunk_data)
        return all_papers
    
    def close(self):
        """Close the database (clears cache)"""
        self.chunk_cache.clear()