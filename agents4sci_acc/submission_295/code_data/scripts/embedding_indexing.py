#!/usr/bin/env python3
"""
Embedding and Vector Database Indexing Script
Creates embeddings from catalyst data and builds searchable vector database
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import hashlib


class CatalystEmbeddingIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_type: str = "faiss"):
        self.model = SentenceTransformer(model_name)
        self.index_type = index_type
        self.index = None
        self.metadata = []
        self.embeddings_cache = {}
        self.data_dir = Path("data")
        self.index_dir = self.data_dir / "indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
    def create_text_representation(self, material: Dict) -> str:
        """Create text representation of material for embedding"""
        parts = []
        
        # Add formula if available
        if "formula" in material:
            parts.append(f"Chemical formula: {material['formula']}")
        
        # Add elements
        if "elements" in material:
            elements_str = ", ".join(material['elements']) if isinstance(material['elements'], list) else str(material['elements'])
            parts.append(f"Elements: {elements_str}")
        
        # Add key properties
        property_mappings = {
            "formation_energy": "Formation energy per atom",
            "band_gap": "Band gap",
            "surface_energy": "Surface energy",
            "work_function": "Work function",
            "adsorption_energy": "Adsorption energy",
            "energy_above_hull": "Energy above hull"
        }
        
        for prop_key, prop_name in property_mappings.items():
            if prop_key in material and material[prop_key] is not None:
                parts.append(f"{prop_name}: {material[prop_key]:.3f}")
        
        # Add source information
        parts.append(f"Data source: {material.get('source', 'unknown')}")
        
        # Combine all parts
        text = " | ".join(parts)
        return text
    
    def generate_embeddings(self, materials: List[Dict]) -> np.ndarray:
        """Generate embeddings for all materials"""
        print(f"Generating embeddings for {len(materials)} materials...")
        
        texts = []
        for material in tqdm(materials, desc="Preparing texts"):
            text = self.create_text_representation(material)
            texts.append(text)
            
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        print(f"Created embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search"""
        print("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (equivalent to cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimension)
        
        # For larger datasets, could use IndexIVFFlat for faster search
        # nlist = min(100, len(embeddings) // 10)
        # quantizer = faiss.IndexFlatIP(dimension)
        # index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        # index.train(embeddings)
        
        # Add vectors to index
        index.add(embeddings)
        
        print(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_index(self, index_name: str = "catalyst_index"):
        """Save index and metadata to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save FAISS index
        index_file = self.index_dir / f"{index_name}_{timestamp}.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata_file = self.index_dir / f"{index_name}_{timestamp}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save index info
        info = {
            "index_file": str(index_file),
            "metadata_file": str(metadata_file),
            "num_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "timestamp": timestamp,
            "model_name": self.model.get_sentence_embedding_dimension()
        }
        
        info_file = self.index_dir / f"{index_name}_latest.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Index saved to: {index_file}")
        print(f"Metadata saved to: {metadata_file}")
        
        return info
    
    def load_index(self, index_name: str = "catalyst_index"):
        """Load index and metadata from disk"""
        info_file = self.index_dir / f"{index_name}_latest.json"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Index info file not found: {info_file}")
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(info["index_file"])
        
        # Load metadata
        with open(info["metadata_file"], 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        return self.index, self.metadata
    
    def search_similar(self, query: str, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search for similar materials based on text query"""
        if self.index is None:
            raise ValueError("No index loaded. Please load or build an index first.")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(distance)))
        
        return results
    
    def search_by_properties(self, properties: Dict, k: int = 10) -> List[Tuple[Dict, float]]:
        """Search based on specific property constraints"""
        # Convert properties to text query
        query_parts = []
        
        for prop, value in properties.items():
            if isinstance(value, (int, float)):
                query_parts.append(f"{prop}: {value}")
            elif isinstance(value, list):
                query_parts.append(f"{prop}: {', '.join(map(str, value))}")
            else:
                query_parts.append(f"{prop}: {value}")
        
        query = " | ".join(query_parts)
        return self.search_similar(query, k)
    
    def process_aggregated_data(self, data_file: str):
        """Process aggregated data file and build index"""
        print(f"Loading data from: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        materials = data.get("materials", [])
        
        if not materials:
            print("No materials found in data file")
            return
        
        # Generate embeddings
        embeddings = self.generate_embeddings(materials)
        
        # Build index
        self.index = self.build_faiss_index(embeddings)
        
        # Store metadata
        self.metadata = materials
        
        # Save index
        self.save_index()
        
        # Create statistics
        self._create_index_statistics()
    
    def _create_index_statistics(self):
        """Create statistics about the indexed data"""
        df = pd.DataFrame(self.metadata)
        
        stats = {
            "total_entries": len(df),
            "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
            "properties_coverage": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check property coverage
        properties = ["formula", "band_gap", "formation_energy", "surface_energy", 
                     "work_function", "adsorption_energy"]
        
        for prop in properties:
            if prop in df.columns:
                stats["properties_coverage"][prop] = {
                    "count": df[prop].notna().sum(),
                    "percentage": (df[prop].notna().sum() / len(df) * 100)
                }
        
        stats_file = self.index_dir / "index_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nIndex statistics saved to: {stats_file}")
        print(f"Total indexed materials: {stats['total_entries']}")
        print("Sources distribution:", stats['sources'])


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build embedding index for catalyst data")
    parser.add_argument("--data-file", type=str, required=True, 
                       help="Path to aggregated data JSON file")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence transformer model name")
    parser.add_argument("--search", type=str, help="Search query after building index")
    
    args = parser.parse_args()
    
    # Create indexer
    indexer = CatalystEmbeddingIndexer(model_name=args.model)
    
    # Process data and build index
    indexer.process_aggregated_data(args.data_file)
    
    # Perform test search if requested
    if args.search:
        print(f"\nSearching for: {args.search}")
        results = indexer.search_similar(args.search, k=5)
        
        print("\nTop 5 results:")
        for i, (material, score) in enumerate(results):
            print(f"\n{i+1}. Score: {score:.3f}")
            print(f"   Formula: {material.get('formula', 'N/A')}")
            print(f"   Source: {material.get('source', 'N/A')}")
            if 'band_gap' in material:
                print(f"   Band gap: {material['band_gap']}")


if __name__ == "__main__":
    main()