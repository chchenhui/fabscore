#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) System for Catalyst Discovery
Retrieves relevant catalyst knowledge to ground LLM generation
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai
from tqdm import tqdm


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    content: Dict[str, Any]
    score: float
    source: str
    relevance_explanation: Optional[str] = None


class CatalystRAGSystem:
    def __init__(self, 
                 index_dir: str = "data/indexes",
                 model_name: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt-4"):
        self.index_dir = Path(index_dir)
        self.embedding_model = SentenceTransformer(model_name)
        self.llm_model = llm_model
        self.index = None
        self.metadata = None
        self.context_cache = {}
        
        # Load API key
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if self.api_key:
            openai.api_key = self.api_key
    
    def load_index(self, index_name: str = "catalyst_index"):
        """Load pre-built FAISS index and metadata"""
        info_file = self.index_dir / f"{index_name}_latest.json"
        
        if not info_file.exists():
            raise FileNotFoundError(f"Index info not found: {info_file}")
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(info["index_file"])
        
        # Load metadata
        with open(info["metadata_file"], 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} vectors")
        return True
    
    def retrieve_context(self, 
                        query: str, 
                        k: int = 10,
                        filter_criteria: Optional[Dict] = None) -> List[RetrievalResult]:
        """Retrieve relevant context based on query"""
        if self.index is None:
            raise ValueError("No index loaded. Please load index first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, k * 2)  # Get more for filtering
        
        # Process results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                material = self.metadata[idx]
                
                # Apply filters if specified
                if filter_criteria and not self._passes_filters(material, filter_criteria):
                    continue
                
                result = RetrievalResult(
                    content=material,
                    score=float(distance),
                    source=material.get("source", "unknown")
                )
                results.append(result)
                
                if len(results) >= k:
                    break
        
        return results
    
    def _passes_filters(self, material: Dict, filters: Dict) -> bool:
        """Check if material passes filter criteria"""
        for key, value in filters.items():
            if key not in material:
                return False
            
            if isinstance(value, dict):
                # Range filter
                mat_value = material[key]
                if "min" in value and mat_value < value["min"]:
                    return False
                if "max" in value and mat_value > value["max"]:
                    return False
            elif isinstance(value, list):
                # Must contain one of the values
                if material[key] not in value:
                    return False
            else:
                # Exact match
                if material[key] != value:
                    return False
        
        return True
    
    def retrieve_for_hypothesis(self, 
                               target_properties: Dict,
                               constraints: Dict,
                               k: int = 20) -> Dict[str, List[RetrievalResult]]:
        """Retrieve context specifically for hypothesis generation"""
        retrieved_context = {
            "similar_materials": [],
            "property_matches": [],
            "structural_analogs": [],
            "reaction_examples": []
        }
        
        # 1. Find materials with similar target properties
        property_query = self._build_property_query(target_properties)
        similar_materials = self.retrieve_context(property_query, k=k//2)
        retrieved_context["similar_materials"] = similar_materials
        
        # 2. Find materials matching specific constraints
        if constraints:
            constraint_results = self.retrieve_context(
                query=self._build_constraint_query(constraints),
                k=k//4,
                filter_criteria=constraints
            )
            retrieved_context["property_matches"] = constraint_results
        
        # 3. Find structural analogs if formula pattern provided
        if "base_structure" in target_properties:
            structural_query = f"Structure similar to {target_properties['base_structure']}"
            structural_results = self.retrieve_context(structural_query, k=k//4)
            retrieved_context["structural_analogs"] = structural_results
        
        return retrieved_context
    
    def _build_property_query(self, properties: Dict) -> str:
        """Build query string from target properties"""
        parts = []
        
        property_descriptions = {
            "activity": "catalytic activity",
            "selectivity": "product selectivity",
            "stability": "structural stability",
            "band_gap": "electronic band gap",
            "work_function": "surface work function",
            "adsorption_energy": "adsorbate binding energy"
        }
        
        for prop, value in properties.items():
            if prop in property_descriptions:
                parts.append(f"{property_descriptions[prop]}: {value}")
            else:
                parts.append(f"{prop}: {value}")
        
        return " | ".join(parts)
    
    def _build_constraint_query(self, constraints: Dict) -> str:
        """Build query from constraints"""
        parts = []
        
        if "elements" in constraints:
            parts.append(f"Contains elements: {', '.join(constraints['elements'])}")
        
        if "max_elements" in constraints:
            parts.append(f"Maximum {constraints['max_elements']} elements")
        
        if "structure_type" in constraints:
            parts.append(f"Structure type: {constraints['structure_type']}")
        
        return " | ".join(parts)
    
    def augment_generation_prompt(self,
                                 base_prompt: str,
                                 retrieved_context: Dict[str, List[RetrievalResult]],
                                 max_examples: int = 5) -> str:
        """Augment LLM prompt with retrieved context"""
        augmented_prompt = base_prompt + "\n\n### Retrieved Context:\n\n"
        
        # Add similar materials
        if retrieved_context["similar_materials"]:
            augmented_prompt += "#### Similar Known Catalysts:\n"
            for i, result in enumerate(retrieved_context["similar_materials"][:max_examples]):
                mat = result.content
                augmented_prompt += f"{i+1}. {mat.get('formula', 'Unknown')} - "
                augmented_prompt += f"Source: {mat.get('source', 'unknown')}"
                
                # Add key properties
                props = []
                if "band_gap" in mat and mat["band_gap"] is not None:
                    props.append(f"Band gap: {mat['band_gap']:.2f} eV")
                if "formation_energy" in mat and mat["formation_energy"] is not None:
                    props.append(f"Formation energy: {mat['formation_energy']:.3f} eV/atom")
                if "adsorption_energy" in mat and mat["adsorption_energy"] is not None:
                    props.append(f"Adsorption energy: {mat['adsorption_energy']:.3f} eV")
                
                if props:
                    augmented_prompt += f" | {', '.join(props)}"
                augmented_prompt += "\n"
        
        # Add property matches
        if retrieved_context["property_matches"]:
            augmented_prompt += "\n#### Materials Matching Constraints:\n"
            for i, result in enumerate(retrieved_context["property_matches"][:max_examples]):
                mat = result.content
                augmented_prompt += f"{i+1}. {mat.get('formula', 'Unknown')}\n"
        
        # Add structural analogs
        if retrieved_context["structural_analogs"]:
            augmented_prompt += "\n#### Structural Analogs:\n"
            for i, result in enumerate(retrieved_context["structural_analogs"][:max_examples//2]):
                mat = result.content
                augmented_prompt += f"{i+1}. {mat.get('formula', 'Unknown')}\n"
        
        augmented_prompt += "\n### Generation Instructions:\n"
        augmented_prompt += "Based on the retrieved context above, generate catalyst candidates that:\n"
        augmented_prompt += "1. Show similarities to successful examples\n"
        augmented_prompt += "2. Satisfy the specified constraints\n"
        augmented_prompt += "3. Explore chemical space near the retrieved materials\n"
        augmented_prompt += "4. Consider element substitutions based on periodic trends\n\n"
        
        return augmented_prompt
    
    def evaluate_relevance(self, 
                          query: str, 
                          retrieved_results: List[RetrievalResult],
                          use_llm: bool = True) -> List[RetrievalResult]:
        """Evaluate and re-rank retrieved results for relevance"""
        if not use_llm or not self.api_key:
            # Simple scoring based on similarity
            return sorted(retrieved_results, key=lambda x: x.score, reverse=True)
        
        # Use LLM to evaluate relevance
        evaluated_results = []
        
        for result in retrieved_results:
            relevance_prompt = f"""
            Query: {query}
            
            Retrieved Material:
            Formula: {result.content.get('formula', 'Unknown')}
            Properties: {json.dumps({k: v for k, v in result.content.items() 
                                    if k in ['band_gap', 'formation_energy', 'adsorption_energy']}, indent=2)}
            
            Rate the relevance of this material to the query on a scale of 0-1 and explain why.
            Response format: {{"score": 0.X, "explanation": "..."}}
            """
            
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": relevance_prompt}],
                    temperature=0.3,
                    max_tokens=100
                )
                
                eval_data = json.loads(response.choices[0].message.content)
                result.score = eval_data["score"]
                result.relevance_explanation = eval_data["explanation"]
                
            except Exception as e:
                print(f"Error evaluating relevance: {e}")
                # Keep original score
            
            evaluated_results.append(result)
        
        # Re-sort by updated scores
        return sorted(evaluated_results, key=lambda x: x.score, reverse=True)
    
    def get_diverse_results(self, 
                           query: str, 
                           k: int = 10,
                           diversity_weight: float = 0.3) -> List[RetrievalResult]:
        """Retrieve diverse results using MMR (Maximal Marginal Relevance)"""
        if self.index is None:
            raise ValueError("No index loaded.")
        
        # Get initial candidates
        candidates = self.retrieve_context(query, k=k*3)
        
        if len(candidates) <= k:
            return candidates
        
        # Implement MMR for diversity
        selected = [candidates[0]]  # Start with most relevant
        candidates = candidates[1:]
        
        while len(selected) < k and candidates:
            mmr_scores = []
            
            for candidate in candidates:
                # Relevance score (already have this)
                relevance = candidate.score
                
                # Diversity score (minimum similarity to selected items)
                diversity_scores = []
                for selected_item in selected:
                    # Simple diversity based on formula similarity
                    div_score = self._calculate_diversity(
                        candidate.content, 
                        selected_item.content
                    )
                    diversity_scores.append(div_score)
                
                diversity = min(diversity_scores) if diversity_scores else 1.0
                
                # MMR score
                mmr = (1 - diversity_weight) * relevance + diversity_weight * diversity
                mmr_scores.append((candidate, mmr))
            
            # Select best MMR score
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            selected.append(mmr_scores[0][0])
            candidates = [c for c, _ in mmr_scores[1:]]
        
        return selected
    
    def _calculate_diversity(self, mat1: Dict, mat2: Dict) -> float:
        """Calculate diversity score between two materials"""
        # Simple diversity based on element overlap
        elements1 = set(mat1.get("elements", []))
        elements2 = set(mat2.get("elements", []))
        
        if not elements1 or not elements2:
            return 0.5
        
        intersection = len(elements1 & elements2)
        union = len(elements1 | elements2)
        
        if union == 0:
            return 1.0
        
        jaccard = intersection / union
        return 1.0 - jaccard


def main():
    """Example usage of RAG system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG retrieval for catalyst discovery")
    parser.add_argument("--index-dir", default="data/indexes", help="Directory containing indexes")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--target-properties", type=str, help="JSON string of target properties")
    parser.add_argument("--k", type=int, default=10, help="Number of results to retrieve")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = CatalystRAGSystem(index_dir=args.index_dir)
    
    # Load index
    rag.load_index()
    
    if args.query:
        # Simple retrieval
        print(f"\nSearching for: {args.query}")
        results = rag.retrieve_context(args.query, k=args.k)
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result.score:.3f}")
            print(f"   Formula: {result.content.get('formula', 'N/A')}")
            print(f"   Source: {result.content.get('source', 'N/A')}")
            
    elif args.target_properties:
        # Hypothesis-driven retrieval
        target_props = json.loads(args.target_properties)
        print(f"\nRetrieving context for: {target_props}")
        
        context = rag.retrieve_for_hypothesis(
            target_properties=target_props,
            constraints={},
            k=args.k
        )
        
        print("\nRetrieved context summary:")
        for category, results in context.items():
            if results:
                print(f"\n{category}: {len(results)} materials")
                for result in results[:3]:
                    print(f"  - {result.content.get('formula', 'Unknown')}")


if __name__ == "__main__":
    main()