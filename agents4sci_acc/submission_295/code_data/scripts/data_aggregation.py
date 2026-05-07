#!/usr/bin/env python3
"""
Data Aggregation Script for Catalyst Discovery
Collects and consolidates catalyst data from multiple sources:
- Literature databases (via APIs)
- Materials Project
- NOMAD repository
- OC20 dataset
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class CatalystDataAggregator:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config.get("output_dir", "data/raw"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if no config file exists"""
        return {
            "materials_project": {
                "api_key": os.getenv("MP_API_KEY", ""),
                "base_url": "https://api.materialsproject.org",
                "elements": ["Fe", "Cu", "Ni", "Co", "Mn", "Ti", "V", "Cr", "Mo", "W"]
            },
            "nomad": {
                "base_url": "https://nomad-lab.eu/prod/v1/api/v1",
                "max_entries": 10000
            },
            "oc20": {
                "data_dir": "data/oc20",
                "catalysts_of_interest": ["Cu", "Fe", "Ni", "Co", "Mn"]
            },
            "literature": {
                "scopus_api_key": os.getenv("SCOPUS_API_KEY", ""),
                "keywords": ["catalyst", "electrocatalyst", "HEA", "high entropy alloy", 
                           "CO2 reduction", "hydrogen evolution", "oxygen evolution"]
            },
            "output_dir": "data/raw"
        }
    
    def fetch_materials_project_data(self) -> List[Dict]:
        """Fetch catalyst-relevant materials from Materials Project"""
        print("Fetching data from Materials Project...")
        materials = []
        
        mp_config = self.config["materials_project"]
        if not mp_config["api_key"]:
            print("Warning: No Materials Project API key found. Skipping MP data.")
            return materials
        
        try:
            from mp_api.client import MPRester
            
            with MPRester(mp_config["api_key"]) as mpr:
                # Query for materials containing transition metals
                elements = mp_config["elements"]
                
                # Search for materials with specific properties relevant to catalysis
                docs = mpr.materials.search(
                    elements=elements,
                    num_elements=(1, 5),  # Single metals to HEAs
                    fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                           "energy_above_hull", "band_gap", "structure", "surface_energy",
                           "work_function", "elastic", "bulk_modulus", "shear_modulus"]
                )
                
                for doc in docs:
                    material_data = {
                        "source": "materials_project",
                        "material_id": str(doc.material_id),
                        "formula": doc.formula_pretty,
                        "formation_energy": doc.formation_energy_per_atom,
                        "energy_above_hull": doc.energy_above_hull,
                        "band_gap": doc.band_gap,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add optional properties if available
                    if hasattr(doc, 'surface_energy'):
                        material_data["surface_energy"] = doc.surface_energy
                    if hasattr(doc, 'work_function'):
                        material_data["work_function"] = doc.work_function
                        
                    materials.append(material_data)
                    
        except Exception as e:
            print(f"Error fetching Materials Project data: {e}")
            
        print(f"Fetched {len(materials)} materials from Materials Project")
        return materials
    
    def fetch_nomad_data(self) -> List[Dict]:
        """Fetch catalyst data from NOMAD repository"""
        print("Fetching data from NOMAD...")
        materials = []
        
        nomad_config = self.config["nomad"]
        base_url = nomad_config["base_url"]
        
        # Define search query for catalysts
        query = {
            "query": {
                "and": [
                    {"quantities": "results.properties.electronic.band_structure_electronic.band_gap"},
                    {"results.material.elements": {"any": self.config["materials_project"]["elements"]}}
                ]
            },
            "pagination": {
                "page_size": 100,
                "page": 1
            },
            "required": {
                "include": ["results.material", "results.properties", "results.method"]
            }
        }
        
        try:
            response = requests.post(
                f"{base_url}/entries/query",
                json=query,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                entries = data.get("data", [])
                
                for entry in entries:
                    material_data = {
                        "source": "nomad",
                        "entry_id": entry.get("entry_id"),
                        "formula": entry.get("results", {}).get("material", {}).get("chemical_formula_reduced"),
                        "elements": entry.get("results", {}).get("material", {}).get("elements"),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Extract properties if available
                    properties = entry.get("results", {}).get("properties", {})
                    if properties:
                        electronic = properties.get("electronic", {})
                        if electronic:
                            material_data["band_gap"] = electronic.get("band_structure_electronic", {}).get("band_gap")
                    
                    materials.append(material_data)
                    
        except Exception as e:
            print(f"Error fetching NOMAD data: {e}")
            
        print(f"Fetched {len(materials)} materials from NOMAD")
        return materials
    
    def fetch_oc20_data(self) -> List[Dict]:
        """Load relevant OC20 dataset entries"""
        print("Loading OC20 dataset...")
        materials = []
        
        oc20_config = self.config["oc20"]
        data_dir = Path(oc20_config["data_dir"])
        
        if not data_dir.exists():
            print(f"Warning: OC20 data directory {data_dir} not found. Skipping OC20 data.")
            return materials
        
        # Load metadata if available
        metadata_file = data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            # Filter for relevant catalysts
            for entry in metadata:
                if any(elem in entry.get("elements", []) for elem in oc20_config["catalysts_of_interest"]):
                    material_data = {
                        "source": "oc20",
                        "system_id": entry.get("system_id"),
                        "formula": entry.get("formula"),
                        "elements": entry.get("elements"),
                        "adsorbate": entry.get("adsorbate"),
                        "adsorption_energy": entry.get("adsorption_energy"),
                        "timestamp": datetime.now().isoformat()
                    }
                    materials.append(material_data)
        
        print(f"Loaded {len(materials)} materials from OC20")
        return materials
    
    def fetch_literature_data(self) -> List[Dict]:
        """Fetch literature data from Scopus/other databases"""
        print("Fetching literature data...")
        papers = []
        
        lit_config = self.config["literature"]
        if not lit_config["scopus_api_key"]:
            print("Warning: No Scopus API key found. Skipping literature data.")
            return papers
        
        # This is a placeholder - implement actual Scopus API calls
        # For now, return empty list
        print("Literature fetching not yet implemented")
        return papers
    
    def aggregate_all_sources(self) -> Dict:
        """Aggregate data from all sources"""
        print("\n=== Starting data aggregation ===")
        
        all_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "sources": ["materials_project", "nomad", "oc20", "literature"],
                "version": "1.0"
            },
            "materials": []
        }
        
        # Fetch data from each source in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.fetch_materials_project_data): "materials_project",
                executor.submit(self.fetch_nomad_data): "nomad",
                executor.submit(self.fetch_oc20_data): "oc20",
                executor.submit(self.fetch_literature_data): "literature"
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    data = future.result()
                    all_data["materials"].extend(data)
                    print(f"✓ Completed {source}")
                except Exception as e:
                    print(f"✗ Error in {source}: {e}")
        
        # Save aggregated data
        output_file = self.output_dir / f"aggregated_catalyst_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n=== Aggregation complete ===")
        print(f"Total materials collected: {len(all_data['materials'])}")
        print(f"Data saved to: {output_file}")
        
        # Create summary statistics
        self._create_summary(all_data)
        
        return all_data
    
    def _create_summary(self, data: Dict):
        """Create summary statistics of aggregated data"""
        df = pd.DataFrame(data["materials"])
        
        summary = {
            "total_entries": len(df),
            "by_source": df["source"].value_counts().to_dict() if not df.empty else {},
            "unique_formulas": df["formula"].nunique() if "formula" in df.columns else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        summary_file = self.output_dir / "aggregation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")


def main():
    """Main execution function"""
    aggregator = CatalystDataAggregator()
    aggregator.aggregate_all_sources()


if __name__ == "__main__":
    main()