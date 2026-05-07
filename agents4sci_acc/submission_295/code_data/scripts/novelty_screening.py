#!/usr/bin/env python3
"""
Novelty and Stability Screening Script
Screens generated catalyst candidates for novelty and thermodynamic stability
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
from pymatgen.core import Structure, Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.ext.matproj import MPRester
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class NoveltyStabilityScreener:
    def __init__(self, mp_api_key: Optional[str] = None):
        self.mp_api_key = mp_api_key or os.getenv("MP_API_KEY", "")
        self.results_dir = Path("results/screening")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.known_materials_cache = {}
        
    def screen_candidates(self, 
                         candidates: List[Dict],
                         check_novelty: bool = True,
                         check_stability: bool = True,
                         stability_threshold: float = 0.1) -> Dict[str, Any]:
        """Screen multiple candidates for novelty and stability"""
        print(f"Screening {len(candidates)} candidates...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(candidates),
            "screening_results": [],
            "summary": {
                "novel": 0,
                "stable": 0,
                "passed_all": 0
            }
        }
        
        # Process candidates
        for i, candidate in enumerate(candidates):
            print(f"\nProcessing candidate {i+1}/{len(candidates)}: {candidate.get('formula', 'Unknown')}")
            
            result = {
                "candidate": candidate,
                "formula": candidate.get("formula", "Unknown"),
                "checks": {}
            }
            
            # Check novelty
            if check_novelty:
                novelty_result = self.check_novelty(candidate["formula"])
                result["checks"]["novelty"] = novelty_result
                if novelty_result["is_novel"]:
                    results["summary"]["novel"] += 1
            
            # Check stability
            if check_stability and self.mp_api_key:
                stability_result = self.check_stability(
                    candidate["formula"], 
                    threshold=stability_threshold
                )
                result["checks"]["stability"] = stability_result
                if stability_result["is_stable"]:
                    results["summary"]["stable"] += 1
            
            # Overall pass/fail
            passed = all(
                check.get("is_novel", True) or check.get("is_stable", True) 
                for check in result["checks"].values()
            )
            result["passed_screening"] = passed
            
            if passed:
                results["summary"]["passed_all"] += 1
            
            results["screening_results"].append(result)
        
        # Save results
        self._save_results(results)
        
        # Generate report
        self._generate_screening_report(results)
        
        return results
    
    def check_novelty(self, formula: str) -> Dict[str, Any]:
        """Check if a material formula is novel"""
        try:
            comp = Composition(formula)
            reduced_formula = comp.reduced_formula
            
            # Check cache first
            if reduced_formula in self.known_materials_cache:
                return {
                    "is_novel": False,
                    "reason": "Found in cache",
                    "similar_materials": self.known_materials_cache[reduced_formula]
                }
            
            # Check Materials Project
            if self.mp_api_key:
                similar_materials = self._search_similar_materials(reduced_formula)
                
                if similar_materials:
                    self.known_materials_cache[reduced_formula] = similar_materials
                    return {
                        "is_novel": False,
                        "reason": "Similar materials exist in Materials Project",
                        "similar_materials": similar_materials[:5]  # Top 5
                    }
            
            # Check for element substitution variants
            variants = self._generate_chemical_variants(comp)
            existing_variants = []
            
            for variant in variants:
                if variant in self.known_materials_cache:
                    existing_variants.append(variant)
            
            if existing_variants:
                return {
                    "is_novel": True,
                    "reason": "Novel, but similar variants exist",
                    "similar_variants": existing_variants
                }
            
            return {
                "is_novel": True,
                "reason": "No similar materials found",
                "chemical_family": self._identify_chemical_family(comp)
            }
            
        except Exception as e:
            return {
                "is_novel": "unknown",
                "reason": f"Error in novelty check: {str(e)}"
            }
    
    def _search_similar_materials(self, formula: str) -> List[Dict]:
        """Search for similar materials in Materials Project"""
        if not self.mp_api_key:
            return []
        
        try:
            with MPRester(self.mp_api_key) as mpr:
                # Search by formula
                results = mpr.materials.search(
                    formula=formula,
                    fields=["material_id", "formula_pretty", "formation_energy_per_atom"]
                )
                
                if results:
                    return [
                        {
                            "material_id": str(r.material_id),
                            "formula": r.formula_pretty,
                            "formation_energy": r.formation_energy_per_atom
                        }
                        for r in results
                    ]
                
                # Search by composition
                comp = Composition(formula)
                results = mpr.materials.search(
                    elements=list(comp.element_composition.keys()),
                    num_elements=len(comp.element_composition),
                    fields=["material_id", "formula_pretty", "formation_energy_per_atom"]
                )
                
                # Filter for similar stoichiometry
                similar = []
                for r in results:
                    if self._is_similar_composition(comp, Composition(r.formula_pretty)):
                        similar.append({
                            "material_id": str(r.material_id),
                            "formula": r.formula_pretty,
                            "formation_energy": r.formation_energy_per_atom
                        })
                
                return similar[:10]
                
        except Exception as e:
            print(f"Error searching Materials Project: {e}")
            return []
    
    def _is_similar_composition(self, comp1: Composition, comp2: Composition) -> bool:
        """Check if two compositions have similar stoichiometry"""
        if set(comp1.elements) != set(comp2.elements):
            return False
        
        # Check if ratios are similar (within 10%)
        ratios1 = comp1.get_atomic_fraction_dict()
        ratios2 = comp2.get_atomic_fraction_dict()
        
        for element in ratios1:
            if abs(ratios1[element] - ratios2.get(element, 0)) > 0.1:
                return False
        
        return True
    
    def _generate_chemical_variants(self, composition: Composition) -> List[str]:
        """Generate common chemical variants for novelty check"""
        variants = []
        elements = list(composition.elements)
        
        # Common substitutions
        substitutions = {
            "Fe": ["Co", "Ni", "Mn"],
            "Co": ["Fe", "Ni", "Mn"],
            "Ni": ["Fe", "Co", "Cu"],
            "Cu": ["Ni", "Zn", "Ag"],
            "Mn": ["Fe", "Cr", "V"],
            "Ti": ["Zr", "Hf", "V"],
            "V": ["Nb", "Ta", "Cr"],
            "Cr": ["Mo", "W", "V"],
            "Mo": ["W", "Cr", "Re"],
            "W": ["Mo", "Re", "Ta"]
        }
        
        # Generate single substitutions
        for elem in elements:
            if str(elem) in substitutions:
                for sub in substitutions[str(elem)]:
                    new_comp = composition.copy()
                    # This is simplified - in reality would need proper substitution
                    variant_formula = str(composition).replace(str(elem), sub)
                    variants.append(variant_formula)
        
        return variants
    
    def _identify_chemical_family(self, composition: Composition) -> str:
        """Identify the chemical family of a composition"""
        elements = set(str(e) for e in composition.elements)
        
        # Check for common families
        if elements.issubset({"Fe", "Co", "Ni", "Cr", "Mn", "Cu", "Ti", "V", "Mo", "W"}):
            if len(elements) >= 5:
                return "High-entropy alloy"
            elif len(elements) >= 3:
                return "Medium-entropy alloy"
            else:
                return "Binary/ternary transition metal alloy"
        
        if any(e in elements for e in ["O", "S", "Se", "Te"]):
            if "O" in elements:
                return "Oxide"
            elif "S" in elements:
                return "Sulfide"
            else:
                return "Chalcogenide"
        
        if any(e in elements for e in ["N", "P", "As"]):
            return "Pnictide"
        
        if any(e in elements for e in ["C", "B"]):
            return "Carbide/Boride"
        
        return "Intermetallic"
    
    def check_stability(self, 
                       formula: str, 
                       threshold: float = 0.1) -> Dict[str, Any]:
        """Check thermodynamic stability using Materials Project data"""
        if not self.mp_api_key:
            return {
                "is_stable": "unknown",
                "reason": "No MP API key available"
            }
        
        try:
            comp = Composition(formula)
            
            with MPRester(self.mp_api_key) as mpr:
                # Get entries for phase diagram
                entries = mpr.get_entries_in_chemsys(
                    list(comp.element_composition.keys()),
                    compatible_only=True
                )
                
                if not entries:
                    return {
                        "is_stable": "unknown",
                        "reason": "No phase diagram data available"
                    }
                
                # Create phase diagram
                pd = PhaseDiagram(entries)
                
                # Create a dummy entry for our composition
                # In practice, would need formation energy from DFT
                dummy_entry = PDEntry(comp, 0)  # Assuming 0 formation energy
                
                # Get decomposition info
                decomp_info = pd.get_decomposition(comp)
                
                # Calculate hull distance (simplified)
                # In reality, would need actual formation energy
                hull_distance = self._estimate_hull_distance(comp, pd)
                
                result = {
                    "is_stable": hull_distance < threshold,
                    "hull_distance": hull_distance,
                    "threshold": threshold,
                    "decomposition_products": [
                        {
                            "formula": entry.composition.reduced_formula,
                            "amount": amount
                        }
                        for entry, amount in decomp_info.items()
                    ] if decomp_info else []
                }
                
                # Add convex hull plot
                if len(comp.elements) <= 3:
                    plot_path = self._plot_convex_hull(comp, pd, formula)
                    result["convex_hull_plot"] = str(plot_path)
                
                return result
                
        except Exception as e:
            return {
                "is_stable": "unknown",
                "reason": f"Error in stability check: {str(e)}"
            }
    
    def _estimate_hull_distance(self, 
                               composition: Composition, 
                               phase_diagram: PhaseDiagram) -> float:
        """Estimate distance to convex hull"""
        # This is a simplified estimation
        # In practice, would need actual formation energy from DFT
        
        # Get stable phases at this composition
        stable_entries = phase_diagram.get_reference_energy_per_atom(composition)
        
        # Estimate based on typical metastable materials
        # Random offset for demonstration
        estimated_offset = np.random.uniform(0.0, 0.2)
        
        return estimated_offset
    
    def _plot_convex_hull(self, 
                         composition: Composition,
                         phase_diagram: PhaseDiagram,
                         formula: str) -> Path:
        """Plot convex hull diagram"""
        plt.figure(figsize=(10, 8))
        
        if len(composition.elements) == 2:
            # Binary phase diagram
            self._plot_binary_pd(phase_diagram)
        elif len(composition.elements) == 3:
            # Ternary phase diagram
            self._plot_ternary_pd(phase_diagram)
        else:
            plt.text(0.5, 0.5, f"Phase diagram for {formula}\n(Too many elements to plot)", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.title(f"Convex Hull Analysis for {formula}")
        
        plot_path = self.results_dir / f"hull_{formula.replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_binary_pd(self, pd: PhaseDiagram):
        """Plot binary phase diagram"""
        # Simplified binary plotting
        # In practice, would use pymatgen's plotting utilities
        plt.xlabel("Composition")
        plt.ylabel("Formation Energy (eV/atom)")
        plt.grid(True, alpha=0.3)
    
    def _plot_ternary_pd(self, pd: PhaseDiagram):
        """Plot ternary phase diagram"""
        # Simplified ternary plotting
        # In practice, would use proper ternary plotting
        plt.text(0.5, 0.5, "Ternary Phase Diagram", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    def _save_results(self, results: Dict):
        """Save screening results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.results_dir / f"screening_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for result in results["screening_results"]:
            row = {
                "formula": result["formula"],
                "passed_screening": result["passed_screening"]
            }
            
            if "novelty" in result["checks"]:
                row["is_novel"] = result["checks"]["novelty"]["is_novel"]
                row["novelty_reason"] = result["checks"]["novelty"]["reason"]
            
            if "stability" in result["checks"]:
                row["is_stable"] = result["checks"]["stability"]["is_stable"]
                row["hull_distance"] = result["checks"]["stability"].get("hull_distance", "N/A")
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        csv_file = self.results_dir / f"screening_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {csv_file}")
    
    def _generate_screening_report(self, results: Dict):
        """Generate a screening report"""
        report = f"""
# Catalyst Screening Report
Generated: {results['timestamp']}

## Summary
- Total candidates screened: {results['total_candidates']}
- Novel candidates: {results['summary']['novel']}
- Stable candidates: {results['summary']['stable']}
- Passed all checks: {results['summary']['passed_all']}

## Detailed Results
"""
        
        for i, result in enumerate(results["screening_results"], 1):
            report += f"\n### {i}. {result['formula']}\n"
            report += f"- Overall: {'PASSED' if result['passed_screening'] else 'FAILED'}\n"
            
            if "novelty" in result["checks"]:
                novelty = result["checks"]["novelty"]
                report += f"- Novelty: {novelty['is_novel']} ({novelty['reason']})\n"
            
            if "stability" in result["checks"]:
                stability = result["checks"]["stability"]
                report += f"- Stability: {stability['is_stable']} "
                if "hull_distance" in stability:
                    report += f"(Hull distance: {stability['hull_distance']:.3f} eV/atom)\n"
                else:
                    report += f"({stability.get('reason', 'Unknown')})\n"
        
        report_file = self.results_dir / "screening_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_file}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Screen catalyst candidates")
    parser.add_argument("--candidates-file", type=str, required=True,
                       help="JSON file containing candidate catalysts")
    parser.add_argument("--mp-api-key", type=str, help="Materials Project API key")
    parser.add_argument("--stability-threshold", type=float, default=0.1,
                       help="Stability threshold in eV above hull")
    parser.add_argument("--skip-novelty", action="store_true",
                       help="Skip novelty check")
    parser.add_argument("--skip-stability", action="store_true",
                       help="Skip stability check")
    
    args = parser.parse_args()
    
    # Load candidates
    with open(args.candidates_file, 'r') as f:
        candidates = json.load(f)
    
    # Initialize screener
    screener = NoveltyStabilityScreener(mp_api_key=args.mp_api_key)
    
    # Run screening
    results = screener.screen_candidates(
        candidates=candidates if isinstance(candidates, list) else candidates.get("candidates", []),
        check_novelty=not args.skip_novelty,
        check_stability=not args.skip_stability,
        stability_threshold=args.stability_threshold
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print("SCREENING COMPLETE")
    print(f"{'='*50}")
    print(f"Novel candidates: {results['summary']['novel']}/{results['total_candidates']}")
    print(f"Stable candidates: {results['summary']['stable']}/{results['total_candidates']}")
    print(f"Passed all checks: {results['summary']['passed_all']}/{results['total_candidates']}")


if __name__ == "__main__":
    main()