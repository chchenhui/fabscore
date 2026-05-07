#!/usr/bin/env python3
"""
Feedback Loop Script
Updates knowledge base with computational validation results
Improves future catalyst generation through learning
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib
from collections import defaultdict


class FeedbackSystem:
    def __init__(self, 
                 knowledge_base_dir: str = "data/knowledge_base",
                 model_dir: str = "models"):
        self.kb_dir = Path(knowledge_base_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self.validation_db = self.kb_dir / "validation_results.json"
        self.success_patterns_db = self.kb_dir / "success_patterns.json"
        self.failure_patterns_db = self.kb_dir / "failure_patterns.json"
        self.descriptor_db = self.kb_dir / "descriptor_correlations.json"
        
        # Load existing data
        self.validation_history = self._load_json(self.validation_db, default=[])
        self.success_patterns = self._load_json(self.success_patterns_db, default={})
        self.failure_patterns = self._load_json(self.failure_patterns_db, default={})
        self.descriptor_correlations = self._load_json(self.descriptor_db, default={})
        
        # Initialize ML models for pattern learning
        self.property_predictor = None
        self.success_classifier = None
        
    def _load_json(self, file_path: Path, default: Any = None) -> Any:
        """Load JSON file with default value"""
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return default
    
    def record_validation_result(self,
                               candidate: Dict,
                               dft_results: Dict,
                               screening_results: Dict,
                               generation_metadata: Dict) -> str:
        """Record validation results for a catalyst candidate"""
        # Create unique ID for this validation
        validation_id = self._generate_validation_id(candidate)
        
        # Compile complete record
        record = {
            "id": validation_id,
            "timestamp": datetime.now().isoformat(),
            "candidate": candidate,
            "generation": {
                "strategy": generation_metadata.get("strategy"),
                "prompt_template": generation_metadata.get("template"),
                "retrieved_context": generation_metadata.get("context_ids", [])
            },
            "screening": screening_results,
            "dft_results": dft_results,
            "success_metrics": self._calculate_success_metrics(dft_results, candidate)
        }
        
        # Add to history
        self.validation_history.append(record)
        
        # Update patterns
        self._update_patterns(record)
        
        # Save updated data
        self._save_validation_history()
        
        return validation_id
    
    def _generate_validation_id(self, candidate: Dict) -> str:
        """Generate unique ID for validation record"""
        content = f"{candidate.get('formula', '')}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _calculate_success_metrics(self, 
                                  dft_results: Dict, 
                                  candidate: Dict) -> Dict[str, Any]:
        """Calculate success metrics from DFT results"""
        metrics = {
            "stability_score": 0.0,
            "activity_score": 0.0,
            "selectivity_score": 0.0,
            "overall_score": 0.0
        }
        
        # Stability score based on formation energy and hull distance
        if "formation_energy" in dft_results:
            fe = dft_results["formation_energy"]
            # Lower formation energy is better (more stable)
            metrics["stability_score"] = max(0, 1 - abs(fe) / 2.0)
        
        if "energy_above_hull" in dft_results:
            hull_dist = dft_results["energy_above_hull"]
            # Closer to hull is better
            hull_score = max(0, 1 - hull_dist / 0.5)
            metrics["stability_score"] = (metrics["stability_score"] + hull_score) / 2
        
        # Activity score based on adsorption energies
        if "adsorption_energies" in dft_results:
            ads_energies = dft_results["adsorption_energies"]
            
            # Example: optimal CO binding for CO2 reduction is around -0.6 eV
            if "CO_top" in ads_energies:
                co_binding = ads_energies["CO_top"]
                optimal_co = -0.6
                metrics["activity_score"] = max(0, 1 - abs(co_binding - optimal_co) / 1.0)
            
            # Selectivity based on binding energy differences
            if "H_top" in ads_energies and "CO_top" in ads_energies:
                # Want CO binding stronger than H for CO2 reduction selectivity
                selectivity = ads_energies["H_top"] - ads_energies["CO_top"]
                metrics["selectivity_score"] = 1 / (1 + np.exp(-selectivity))
        
        # Overall score
        weights = {"stability": 0.3, "activity": 0.5, "selectivity": 0.2}
        metrics["overall_score"] = (
            weights["stability"] * metrics["stability_score"] +
            weights["activity"] * metrics["activity_score"] +
            weights["selectivity"] * metrics["selectivity_score"]
        )
        
        # Add binary success flag
        metrics["is_successful"] = metrics["overall_score"] > 0.6
        
        return metrics
    
    def _update_patterns(self, record: Dict):
        """Update success and failure patterns"""
        success_metrics = record["success_metrics"]
        candidate = record["candidate"]
        
        if success_metrics["is_successful"]:
            # Update success patterns
            self._update_success_patterns(record)
        else:
            # Update failure patterns
            self._update_failure_patterns(record)
        
        # Update descriptor correlations
        self._update_descriptor_correlations(record)
    
    def _update_success_patterns(self, record: Dict):
        """Extract and update patterns from successful candidates"""
        candidate = record["candidate"]
        
        # Element combinations
        if "formula" in candidate:
            elements = self._extract_elements(candidate["formula"])
            element_key = "-".join(sorted(elements))
            
            if element_key not in self.success_patterns:
                self.success_patterns[element_key] = {
                    "count": 0,
                    "avg_score": 0.0,
                    "examples": []
                }
            
            pattern = self.success_patterns[element_key]
            pattern["count"] += 1
            pattern["avg_score"] = (
                (pattern["avg_score"] * (pattern["count"] - 1) + 
                 record["success_metrics"]["overall_score"]) / pattern["count"]
            )
            pattern["examples"].append({
                "formula": candidate["formula"],
                "score": record["success_metrics"]["overall_score"]
            })
            
            # Keep only top 5 examples
            pattern["examples"] = sorted(
                pattern["examples"], 
                key=lambda x: x["score"], 
                reverse=True
            )[:5]
        
        # Structure type patterns
        if "structure" in candidate:
            struct_type = candidate["structure"]
            if struct_type not in self.success_patterns:
                self.success_patterns[struct_type] = {"count": 0, "formulas": []}
            
            self.success_patterns[struct_type]["count"] += 1
            self.success_patterns[struct_type]["formulas"].append(candidate["formula"])
    
    def _update_failure_patterns(self, record: Dict):
        """Extract and update patterns from failed candidates"""
        candidate = record["candidate"]
        metrics = record["success_metrics"]
        
        # Identify failure reasons
        failure_reasons = []
        
        if metrics["stability_score"] < 0.3:
            failure_reasons.append("poor_stability")
        if metrics["activity_score"] < 0.3:
            failure_reasons.append("poor_activity")
        if metrics["selectivity_score"] < 0.3:
            failure_reasons.append("poor_selectivity")
        
        for reason in failure_reasons:
            if reason not in self.failure_patterns:
                self.failure_patterns[reason] = {
                    "count": 0,
                    "common_features": defaultdict(int)
                }
            
            self.failure_patterns[reason]["count"] += 1
            
            # Track common features in failures
            if "formula" in candidate:
                elements = self._extract_elements(candidate["formula"])
                for elem in elements:
                    self.failure_patterns[reason]["common_features"][f"contains_{elem}"] += 1
    
    def _update_descriptor_correlations(self, record: Dict):
        """Update correlations between descriptors and performance"""
        dft_results = record["dft_results"]
        metrics = record["success_metrics"]
        
        # Collect descriptor values
        descriptors = {
            "formation_energy": dft_results.get("formation_energy"),
            "band_gap": dft_results.get("band_gap"),
            "d_band_center": dft_results.get("d_band_center"),
            "work_function": dft_results.get("work_function"),
            "co_binding": dft_results.get("adsorption_energies", {}).get("CO_top"),
            "h_binding": dft_results.get("adsorption_energies", {}).get("H_top")
        }
        
        # Update correlations with performance metrics
        for descriptor, value in descriptors.items():
            if value is not None:
                if descriptor not in self.descriptor_correlations:
                    self.descriptor_correlations[descriptor] = {
                        "values": [],
                        "scores": []
                    }
                
                self.descriptor_correlations[descriptor]["values"].append(value)
                self.descriptor_correlations[descriptor]["scores"].append(
                    metrics["overall_score"]
                )
    
    def _extract_elements(self, formula: str) -> List[str]:
        """Extract element symbols from formula"""
        import re
        # Simple regex to extract element symbols
        return re.findall(r'[A-Z][a-z]?', formula)
    
    def train_property_predictor(self):
        """Train ML model to predict catalyst properties"""
        if len(self.validation_history) < 20:
            print("Not enough data for training (need at least 20 examples)")
            return
        
        print("Training property predictor...")
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if X is None or len(X) == 0:
            print("Failed to prepare training data")
            return
        
        # Train random forest
        self.property_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.property_predictor.fit(X, y)
        
        # Save model
        model_file = self.model_dir / "property_predictor.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.property_predictor, f)
        
        print(f"Property predictor trained and saved to {model_file}")
        
        # Feature importance analysis
        self._analyze_feature_importance()
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training"""
        features = []
        targets = []
        
        for record in self.validation_history:
            # Extract features
            feature_vec = self._extract_features(record["candidate"])
            if feature_vec is not None:
                features.append(feature_vec)
                targets.append(record["success_metrics"]["overall_score"])
        
        if not features:
            return None, None
        
        return np.array(features), np.array(targets)
    
    def _extract_features(self, candidate: Dict) -> Optional[np.ndarray]:
        """Extract numerical features from candidate"""
        features = []
        
        # Composition features
        if "formula" in candidate:
            elements = self._extract_elements(candidate["formula"])
            
            # One-hot encoding for common elements
            common_elements = ["Fe", "Co", "Ni", "Cu", "Mn", "Ti", "V", "Cr", "Mo", "W"]
            for elem in common_elements:
                features.append(1.0 if elem in elements else 0.0)
            
            # Number of elements
            features.append(len(elements))
        else:
            return None
        
        # Add any numerical properties if available
        if "properties" in candidate:
            for prop in ["band_gap", "formation_energy", "work_function"]:
                features.append(candidate["properties"].get(prop, 0.0))
        
        return np.array(features)
    
    def _analyze_feature_importance(self):
        """Analyze and save feature importance"""
        if self.property_predictor is None:
            return
        
        importances = self.property_predictor.feature_importances_
        
        # Define feature names
        feature_names = [
            "has_Fe", "has_Co", "has_Ni", "has_Cu", "has_Mn",
            "has_Ti", "has_V", "has_Cr", "has_Mo", "has_W",
            "num_elements"
        ]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        importance_report = "Feature Importances:\n"
        for i in range(min(len(feature_names), len(indices))):
            idx = indices[i]
            if idx < len(feature_names):
                importance_report += f"{feature_names[idx]}: {importances[idx]:.3f}\n"
        
        # Save report
        report_file = self.model_dir / "feature_importance.txt"
        with open(report_file, 'w') as f:
            f.write(importance_report)
        
        print(f"Feature importance saved to {report_file}")
    
    def generate_improvement_suggestions(self, 
                                       failed_candidate: Dict,
                                       failure_reasons: List[str]) -> List[Dict]:
        """Generate suggestions for improving failed candidates"""
        suggestions = []
        
        # Analyze failure patterns
        for reason in failure_reasons:
            if reason == "poor_stability":
                suggestions.extend(self._suggest_stability_improvements(failed_candidate))
            elif reason == "poor_activity":
                suggestions.extend(self._suggest_activity_improvements(failed_candidate))
            elif reason == "poor_selectivity":
                suggestions.extend(self._suggest_selectivity_improvements(failed_candidate))
        
        # Use success patterns to suggest modifications
        successful_patterns = self._find_similar_successes(failed_candidate)
        for pattern in successful_patterns[:3]:
            suggestions.append({
                "type": "composition_modification",
                "suggestion": f"Consider composition similar to {pattern['formula']}",
                "rationale": f"Similar successful catalyst with score {pattern['score']:.2f}"
            })
        
        return suggestions
    
    def _suggest_stability_improvements(self, candidate: Dict) -> List[Dict]:
        """Suggest modifications to improve stability"""
        suggestions = []
        
        # Based on learned patterns
        if "Fe" in candidate.get("formula", ""):
            suggestions.append({
                "type": "element_substitution",
                "suggestion": "Replace some Fe with Co or Ni",
                "rationale": "Co and Ni typically form more stable alloys"
            })
        
        # Suggest structure modifications
        suggestions.append({
            "type": "structure_modification",
            "suggestion": "Consider ordered intermetallic structure",
            "rationale": "Ordered structures often have lower formation energy"
        })
        
        return suggestions
    
    def _suggest_activity_improvements(self, candidate: Dict) -> List[Dict]:
        """Suggest modifications to improve activity"""
        suggestions = []
        
        # Based on descriptor correlations
        if self.descriptor_correlations.get("d_band_center"):
            optimal_dbc = self._find_optimal_descriptor_value("d_band_center")
            suggestions.append({
                "type": "electronic_tuning",
                "suggestion": f"Tune d-band center toward {optimal_dbc:.2f} eV",
                "rationale": "Optimal d-band center correlates with high activity"
            })
        
        return suggestions
    
    def _suggest_selectivity_improvements(self, candidate: Dict) -> List[Dict]:
        """Suggest modifications to improve selectivity"""
        return [{
            "type": "binding_optimization",
            "suggestion": "Increase CO binding relative to H binding",
            "rationale": "Stronger CO binding improves CO2 reduction selectivity"
        }]
    
    def _find_similar_successes(self, candidate: Dict) -> List[Dict]:
        """Find successful catalysts similar to the candidate"""
        similar = []
        
        candidate_elements = set(self._extract_elements(candidate.get("formula", "")))
        
        for record in self.validation_history:
            if record["success_metrics"]["is_successful"]:
                record_elements = set(self._extract_elements(
                    record["candidate"].get("formula", "")
                ))
                
                # Calculate similarity (Jaccard index)
                if candidate_elements and record_elements:
                    similarity = len(candidate_elements & record_elements) / len(
                        candidate_elements | record_elements
                    )
                    
                    if similarity > 0.5:
                        similar.append({
                            "formula": record["candidate"]["formula"],
                            "score": record["success_metrics"]["overall_score"],
                            "similarity": similarity
                        })
        
        return sorted(similar, key=lambda x: x["score"], reverse=True)
    
    def _find_optimal_descriptor_value(self, descriptor: str) -> float:
        """Find descriptor value that correlates with high performance"""
        if descriptor not in self.descriptor_correlations:
            return 0.0
        
        values = np.array(self.descriptor_correlations[descriptor]["values"])
        scores = np.array(self.descriptor_correlations[descriptor]["scores"])
        
        # Find value with highest average score in its neighborhood
        # Simple approach: return value of best performing catalyst
        best_idx = np.argmax(scores)
        return values[best_idx]
    
    def update_vector_database(self, new_validated_materials: List[Dict]):
        """Update vector database with validated materials"""
        # This would integrate with embedding_indexing.py
        update_file = self.kb_dir / f"vector_db_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(update_file, 'w') as f:
            json.dump({
                "materials": new_validated_materials,
                "metadata": {
                    "source": "dft_validation",
                    "timestamp": datetime.now().isoformat()
                }
            }, f, indent=2)
        
        print(f"Vector database update saved to {update_file}")
    
    def _save_validation_history(self):
        """Save all databases"""
        with open(self.validation_db, 'w') as f:
            json.dump(self.validation_history, f, indent=2)
        
        with open(self.success_patterns_db, 'w') as f:
            json.dump(self.success_patterns, f, indent=2)
        
        with open(self.failure_patterns_db, 'w') as f:
            json.dump(self.failure_patterns, f, indent=2)
        
        with open(self.descriptor_db, 'w') as f:
            json.dump(self.descriptor_correlations, f, indent=2)
    
    def generate_learning_report(self) -> Path:
        """Generate comprehensive learning report"""
        report = f"""# Catalyst Discovery Learning Report
Generated: {datetime.now().isoformat()}

## Summary Statistics
- Total validations: {len(self.validation_history)}
- Successful candidates: {sum(1 for r in self.validation_history if r['success_metrics']['is_successful'])}
- Success rate: {sum(1 for r in self.validation_history if r['success_metrics']['is_successful']) / len(self.validation_history) * 100:.1f}%

## Success Patterns
"""
        
        # Top performing element combinations
        element_patterns = [(k, v) for k, v in self.success_patterns.items() 
                          if isinstance(v, dict) and "avg_score" in v]
        element_patterns.sort(key=lambda x: x[1]["avg_score"], reverse=True)
        
        report += "\n### Top Element Combinations:\n"
        for elements, pattern in element_patterns[:5]:
            report += f"- {elements}: avg score {pattern['avg_score']:.3f} ({pattern['count']} examples)\n"
        
        # Failure analysis
        report += "\n## Common Failure Modes:\n"
        for reason, data in self.failure_patterns.items():
            report += f"\n### {reason.replace('_', ' ').title()} ({data['count']} cases)\n"
            
            if "common_features" in data:
                top_features = sorted(
                    data["common_features"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                report += "Common features:\n"
                for feature, count in top_features:
                    report += f"- {feature}: {count} occurrences\n"
        
        # Descriptor insights
        report += "\n## Key Descriptor Correlations:\n"
        for descriptor, data in self.descriptor_correlations.items():
            if len(data["values"]) > 5:
                values = np.array(data["values"])
                scores = np.array(data["scores"])
                correlation = np.corrcoef(values, scores)[0, 1]
                
                report += f"- {descriptor}: correlation = {correlation:.3f}\n"
        
        report_file = self.kb_dir / "learning_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report_file


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feedback loop system")
    parser.add_argument("--action", choices=["record", "train", "report", "suggest"],
                       required=True, help="Action to perform")
    parser.add_argument("--validation-file", type=str,
                       help="File containing validation results")
    parser.add_argument("--candidate-file", type=str,
                       help="File containing candidate for suggestions")
    
    args = parser.parse_args()
    
    # Initialize feedback system
    feedback = FeedbackSystem()
    
    if args.action == "record":
        if not args.validation_file:
            print("Error: --validation-file required for recording")
            return
        
        # Load validation results
        with open(args.validation_file, 'r') as f:
            validation_data = json.load(f)
        
        # Record results
        for result in validation_data.get("results", []):
            validation_id = feedback.record_validation_result(
                candidate=result["candidate"],
                dft_results=result["dft_results"],
                screening_results=result.get("screening", {}),
                generation_metadata=result.get("generation", {})
            )
            print(f"Recorded validation {validation_id}")
        
        print(f"Recorded {len(validation_data.get('results', []))} validations")
        
    elif args.action == "train":
        feedback.train_property_predictor()
        
    elif args.action == "report":
        report_path = feedback.generate_learning_report()
        print(f"Learning report generated: {report_path}")
        
    elif args.action == "suggest":
        if not args.candidate_file:
            print("Error: --candidate-file required for suggestions")
            return
        
        # Load candidate
        with open(args.candidate_file, 'r') as f:
            candidate_data = json.load(f)
        
        # Generate suggestions
        suggestions = feedback.generate_improvement_suggestions(
            failed_candidate=candidate_data["candidate"],
            failure_reasons=candidate_data.get("failure_reasons", ["poor_activity"])
        )
        
        print("\nImprovement Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion['type']}:")
            print(f"   Suggestion: {suggestion['suggestion']}")
            print(f"   Rationale: {suggestion['rationale']}")


if __name__ == "__main__":
    main()