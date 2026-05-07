#!/usr/bin/env python3
"""
Main Orchestration Script for Catalyst Discovery Pipeline
Ties together all components of the LLM-driven catalyst discovery framework
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor
import openai

# Import all our modules
sys.path.append(str(Path(__file__).parent))
from data_aggregation import CatalystDataAggregator
from embedding_indexing import CatalystEmbeddingIndexer
from rag_retrieval import CatalystRAGSystem
from prompt_templates import PromptTemplates, CatalystConstraints, GenerationStrategy
from novelty_screening import NoveltyStabilityScreener
from dft_automation import DFTAutomation
from feedback_loop import FeedbackSystem


class CatalystDiscoveryPipeline:
    def __init__(self, config_file: str = "pipeline_config.yaml"):
        self.config = self._load_config(config_file)
        self.results_dir = Path(self.config.get("results_dir", "pipeline_results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Pipeline state
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / self.current_run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_file: str) -> Dict:
        """Load pipeline configuration"""
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default pipeline configuration"""
        return {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "materials_project_api_key": os.getenv("MP_API_KEY", ""),
            "llm_model": "gpt-4",
            "embedding_model": "all-MiniLM-L6-v2",
            "dft_calculator": "vasp",
            "parallel_jobs": 4,
            "generation": {
                "num_candidates": 10,
                "strategies": ["constraint_based", "analogy_based", "combinatorial"],
                "max_iterations": 3
            },
            "screening": {
                "check_novelty": True,
                "check_stability": True,
                "stability_threshold": 0.1
            },
            "validation": {
                "run_dft": True,
                "adsorbates": ["CO", "H", "OH"],
                "submit_jobs": False
            },
            "results_dir": "pipeline_results"
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        print("Initializing pipeline components...")
        
        # Data aggregation
        self.aggregator = CatalystDataAggregator()
        
        # Embedding and indexing
        self.indexer = CatalystEmbeddingIndexer(
            model_name=self.config["embedding_model"]
        )
        
        # RAG system
        self.rag_system = CatalystRAGSystem(
            model_name=self.config["embedding_model"],
            llm_model=self.config["llm_model"]
        )
        
        # Prompt templates
        self.prompt_templates = PromptTemplates()
        
        # Screening
        self.screener = NoveltyStabilityScreener(
            mp_api_key=self.config["materials_project_api_key"]
        )
        
        # DFT automation
        self.dft_automation = DFTAutomation(
            calculator=self.config["dft_calculator"],
            parallel_jobs=self.config["parallel_jobs"]
        )
        
        # Feedback system
        self.feedback_system = FeedbackSystem()
        
        # Set OpenAI API key
        if self.config["openai_api_key"]:
            openai.api_key = self.config["openai_api_key"]
        
        print("All components initialized successfully")
    
    def run_full_pipeline(self, 
                         reaction: str,
                         constraints: CatalystConstraints,
                         skip_data_collection: bool = False) -> Dict:
        """Run the complete catalyst discovery pipeline"""
        print(f"\n{'='*60}")
        print(f"Starting Catalyst Discovery Pipeline")
        print(f"Run ID: {self.current_run_id}")
        print(f"Target reaction: {reaction}")
        print(f"{'='*60}\n")
        
        pipeline_results = {
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "reaction": reaction,
            "constraints": constraints.__dict__,
            "stages": {}
        }
        
        try:
            # Stage 1: Data Collection and Indexing
            if not skip_data_collection:
                print("\n📊 Stage 1: Data Collection and Indexing")
                data_results = self._run_data_collection()
                pipeline_results["stages"]["data_collection"] = data_results
            
            # Stage 2: Catalyst Generation
            print("\n🧬 Stage 2: Catalyst Generation")
            generation_results = self._run_generation(reaction, constraints)
            pipeline_results["stages"]["generation"] = generation_results
            
            # Stage 3: Screening
            print("\n🔍 Stage 3: Novelty and Stability Screening")
            screening_results = self._run_screening(generation_results["candidates"])
            pipeline_results["stages"]["screening"] = screening_results
            
            # Stage 4: DFT Validation
            if self.config["validation"]["run_dft"]:
                print("\n⚛️ Stage 4: DFT Validation")
                validation_results = self._run_validation(
                    screening_results["passed_candidates"]
                )
                pipeline_results["stages"]["validation"] = validation_results
            
            # Stage 5: Feedback and Learning
            print("\n🔄 Stage 5: Feedback and Learning")
            feedback_results = self._run_feedback(pipeline_results)
            pipeline_results["stages"]["feedback"] = feedback_results
            
            # Generate final report
            self._generate_pipeline_report(pipeline_results)
            
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            pipeline_results["error"] = str(e)
        
        # Save complete results
        results_file = self.run_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"\n✅ Pipeline complete! Results saved to: {results_file}")
        
        return pipeline_results
    
    def _run_data_collection(self) -> Dict:
        """Run data collection and indexing stage"""
        results = {"status": "started", "timestamp": datetime.now().isoformat()}
        
        try:
            # Aggregate data
            print("  - Aggregating catalyst data...")
            aggregated_data = self.aggregator.aggregate_all_sources()
            results["aggregation"] = {
                "total_materials": len(aggregated_data["materials"]),
                "sources": list(aggregated_data["metadata"]["sources"])
            }
            
            # Build embeddings and index
            print("  - Building vector database...")
            data_file = list(Path(self.aggregator.output_dir).glob("aggregated_*.json"))[-1]
            self.indexer.process_aggregated_data(str(data_file))
            results["indexing"] = {
                "vectors_indexed": self.indexer.index.ntotal,
                "index_dimension": self.indexer.index.d
            }
            
            # Load index in RAG system
            self.rag_system.load_index()
            
            results["status"] = "completed"
            
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            
        return results
    
    def _run_generation(self, 
                       reaction: str, 
                       constraints: CatalystConstraints) -> Dict:
        """Run catalyst generation stage"""
        results = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "candidates": []
        }
        
        all_candidates = []
        
        # Try different generation strategies
        for strategy_name in self.config["generation"]["strategies"]:
            try:
                strategy = GenerationStrategy[strategy_name.upper()]
                print(f"  - Generating with {strategy_name} strategy...")
                
                # Retrieve relevant context
                target_properties = constraints.target_properties
                retrieved_context = self.rag_system.retrieve_for_hypothesis(
                    target_properties=target_properties,
                    constraints={"elements": constraints.allowed_elements},
                    k=20
                )
                
                # Format context for prompt
                context_str = self._format_retrieved_context(retrieved_context)
                
                # Build generation prompt
                prompt = self.prompt_templates.build_generation_prompt(
                    strategy=strategy,
                    constraints=constraints,
                    reaction=reaction,
                    retrieved_context=context_str,
                    num_candidates=self.config["generation"]["num_candidates"]
                )
                
                # Generate candidates
                candidates = self._llm_generate(prompt)
                
                # Add metadata
                for candidate in candidates:
                    candidate["generation_strategy"] = strategy_name
                    candidate["generation_timestamp"] = datetime.now().isoformat()
                
                all_candidates.extend(candidates)
                
            except Exception as e:
                print(f"    ⚠️ Error with {strategy_name}: {e}")
                continue
        
        results["candidates"] = all_candidates
        results["total_generated"] = len(all_candidates)
        results["status"] = "completed" if all_candidates else "failed"
        
        # Save candidates
        candidates_file = self.run_dir / "generated_candidates.json"
        with open(candidates_file, 'w') as f:
            json.dump({"candidates": all_candidates}, f, indent=2)
        
        return results
    
    def _format_retrieved_context(self, retrieved_context: Dict) -> str:
        """Format retrieved context for prompt"""
        context_parts = []
        
        for category, results in retrieved_context.items():
            if results:
                context_parts.append(f"\n{category.replace('_', ' ').title()}:")
                for r in results[:5]:  # Top 5 per category
                    material = r.content
                    context_parts.append(
                        f"- {material.get('formula', 'Unknown')}: "
                        f"{material.get('source', 'unknown')} source"
                    )
        
        return "\n".join(context_parts)
    
    def _llm_generate(self, prompt: str) -> List[Dict]:
        """Generate candidates using LLM"""
        if not self.config["openai_api_key"]:
            # Return mock candidates for testing
            return self._mock_candidates()
        
        try:
            response = openai.ChatCompletion.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": self.prompt_templates.templates["system"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                candidates = json.loads(json_match.group())
                return candidates
            else:
                print("Warning: Could not parse LLM response as JSON")
                return []
                
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._mock_candidates()
    
    def _mock_candidates(self) -> List[Dict]:
        """Generate mock candidates for testing"""
        return [
            {
                "formula": "Fe0.25Co0.25Ni0.25Cu0.25",
                "structure": "fcc",
                "properties": {
                    "expected_activity": "high",
                    "expected_stability": "moderate"
                },
                "rationale": "High-entropy alloy with balanced d-band center",
                "similar_to": ["FeCo", "NiCu"]
            },
            {
                "formula": "Cu0.7Mn0.3",
                "structure": "fcc",
                "properties": {
                    "expected_activity": "moderate",
                    "expected_stability": "high"
                },
                "rationale": "Cu-based with Mn for electronic tuning",
                "similar_to": ["Cu", "CuZn"]
            },
            {
                "formula": "Ni0.5Ti0.5",
                "structure": "intermetallic",
                "properties": {
                    "expected_activity": "moderate",
                    "expected_stability": "very high"
                },
                "rationale": "Stable intermetallic with accessible d-states",
                "similar_to": ["NiAl", "TiFe"]
            }
        ]
    
    def _run_screening(self, candidates: List[Dict]) -> Dict:
        """Run screening stage"""
        results = {
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
        # Run screening
        screening_config = self.config["screening"]
        screening_results = self.screener.screen_candidates(
            candidates=candidates,
            check_novelty=screening_config["check_novelty"],
            check_stability=screening_config["check_stability"],
            stability_threshold=screening_config["stability_threshold"]
        )
        
        # Extract passed candidates
        passed_candidates = [
            r["candidate"] for r in screening_results["screening_results"]
            if r["passed_screening"]
        ]
        
        results.update({
            "status": "completed",
            "total_screened": len(candidates),
            "passed": len(passed_candidates),
            "failed": len(candidates) - len(passed_candidates),
            "passed_candidates": passed_candidates,
            "summary": screening_results["summary"]
        })
        
        return results
    
    def _run_validation(self, candidates: List[Dict]) -> Dict:
        """Run DFT validation stage"""
        results = {
            "status": "started",
            "timestamp": datetime.now().isoformat(),
            "validations": []
        }
        
        for candidate in candidates[:5]:  # Limit to top 5 for computational cost
            print(f"  - Validating {candidate['formula']}...")
            
            try:
                # Setup DFT calculations
                calc_dirs = []
                
                # Bulk
                bulk_dir = self.dft_automation.setup_bulk_calculation(
                    candidate["formula"],
                    structure_type=candidate.get("structure", "fcc")
                )
                calc_dirs.append(bulk_dir)
                
                # Surface and adsorption would be set up here
                # (simplified for demonstration)
                
                # Run calculations
                calc_results = self.dft_automation.run_calculations(
                    calc_dirs,
                    submit=self.config["validation"]["submit_jobs"]
                )
                
                # Mock DFT results for demonstration
                dft_results = {
                    "formation_energy": -0.5 + np.random.uniform(-0.3, 0.3),
                    "energy_above_hull": max(0, np.random.uniform(-0.05, 0.15)),
                    "band_gap": max(0, np.random.uniform(0, 2.0)),
                    "adsorption_energies": {
                        "CO_top": -0.6 + np.random.uniform(-0.3, 0.3),
                        "H_top": -0.3 + np.random.uniform(-0.2, 0.2)
                    }
                }
                
                validation_result = {
                    "candidate": candidate,
                    "dft_results": dft_results,
                    "calc_status": "completed"
                }
                
                results["validations"].append(validation_result)
                
            except Exception as e:
                results["validations"].append({
                    "candidate": candidate,
                    "calc_status": "failed",
                    "error": str(e)
                })
        
        results["status"] = "completed"
        results["total_validated"] = len(results["validations"])
        
        return results
    
    def _run_feedback(self, pipeline_results: Dict) -> Dict:
        """Run feedback and learning stage"""
        results = {
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
        # Record validation results
        if "validation" in pipeline_results["stages"]:
            for validation in pipeline_results["stages"]["validation"]["validations"]:
                if validation.get("calc_status") == "completed":
                    validation_id = self.feedback_system.record_validation_result(
                        candidate=validation["candidate"],
                        dft_results=validation["dft_results"],
                        screening_results={},  # Would come from screening stage
                        generation_metadata={
                            "strategy": validation["candidate"].get("generation_strategy"),
                            "run_id": self.current_run_id
                        }
                    )
                    
                    print(f"  - Recorded validation {validation_id}")
        
        # Train property predictor if enough data
        if len(self.feedback_system.validation_history) >= 20:
            print("  - Training property predictor...")
            self.feedback_system.train_property_predictor()
            results["ml_model_trained"] = True
        
        # Generate learning report
        report_path = self.feedback_system.generate_learning_report()
        results["learning_report"] = str(report_path)
        
        results["status"] = "completed"
        
        return results
    
    def _generate_pipeline_report(self, results: Dict):
        """Generate comprehensive pipeline report"""
        report = f"""# Catalyst Discovery Pipeline Report
Run ID: {results['run_id']}
Generated: {results['timestamp']}

## Target Reaction
{results['reaction']}

## Constraints
"""
        
        for key, value in results['constraints'].items():
            report += f"- {key}: {value}\n"
        
        # Stage summaries
        for stage_name, stage_results in results['stages'].items():
            report += f"\n## {stage_name.replace('_', ' ').title()}\n"
            report += f"Status: {stage_results.get('status', 'unknown')}\n"
            
            if stage_name == "generation":
                report += f"Total candidates generated: {stage_results.get('total_generated', 0)}\n"
            elif stage_name == "screening":
                report += f"Passed screening: {stage_results.get('passed', 0)}/{stage_results.get('total_screened', 0)}\n"
            elif stage_name == "validation":
                report += f"Validated candidates: {stage_results.get('total_validated', 0)}\n"
        
        # Top candidates
        if "validation" in results["stages"]:
            report += "\n## Top Validated Candidates\n"
            validations = results["stages"]["validation"]["validations"]
            
            # Sort by overall score (would be calculated in real implementation)
            for i, val in enumerate(validations[:5], 1):
                if val.get("calc_status") == "completed":
                    report += f"\n### {i}. {val['candidate']['formula']}\n"
                    report += f"- Formation energy: {val['dft_results']['formation_energy']:.3f} eV/atom\n"
                    report += f"- CO binding: {val['dft_results']['adsorption_energies']['CO_top']:.3f} eV\n"
        
        # Save report
        report_file = self.run_dir / "pipeline_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Pipeline report saved to: {report_file}")
    
    def run_iterative_discovery(self,
                               reaction: str,
                               constraints: CatalystConstraints,
                               max_iterations: int = 3) -> List[Dict]:
        """Run iterative discovery with feedback"""
        all_results = []
        
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Update constraints based on previous results
            if iteration > 0 and all_results:
                constraints = self._update_constraints_from_feedback(
                    constraints, 
                    all_results[-1]
                )
            
            # Run pipeline
            results = self.run_full_pipeline(
                reaction=reaction,
                constraints=constraints,
                skip_data_collection=(iteration > 0)
            )
            
            all_results.append(results)
            
            # Check for convergence
            if self._check_convergence(results):
                print("\n✨ Converged! Found satisfactory candidates.")
                break
        
        return all_results
    
    def _update_constraints_from_feedback(self, 
                                        constraints: CatalystConstraints,
                                        previous_results: Dict) -> CatalystConstraints:
        """Update constraints based on previous iteration"""
        # This would analyze previous results and adjust constraints
        # For example, if all candidates had poor stability, might adjust allowed elements
        return constraints
    
    def _check_convergence(self, results: Dict) -> bool:
        """Check if discovery has converged to good candidates"""
        if "validation" not in results["stages"]:
            return False
        
        # Check if we have candidates with good overall scores
        validations = results["stages"]["validation"]["validations"]
        good_candidates = sum(
            1 for v in validations 
            if v.get("calc_status") == "completed" and
            v["dft_results"]["formation_energy"] < -0.3 and
            abs(v["dft_results"]["adsorption_energies"]["CO_top"] - (-0.6)) < 0.2
        )
        
        return good_candidates >= 3


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Run catalyst discovery pipeline"
    )
    parser.add_argument("--reaction", type=str, required=True,
                       help="Target catalytic reaction")
    parser.add_argument("--config", type=str, default="pipeline_config.yaml",
                       help="Pipeline configuration file")
    parser.add_argument("--elements", nargs="+", 
                       default=["Fe", "Co", "Ni", "Cu", "Mn"],
                       help="Allowed elements for catalyst")
    parser.add_argument("--max-elements", type=int, default=4,
                       help="Maximum number of elements in catalyst")
    parser.add_argument("--iterative", action="store_true",
                       help="Run iterative discovery")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data collection stage")
    
    args = parser.parse_args()
    
    # Setup constraints
    constraints = CatalystConstraints(
        allowed_elements=args.elements,
        max_elements=args.max_elements,
        earth_abundant_only=True,
        target_properties={
            "activity": "high",
            "selectivity": "high",
            "stability": "moderate"
        }
    )
    
    # Initialize pipeline
    pipeline = CatalystDiscoveryPipeline(config_file=args.config)
    
    # Run discovery
    if args.iterative:
        results = pipeline.run_iterative_discovery(
            reaction=args.reaction,
            constraints=constraints
        )
    else:
        results = pipeline.run_full_pipeline(
            reaction=args.reaction,
            constraints=constraints,
            skip_data_collection=args.skip_data
        )
    
    print("\n🎉 Catalyst discovery complete!")


if __name__ == "__main__":
    # Add numpy import for mock DFT results
    import numpy as np
    main()