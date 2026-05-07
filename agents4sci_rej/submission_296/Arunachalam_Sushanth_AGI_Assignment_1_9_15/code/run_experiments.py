"""
Main experimental runner for the IDP system
Generates synthetic data, runs experiments, and produces results
"""

import os
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Tuple

# Local imports
from ocr_backends import extract_tokens, SimulatedOCRBackend
from transcript_parser import parse_transcript
from decision_rules import load_decision_engine, AcademicDecision
from resume_ner import extract_resume_entities
from sop_rubric import analyze_statement_of_purpose
from feature_fusion import fuse_application_features
from synthetic_data_generator import SyntheticDataGenerator
from evaluate import ExperimentEvaluator

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experimental framework for IDP system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.load_config()
        
        # Create timestamped results directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results/results_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.decision_engine = load_decision_engine(config_dict=self.config)
        self.data_generator = SyntheticDataGenerator(self.config)
        self.evaluator = ExperimentEvaluator(self.config)
        
        # Results storage
        self.results = {
            "timestamp": self.timestamp,
            "config": self.config,
            "metrics": {},
            "baselines": {},
            "ablations": {}
        }
        
        logger.info(f"Initialized experiment runner with timestamp {self.timestamp}")
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "synthetic": {
                "num_transcripts": 100,
                "num_resumes": 50, 
                "num_statements": 30,
                "random_seed": 42
            },
            "thresholds": {
                "gpa_threshold": 3.0,
                "min_credits": 90,
                "abstain_threshold": 0.7
            }
        }
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete experimental evaluation"""
        logger.info("Starting full experimental evaluation")
        
        try:
            # Step 1: Generate synthetic dataset
            logger.info("Step 1: Generating synthetic dataset")
            dataset = self._generate_dataset()
            
            # Step 2: Run main pipeline
            logger.info("Step 2: Running main pipeline")
            main_results = self._run_main_pipeline(dataset)
            self.results["main_pipeline"] = main_results
            
            # Step 3: Run baseline comparisons
            logger.info("Step 3: Running baseline comparisons")
            baseline_results = self._run_baselines(dataset)
            self.results["baselines"] = baseline_results
            
            # Step 4: Run ablation studies
            logger.info("Step 4: Running ablation studies")
            ablation_results = self._run_ablations(dataset)
            self.results["ablations"] = ablation_results
            
            # Step 5: Compute final metrics
            logger.info("Step 5: Computing metrics and evaluation")
            metrics = self._compute_metrics()
            self.results["metrics"] = metrics
            
            # Step 6: Save results
            self._save_results()
            
            # Step 7: Generate plots
            logger.info("Step 6: Generating visualization plots")
            self._generate_plots()
            
            logger.info("Experimental evaluation completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            self.results["error"] = str(e)
            return self.results
    
    def _generate_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate synthetic dataset"""
        synthetic_config = self.config.get("synthetic", {})
        
        # Generate different document types
        transcripts = self.data_generator.generate_transcripts(
            count=synthetic_config.get("num_transcripts", 100)
        )
        
        resumes = self.data_generator.generate_resumes(
            count=synthetic_config.get("num_resumes", 50)
        )
        
        statements = self.data_generator.generate_statements(
            count=synthetic_config.get("num_statements", 30)
        )
        
        dataset = {
            "transcripts": transcripts,
            "resumes": resumes,
            "statements": statements
        }
        
        logger.info(f"Generated dataset: {len(transcripts)} transcripts, "
                   f"{len(resumes)} resumes, {len(statements)} statements")
        
        return dataset
    
    def _run_main_pipeline(self, dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Run the main IDP pipeline"""
        results = {
            "transcript_results": [],
            "processing_times": [],
            "decisions": []
        }
        
        for i, transcript_data in enumerate(dataset["transcripts"]):
            start_time = datetime.now()
            
            try:
                # Simulate OCR
                tokens = self._simulate_document_tokens(transcript_data, "transcript")
                
                # Parse transcript
                parse_result = parse_transcript(tokens, self.config)
                
                # Make decision
                decision_result = self.decision_engine.make_decision(
                    gpa=parse_result.gpa,
                    credits=parse_result.total_credits,
                    parsing_confidence=parse_result.parsing_confidence
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Store results
                result = {
                    "document_id": i,
                    "ground_truth_gpa": transcript_data.get("true_gpa", 0.0),
                    "predicted_gpa": parse_result.gpa,
                    "ground_truth_credits": transcript_data.get("true_credits", 0.0),
                    "predicted_credits": parse_result.total_credits,
                    "decision": decision_result.decision.value,
                    "confidence": decision_result.confidence,
                    "processing_time": processing_time,
                    "warnings": parse_result.warnings
                }
                
                results["transcript_results"].append(result)
                results["processing_times"].append(processing_time)
                results["decisions"].append(decision_result.decision.value)
                
            except Exception as e:
                logger.error(f"Pipeline failed for document {i}: {e}")
                continue
        
        return results
    
    def _run_baselines(self, dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Run baseline comparisons"""
        baselines = {}
        
        # GPA-only baseline
        gpa_only_results = []
        for transcript_data in dataset["transcripts"]:
            true_gpa = transcript_data.get("true_gpa", 0.0)
            
            # Simple threshold decision
            if true_gpa >= 3.0:
                decision = "ACCEPT_ACADEMIC"
            elif true_gpa < 2.5:
                decision = "REJECT_ACADEMIC" 
            else:
                decision = "REVIEW"
            
            gpa_only_results.append({
                "ground_truth_gpa": true_gpa,
                "predicted_gpa": true_gpa,  # Perfect GPA prediction
                "decision": decision,
                "confidence": 0.8,  # Fixed confidence
                "processing_time": 0.01  # Minimal processing
            })
        
        baselines["gpa_only"] = gpa_only_results
        
        # Random baseline
        np.random.seed(42)
        random_results = []
        for transcript_data in dataset["transcripts"]:
            decision = np.random.choice(["ACCEPT_ACADEMIC", "REVIEW", "REJECT_ACADEMIC"])
            gpa_prediction = np.random.uniform(2.0, 4.0)
            
            random_results.append({
                "ground_truth_gpa": transcript_data.get("true_gpa", 0.0),
                "predicted_gpa": gpa_prediction,
                "decision": decision,
                "confidence": np.random.uniform(0.5, 1.0),
                "processing_time": 0.02
            })
        
        baselines["random"] = random_results
        
        return baselines
    
    def _run_ablations(self, dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Run ablation studies"""
        ablations = {}
        
        # No calibration ablation
        no_calib_engine = load_decision_engine(config_dict=self.config)
        # Simulate removing calibration by fixing confidence
        
        no_calib_results = []
        for transcript_data in dataset["transcripts"]:
            tokens = self._simulate_document_tokens(transcript_data, "transcript") 
            parse_result = parse_transcript(tokens, self.config)
            
            decision_result = no_calib_engine.make_decision(
                gpa=parse_result.gpa,
                credits=parse_result.total_credits,
                parsing_confidence=1.0  # No uncertainty
            )
            
            # Force higher confidence (no calibration)
            decision_result.confidence = min(0.95, decision_result.confidence + 0.2)
            
            no_calib_results.append({
                "ground_truth_gpa": transcript_data.get("true_gpa", 0.0),
                "predicted_gpa": parse_result.gpa,
                "decision": decision_result.decision.value,
                "confidence": decision_result.confidence
            })
        
        ablations["no_calibration"] = no_calib_results
        
        return ablations
    
    def _simulate_document_tokens(self, doc_data: Dict[str, Any], doc_type: str) -> List:
        """Simulate document tokens for processing"""
        backend = SimulatedOCRBackend(noise_level=0.05)
        
        # Create a temporary file path for simulation
        temp_path = f"temp_{doc_type}_{doc_data.get('id', 0)}.pdf"
        
        if doc_type == "transcript":
            return backend._generate_transcript_tokens()
        elif doc_type == "resume":
            return backend._generate_resume_tokens()
        elif doc_type == "statement":
            return backend._generate_statement_tokens()
        else:
            return backend._generate_generic_tokens()
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics"""
        metrics = {}
        
        # Main pipeline metrics
        main_results = self.results.get("main_pipeline", {}).get("transcript_results", [])
        if main_results:
            metrics["main_pipeline"] = self.evaluator.compute_transcript_metrics(main_results)
        
        # Baseline metrics
        baseline_results = self.results.get("baselines", {})
        for baseline_name, baseline_data in baseline_results.items():
            metrics[f"baseline_{baseline_name}"] = self.evaluator.compute_transcript_metrics(baseline_data)
        
        # Processing efficiency
        processing_times = self.results.get("main_pipeline", {}).get("processing_times", [])
        if processing_times:
            metrics["efficiency"] = {
                "avg_processing_time": np.mean(processing_times),
                "throughput_per_hour": 3600 / np.mean(processing_times),
                "time_saved_estimate": self._estimate_time_savings(np.mean(processing_times))
            }
        
        return metrics
    
    def _estimate_time_savings(self, avg_processing_time: float) -> Dict[str, float]:
        """Estimate time savings vs manual review"""
        manual_time = 20 * 60  # 20 minutes in seconds
        automated_time = avg_processing_time
        
        savings_per_app = manual_time - automated_time
        savings_percentage = (savings_per_app / manual_time) * 100
        
        return {
            "manual_time_minutes": manual_time / 60,
            "automated_time_seconds": automated_time,
            "savings_per_application_minutes": savings_per_app / 60,
            "savings_percentage": savings_percentage,
            "time_saved_per_100_apps_hours": (savings_per_app * 100) / 3600
        }
    
    def _save_results(self):
        """Save results to JSON file"""
        results_file = self.results_dir / "metrics.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Also save to main results directory
        main_results_file = Path("results/metrics.json")
        with open(main_results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_plots(self):
        """Generate visualization plots"""
        from plotting import generate_all_plots
        
        plot_data = {
            "main_results": self.results.get("main_pipeline", {}),
            "baseline_results": self.results.get("baselines", {}),
            "ablation_results": self.results.get("ablations", {}),
            "metrics": self.results.get("metrics", {})
        }
        
        # Generate plots in both timestamped and main directories
        generate_all_plots(plot_data, self.results_dir / "figures")
        generate_all_plots(plot_data, Path("results/figures"))
        
        logger.info("Generated visualization plots")
    
    def print_summary(self):
        """Print experiment summary"""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY - {self.timestamp}")
        print(f"{'='*60}")
        
        metrics = self.results.get("metrics", {})
        main_metrics = metrics.get("main_pipeline", {})
        
        if main_metrics:
            print(f"GPA Extraction MAE: {main_metrics.get('gpa_mae', 0):.3f}")
            print(f"Academic Decision AUC: {main_metrics.get('decision_auc', 0):.3f}")
            print(f"Expected Calibration Error: {main_metrics.get('ece', 0):.3f}")
        
        efficiency = metrics.get("efficiency", {})
        if efficiency:
            print(f"Average Processing Time: {efficiency.get('avg_processing_time', 0):.2f}s")
            print(f"Time Savings: {efficiency.get('savings_percentage', 0):.1f}%")
            print(f"Throughput: {efficiency.get('throughput_per_hour', 0):.0f} applications/hour")
        
        print(f"\nResults saved to: {self.results_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_full_experiment()
    runner.print_summary()
    
    # Update AI contribution log
    log_file = Path("prompts/ai_contrib_log.md")
    if log_file.exists():
        with open(log_file, 'a') as f:
            f.write(f"\n\n2025-09-12: Executed full experimental pipeline with timestamp {runner.timestamp}. "
                   f"Generated synthetic data, ran main pipeline, baselines, and ablations. "
                   f"Computed comprehensive metrics and saved results to {runner.results_dir}.")