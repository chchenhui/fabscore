"""
Evaluation module for computing metrics and generating plots
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

class ExperimentEvaluator:
    """Evaluates experimental results and generates metrics"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def compute_transcript_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute metrics for transcript parsing results"""
        if not results:
            return {}
        
        # GPA metrics
        true_gpas = [r["ground_truth_gpa"] for r in results]
        pred_gpas = [r["predicted_gpa"] for r in results]
        
        gpa_mae = np.mean(np.abs(np.array(true_gpas) - np.array(pred_gpas)))
        gpa_rmse = np.sqrt(np.mean((np.array(true_gpas) - np.array(pred_gpas))**2))
        
        # Decision metrics
        decisions = [r["decision"] for r in results]
        confidences = [r["confidence"] for r in results]
        
        # Compute decision accuracy (approximate)
        decision_accuracy = self._compute_decision_accuracy(true_gpas, decisions)
        
        # Calibration metrics
        ece = self._compute_expected_calibration_error(decisions, confidences, true_gpas)
        
        # Processing time
        processing_times = [r.get("processing_time", 0) for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        metrics = {
            "gpa_mae": float(gpa_mae),
            "gpa_rmse": float(gpa_rmse),
            "decision_accuracy": float(decision_accuracy),
            "ece": float(ece),
            "avg_processing_time": float(avg_processing_time),
            "num_samples": len(results)
        }
        
        return metrics
    
    def _compute_decision_accuracy(self, true_gpas: List[float], decisions: List[str]) -> float:
        """Compute decision accuracy based on GPA thresholds"""
        correct = 0
        total = len(true_gpas)
        
        for true_gpa, decision in zip(true_gpas, decisions):
            # Determine expected decision based on GPA
            if true_gpa >= 3.0:
                expected = "ACCEPT_ACADEMIC"
            elif true_gpa < 2.5:
                expected = "REJECT_ACADEMIC"
            else:
                expected = "REVIEW"
            
            if decision == expected:
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_expected_calibration_error(self, decisions: List[str], 
                                          confidences: List[float], 
                                          true_gpas: List[float],
                                          n_bins: int = 10) -> float:
        """Compute Expected Calibration Error"""
        if not confidences or not decisions:
            return 0.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = [(conf >= bin_lower) and (conf < bin_upper) for conf in confidences]
            prop_in_bin = sum(in_bin) / len(in_bin)
            
            if prop_in_bin > 0:
                # Compute accuracy for this bin
                bin_decisions = [decisions[i] for i, in_b in enumerate(in_bin) if in_b]
                bin_gpas = [true_gpas[i] for i, in_b in enumerate(in_bin) if in_b]
                bin_confidences = [confidences[i] for i, in_b in enumerate(in_bin) if in_b]
                
                bin_accuracy = self._compute_decision_accuracy(bin_gpas, bin_decisions)
                avg_confidence = np.mean(bin_confidences)
                
                ece += np.abs(avg_confidence - bin_accuracy) * prop_in_bin
        
        return ece
    
    def generate_comparison_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate markdown comparison table"""
        
        methods = ["Random Baseline", "GPA-Only", "Proposed IDP"]
        
        table_lines = [
            "| Method | GPA MAE | Decision Acc | ECE | Time (s) | Samples |",
            "|--------|---------|--------------|-----|----------|---------|"
        ]
        
        for method in methods:
            key = method.lower().replace(" ", "_").replace("-", "_")
            metrics = results.get(key, {})
            
            line = f"| {method} | {metrics.get('gpa_mae', 0):.3f} | " \
                   f"{metrics.get('decision_accuracy', 0):.3f} | " \
                   f"{metrics.get('ece', 0):.3f} | " \
                   f"{metrics.get('avg_processing_time', 0):.2f} | " \
                   f"{metrics.get('num_samples', 0)} |"
            
            table_lines.append(line)
        
        return "\n".join(table_lines)