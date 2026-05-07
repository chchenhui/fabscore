"""
Evaluation Metrics for UnitMath
Includes standard classification metrics and unit-specific error analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json

from .unit_parser import Quantity, QuantityExtractor
from .symbolic_calculator import CalculationResult, ConfidenceInterval


class ErrorType(Enum):
    """Types of errors in unit-aware reasoning"""
    SCALE_ERROR = "scale_error"  # Wrong magnitude (e.g., mg vs g)
    UNIT_ERROR = "unit_error"    # Wrong unit type
    ARITHMETIC_ERROR = "arithmetic_error"  # Calculation mistake
    DIMENSION_ERROR = "dimension_error"    # Comparing incompatible units
    PERCENTAGE_POINT_ERROR = "percentage_point_error"  # % vs pp confusion
    CI_OVERLAP_ERROR = "ci_overlap_error"  # Confidence interval misinterpretation
    CORRECT = "correct"


@dataclass
class EvaluationResult:
    """Complete evaluation results"""
    # Standard metrics
    precision: float
    recall: float
    f1: float
    accuracy: float
    
    # Unit-specific metrics
    scale_error_rate: float
    unit_error_rate: float
    arithmetic_error_rate: float
    dimension_error_rate: float
    percentage_point_error_rate: float
    ci_overlap_accuracy: float
    
    # Detailed breakdown
    error_distribution: Dict[str, int]
    confusion_matrix: np.ndarray
    per_class_metrics: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "precision": f"{self.precision * 100:.1f}",
            "recall": f"{self.recall * 100:.1f}",
            "macro_f1": f"{self.f1 * 100:.1f}",
            "accuracy": f"{self.accuracy * 100:.1f}",
            "scale_error_rate": f"{self.scale_error_rate * 100:.1f}",
            "unit_error_rate": f"{self.unit_error_rate * 100:.1f}",
            "arithmetic_error_rate": f"{self.arithmetic_error_rate * 100:.1f}",
            "dimension_error_rate": f"{self.dimension_error_rate * 100:.1f}",
            "percentage_point_error_rate": f"{self.percentage_point_error_rate * 100:.1f}",
            "ci_overlap_accuracy": f"{self.ci_overlap_accuracy * 100:.1f}",
            "error_distribution": self.error_distribution,
            "confusion_matrix": self.confusion_matrix.tolist()
        }


def evaluate_claim_verification(predictions, ground_truth, labels=["Supported", "Refuted"]):
    """
    Standard evaluation function using sklearn metrics
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of ground truth labels
        labels: List of possible labels
    
    Returns:
        Dictionary with precision, recall, and macro F1
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth,
        predictions,
        labels=labels,
        average='macro',
        zero_division=0  # Avoid division by zero if a class is missing
    )

    return {
        "precision": f"{precision * 100:.1f}",
        "recall": f"{recall * 100:.1f}",
        "macro_f1": f"{f1 * 100:.1f}"
    }


class UnitMathEvaluator:
    """Comprehensive evaluator for unit-aware table reasoning"""
    
    def __init__(self):
        self.quantity_extractor = QuantityExtractor()
        self.error_analyzer = NumericErrorAnalyzer()
        
    def evaluate(self,
                predictions: List[str],
                ground_truth: List[str],
                calculation_results: Optional[List[CalculationResult]] = None,
                tables: Optional[List[Dict]] = None,
                claims: Optional[List[str]] = None,
                labels: List[str] = ["Supported", "Refuted"]) -> EvaluationResult:
        """
        Comprehensive evaluation with unit-specific error analysis
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            calculation_results: Results from symbolic calculator
            tables: Original table data
            claims: Original claims
            labels: Possible label values
        
        Returns:
            EvaluationResult with comprehensive metrics
        """
        
        # Standard metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth,
            predictions,
            labels=labels,
            average='macro',
            zero_division=0
        )
        
        # Accuracy
        accuracy = np.mean([p == g for p, g in zip(predictions, ground_truth)])
        
        # Confusion matrix
        conf_matrix = confusion_matrix(ground_truth, predictions, labels=labels)
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            ground_truth,
            predictions,
            labels=labels,
            average=None,
            zero_division=0
        )
        
        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[label] = {
                "precision": per_class_precision[i],
                "recall": per_class_recall[i],
                "f1": per_class_f1[i]
            }
        
        # Unit-specific error analysis
        error_distribution = {error_type.value: 0 for error_type in ErrorType}
        
        if calculation_results and tables and claims:
            error_types = self._analyze_errors(
                predictions,
                ground_truth,
                calculation_results,
                tables,
                claims
            )
            
            for error_type in error_types:
                error_distribution[error_type.value] += 1
        
        # Calculate error rates
        total_errors = sum(p != g for p, g in zip(predictions, ground_truth))
        if total_errors > 0:
            scale_error_rate = error_distribution[ErrorType.SCALE_ERROR.value] / total_errors
            unit_error_rate = error_distribution[ErrorType.UNIT_ERROR.value] / total_errors
            arithmetic_error_rate = error_distribution[ErrorType.ARITHMETIC_ERROR.value] / total_errors
            dimension_error_rate = error_distribution[ErrorType.DIMENSION_ERROR.value] / total_errors
            percentage_point_error_rate = error_distribution[ErrorType.PERCENTAGE_POINT_ERROR.value] / total_errors
        else:
            scale_error_rate = 0
            unit_error_rate = 0
            arithmetic_error_rate = 0
            dimension_error_rate = 0
            percentage_point_error_rate = 0
        
        # CI overlap accuracy
        ci_overlap_accuracy = self._calculate_ci_overlap_accuracy(
            calculation_results
        ) if calculation_results else 0
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            scale_error_rate=scale_error_rate,
            unit_error_rate=unit_error_rate,
            arithmetic_error_rate=arithmetic_error_rate,
            dimension_error_rate=dimension_error_rate,
            percentage_point_error_rate=percentage_point_error_rate,
            ci_overlap_accuracy=ci_overlap_accuracy,
            error_distribution=error_distribution,
            confusion_matrix=conf_matrix,
            per_class_metrics=per_class_metrics
        )
    
    def _analyze_errors(self,
                       predictions: List[str],
                       ground_truth: List[str],
                       calculation_results: List[CalculationResult],
                       tables: List[Dict],
                       claims: List[str]) -> List[ErrorType]:
        """Analyze the type of errors made"""
        error_types = []
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if pred != truth:
                # Misclassification - analyze the error
                error_type = self.error_analyzer.analyze_single_error(
                    calculation_results[i] if i < len(calculation_results) else None,
                    tables[i] if i < len(tables) else None,
                    claims[i] if i < len(claims) else None
                )
                error_types.append(error_type)
            else:
                error_types.append(ErrorType.CORRECT)
        
        return error_types
    
    def _calculate_ci_overlap_accuracy(self,
                                      calculation_results: List[CalculationResult]) -> float:
        """Calculate accuracy of confidence interval overlap detection"""
        ci_results = [
            r for r in calculation_results 
            if r and r.operation and 'ci_overlap' in r.operation.value
        ]
        
        if not ci_results:
            return 0
        
        # Check if CI overlap calculations were correct
        # This would require ground truth CI data
        # For now, return confidence average
        confidences = [r.confidence for r in ci_results if not r.error_message]
        
        return np.mean(confidences) if confidences else 0


class NumericErrorAnalyzer:
    """Analyzes numeric errors in calculations"""
    
    def __init__(self):
        self.quantity_extractor = QuantityExtractor()
    
    def analyze_single_error(self,
                           calc_result: Optional[CalculationResult],
                           table: Optional[Dict],
                           claim: Optional[str]) -> ErrorType:
        """Analyze a single error to determine its type"""
        
        if not calc_result:
            return ErrorType.ARITHMETIC_ERROR
        
        if calc_result.error_message:
            # Check error message for clues
            error_msg = calc_result.error_message.lower()
            
            if 'incompatible' in error_msg or 'dimension' in error_msg:
                return ErrorType.DIMENSION_ERROR
            elif 'unit' in error_msg:
                return ErrorType.UNIT_ERROR
            elif 'percentage' in error_msg:
                return ErrorType.PERCENTAGE_POINT_ERROR
            else:
                return ErrorType.ARITHMETIC_ERROR
        
        # Extract quantities from claim
        if claim:
            claim_quantities = self.quantity_extractor.extract_from_claim(claim)
            
            # Check for scale errors
            if self._has_scale_error(calc_result, claim_quantities):
                return ErrorType.SCALE_ERROR
            
            # Check for unit errors
            if self._has_unit_error(calc_result, claim_quantities):
                return ErrorType.UNIT_ERROR
            
            # Check for percentage point confusion
            if self._has_percentage_point_error(calc_result, claim_quantities):
                return ErrorType.PERCENTAGE_POINT_ERROR
        
        return ErrorType.ARITHMETIC_ERROR
    
    def _has_scale_error(self,
                        calc_result: CalculationResult,
                        claim_quantities: List[Quantity]) -> bool:
        """Check if there's a scale/magnitude error"""
        if not claim_quantities:
            return False
        
        # Check if the calculated value differs by orders of magnitude
        for q in claim_quantities:
            if calc_result.unit and self._same_dimension(calc_result.unit, q.unit):
                ratio = calc_result.value / q.value if q.value != 0 else 0
                if ratio > 100 or ratio < 0.01:
                    return True
        
        return False
    
    def _has_unit_error(self,
                       calc_result: CalculationResult,
                       claim_quantities: List[Quantity]) -> bool:
        """Check if there's a unit mismatch"""
        if not claim_quantities or not calc_result.unit:
            return False
        
        # Check if units are different but same dimension
        for q in claim_quantities:
            if self._same_dimension(calc_result.unit, q.unit) and calc_result.unit != q.unit:
                return True
        
        return False
    
    def _has_percentage_point_error(self,
                                   calc_result: CalculationResult,
                                   claim_quantities: List[Quantity]) -> bool:
        """Check for percentage vs percentage point confusion"""
        calc_unit = calc_result.unit.lower() if calc_result.unit else ""
        
        for q in claim_quantities:
            claim_unit = q.unit.lower()
            
            # Check if one is percentage and other is percentage point
            if ('percent' in calc_unit and 'point' in claim_unit) or \
               ('point' in calc_unit and 'percent' in claim_unit and 'point' not in claim_unit):
                return True
        
        return False
    
    def _same_dimension(self, unit1: str, unit2: str) -> bool:
        """Check if two units have the same dimension"""
        # Simplified check - in practice, use pint for proper dimensional analysis
        dimensions = {
            'mass': ['kg', 'g', 'mg', 'μg', 'lb'],
            'length': ['m', 'cm', 'mm', 'km', 'ft'],
            'percentage': ['%', 'percent', 'percentage_point', 'pp']
        }
        
        for dim, units in dimensions.items():
            if any(u in unit1.lower() for u in units) and \
               any(u in unit2.lower() for u in units):
                return True
        
        return False


class StressTestEvaluator:
    """Evaluates model robustness through stress tests"""
    
    def __init__(self, model):
        self.model = model
        self.quantity_extractor = QuantityExtractor()
    
    def test_unit_rescaling(self,
                           tables: List[Dict],
                           claims: List[str],
                           ground_truth: List[str]) -> Dict[str, float]:
        """Test invariance to unit rescaling (e.g., mg→g)"""
        results = {}
        
        # Original predictions
        original_preds = []
        for table, claim in zip(tables, claims):
            output = self.model(table, claim)
            pred = "Supported" if output.prediction[0, 0] > output.prediction[0, 1] else "Refuted"
            original_preds.append(pred)
        
        # Rescaled predictions
        rescaled_tables = self._rescale_units(tables)
        rescaled_preds = []
        for table, claim in zip(rescaled_tables, claims):
            output = self.model(table, claim)
            pred = "Supported" if output.prediction[0, 0] > output.prediction[0, 1] else "Refuted"
            rescaled_preds.append(pred)
        
        # Calculate invariance
        invariance = np.mean([o == r for o, r in zip(original_preds, rescaled_preds)])
        
        # Calculate accuracy on rescaled data
        rescaled_accuracy = np.mean([p == g for p, g in zip(rescaled_preds, ground_truth)])
        
        results['unit_rescaling_invariance'] = invariance
        results['rescaled_accuracy'] = rescaled_accuracy
        
        return results
    
    def test_percentage_swap(self,
                            tables: List[Dict],
                            claims: List[str],
                            ground_truth: List[str]) -> Dict[str, float]:
        """Test handling of percentage vs percentage points"""
        results = {}
        
        # Create claims with swapped percentage types
        swapped_claims = self._swap_percentage_types(claims)
        
        # Get predictions on swapped claims
        swapped_preds = []
        for table, claim in zip(tables, swapped_claims):
            output = self.model(table, claim)
            pred = "Supported" if output.prediction[0, 0] > output.prediction[0, 1] else "Refuted"
            swapped_preds.append(pred)
        
        # Calculate sensitivity to percentage type
        original_preds = []
        for table, claim in zip(tables, claims):
            output = self.model(table, claim)
            pred = "Supported" if output.prediction[0, 0] > output.prediction[0, 1] else "Refuted"
            original_preds.append(pred)
        
        sensitivity = 1 - np.mean([o == s for o, s in zip(original_preds, swapped_preds)])
        
        results['percentage_type_sensitivity'] = sensitivity
        
        return results
    
    def _rescale_units(self, tables: List[Dict]) -> List[Dict]:
        """Rescale units in tables (e.g., mg to g)"""
        rescaled = []
        
        for table in tables:
            new_table = table.copy()
            
            # Rescale data cells
            if 'data' in new_table:
                for i, row in enumerate(new_table['data']):
                    new_row = []
                    for cell in row:
                        cell_str = str(cell)
                        quantities = self.quantity_extractor.parser.parse_text(cell_str)
                        
                        if quantities:
                            q = quantities[0]
                            # Simple rescaling: mg to g
                            if 'mg' in q.unit:
                                new_value = q.value / 1000
                                new_unit = q.unit.replace('mg', 'g')
                                new_cell = f"{new_value} {new_unit}"
                            else:
                                new_cell = cell
                        else:
                            new_cell = cell
                        
                        new_row.append(new_cell)
                    new_table['data'][i] = new_row
            
            rescaled.append(new_table)
        
        return rescaled
    
    def _swap_percentage_types(self, claims: List[str]) -> List[str]:
        """Swap percentage and percentage points in claims"""
        swapped = []
        
        for claim in claims:
            new_claim = claim
            
            # Simple swapping
            if 'percentage points' in claim:
                new_claim = claim.replace('percentage points', 'percent')
            elif 'percent' in claim and 'points' not in claim:
                new_claim = claim.replace('percent', 'percentage points')
            
            swapped.append(new_claim)
        
        return swapped