"""
Symbolic Calculator with Unit Conversions
Handles unit conversions, percentage calculations, ratios, and confidence intervals
"""

import pint
import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from .unit_parser import Quantity, UnitOntology


class OperationType(Enum):
    """Types of supported operations"""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    PERCENTAGE_CHANGE = "percentage_change"
    PERCENTAGE_POINT_DIFF = "percentage_point_diff"
    FOLD_CHANGE = "fold_change"
    RATIO = "ratio"
    CI_OVERLAP = "ci_overlap"
    CONVERT = "convert"
    COMPARE = "compare"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    MEDIAN = "median"


@dataclass
class CalculationResult:
    """Result of a symbolic calculation"""
    value: float
    unit: str
    operation: OperationType
    inputs: List[Quantity]
    confidence: float = 1.0
    error_message: Optional[str] = None
    
    def __repr__(self):
        if self.error_message:
            return f"CalculationError: {self.error_message}"
        return f"{self.value} {self.unit} (from {self.operation.value})"


@dataclass
class ConfidenceInterval:
    """Represents a confidence interval"""
    center: float
    lower: float
    upper: float
    confidence_level: float = 0.95
    unit: str = ""
    
    def overlaps(self, other: 'ConfidenceInterval') -> bool:
        """Check if two confidence intervals overlap"""
        return not (self.upper < other.lower or other.upper < self.lower)
    
    def contains(self, value: float) -> bool:
        """Check if a value is within the interval"""
        return self.lower <= value <= self.upper


class UnitConverter:
    """Handles unit conversions using pint"""
    
    def __init__(self):
        self.ureg = pint.UnitRegistry()
        self.ontology = UnitOntology()
        
        # Define custom conversions
        self._define_custom_units()
        
    def _define_custom_units(self):
        """Define custom units and conversions"""
        # Percentage and percentage points
        self.ureg.define('percent = 0.01 = %')
        self.ureg.define('percentage_point = 1 = pp')
        
        # Fold changes
        self.ureg.define('fold = 1 = x')
        
        # Common biological units
        self.ureg.define('cell = 1')
        self.ureg.define('molecule = 1')
        
    def convert(self, quantity: Quantity, target_unit: str) -> CalculationResult:
        """Convert a quantity to a target unit"""
        try:
            # Create pint quantity
            q = self.ureg.Quantity(quantity.value, quantity.unit)
            
            # Convert to target unit
            converted = q.to(target_unit)
            
            return CalculationResult(
                value=converted.magnitude,
                unit=str(converted.units),
                operation=OperationType.CONVERT,
                inputs=[quantity],
                confidence=quantity.confidence
            )
        except pint.DimensionalityError as e:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.CONVERT,
                inputs=[quantity],
                confidence=0,
                error_message=f"Cannot convert {quantity.unit} to {target_unit}: {str(e)}"
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.CONVERT,
                inputs=[quantity],
                confidence=0,
                error_message=f"Conversion error: {str(e)}"
            )
    
    def check_compatibility(self, unit1: str, unit2: str) -> bool:
        """Check if two units are dimensionally compatible"""
        try:
            q1 = self.ureg.Quantity(1, unit1)
            q2 = self.ureg.Quantity(1, unit2)
            q1.to(unit2)  # Try to convert
            return True
        except:
            return False


class SymbolicCalculator:
    """Performs symbolic calculations with unit awareness"""
    
    def __init__(self):
        self.converter = UnitConverter()
        self.ureg = self.converter.ureg
        
    def add(self, q1: Quantity, q2: Quantity) -> CalculationResult:
        """Add two quantities with unit conversion if needed"""
        if not self.converter.check_compatibility(q1.unit, q2.unit):
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.ADD,
                inputs=[q1, q2],
                confidence=0,
                error_message=f"Cannot add {q1.unit} and {q2.unit}: incompatible units"
            )
        
        try:
            # Convert q2 to q1's unit
            pq1 = self.ureg.Quantity(q1.value, q1.unit)
            pq2 = self.ureg.Quantity(q2.value, q2.unit)
            result = pq1 + pq2
            
            return CalculationResult(
                value=result.magnitude,
                unit=str(result.units),
                operation=OperationType.ADD,
                inputs=[q1, q2],
                confidence=min(q1.confidence, q2.confidence)
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.ADD,
                inputs=[q1, q2],
                confidence=0,
                error_message=str(e)
            )
    
    def subtract(self, q1: Quantity, q2: Quantity) -> CalculationResult:
        """Subtract two quantities with unit conversion if needed"""
        if not self.converter.check_compatibility(q1.unit, q2.unit):
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.SUBTRACT,
                inputs=[q1, q2],
                confidence=0,
                error_message=f"Cannot subtract {q2.unit} from {q1.unit}: incompatible units"
            )
        
        try:
            pq1 = self.ureg.Quantity(q1.value, q1.unit)
            pq2 = self.ureg.Quantity(q2.value, q2.unit)
            result = pq1 - pq2
            
            return CalculationResult(
                value=result.magnitude,
                unit=str(result.units),
                operation=OperationType.SUBTRACT,
                inputs=[q1, q2],
                confidence=min(q1.confidence, q2.confidence)
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.SUBTRACT,
                inputs=[q1, q2],
                confidence=0,
                error_message=str(e)
            )
    
    def multiply(self, q1: Quantity, q2: Quantity) -> CalculationResult:
        """Multiply two quantities"""
        try:
            pq1 = self.ureg.Quantity(q1.value, q1.unit)
            pq2 = self.ureg.Quantity(q2.value, q2.unit)
            result = pq1 * pq2
            
            return CalculationResult(
                value=result.magnitude,
                unit=str(result.units),
                operation=OperationType.MULTIPLY,
                inputs=[q1, q2],
                confidence=min(q1.confidence, q2.confidence)
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.MULTIPLY,
                inputs=[q1, q2],
                confidence=0,
                error_message=str(e)
            )
    
    def divide(self, q1: Quantity, q2: Quantity) -> CalculationResult:
        """Divide two quantities"""
        if q2.value == 0:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.DIVIDE,
                inputs=[q1, q2],
                confidence=0,
                error_message="Division by zero"
            )
        
        try:
            pq1 = self.ureg.Quantity(q1.value, q1.unit)
            pq2 = self.ureg.Quantity(q2.value, q2.unit)
            result = pq1 / pq2
            
            return CalculationResult(
                value=result.magnitude,
                unit=str(result.units),
                operation=OperationType.DIVIDE,
                inputs=[q1, q2],
                confidence=min(q1.confidence, q2.confidence)
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.DIVIDE,
                inputs=[q1, q2],
                confidence=0,
                error_message=str(e)
            )
    
    def percentage_change(self, original: Quantity, new: Quantity) -> CalculationResult:
        """Calculate percentage change between two values"""
        if not self.converter.check_compatibility(original.unit, new.unit):
            return CalculationResult(
                value=0,
                unit="percent",
                operation=OperationType.PERCENTAGE_CHANGE,
                inputs=[original, new],
                confidence=0,
                error_message=f"Cannot calculate percentage change between {original.unit} and {new.unit}"
            )
        
        if original.value == 0:
            return CalculationResult(
                value=0,
                unit="percent",
                operation=OperationType.PERCENTAGE_CHANGE,
                inputs=[original, new],
                confidence=0,
                error_message="Cannot calculate percentage change from zero"
            )
        
        try:
            # Convert new to original's unit
            converted_new = self.converter.convert(new, original.unit)
            if converted_new.error_message:
                return converted_new
            
            change = ((converted_new.value - original.value) / original.value) * 100
            
            return CalculationResult(
                value=change,
                unit="percent",
                operation=OperationType.PERCENTAGE_CHANGE,
                inputs=[original, new],
                confidence=min(original.confidence, new.confidence)
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="percent",
                operation=OperationType.PERCENTAGE_CHANGE,
                inputs=[original, new],
                confidence=0,
                error_message=str(e)
            )
    
    def percentage_point_diff(self, p1: Quantity, p2: Quantity) -> CalculationResult:
        """Calculate percentage point difference"""
        # Both should be percentages
        if 'percent' not in p1.unit.lower() or 'percent' not in p2.unit.lower():
            return CalculationResult(
                value=0,
                unit="percentage_point",
                operation=OperationType.PERCENTAGE_POINT_DIFF,
                inputs=[p1, p2],
                confidence=0,
                error_message="Both values must be percentages for percentage point difference"
            )
        
        diff = p1.value - p2.value
        
        return CalculationResult(
            value=diff,
            unit="percentage_point",
            operation=OperationType.PERCENTAGE_POINT_DIFF,
            inputs=[p1, p2],
            confidence=min(p1.confidence, p2.confidence)
        )
    
    def fold_change(self, original: Quantity, new: Quantity) -> CalculationResult:
        """Calculate fold change between two values"""
        if not self.converter.check_compatibility(original.unit, new.unit):
            return CalculationResult(
                value=0,
                unit="fold",
                operation=OperationType.FOLD_CHANGE,
                inputs=[original, new],
                confidence=0,
                error_message=f"Cannot calculate fold change between {original.unit} and {new.unit}"
            )
        
        if original.value == 0:
            return CalculationResult(
                value=0,
                unit="fold",
                operation=OperationType.FOLD_CHANGE,
                inputs=[original, new],
                confidence=0,
                error_message="Cannot calculate fold change from zero"
            )
        
        try:
            # Convert new to original's unit
            converted_new = self.converter.convert(new, original.unit)
            if converted_new.error_message:
                return converted_new
            
            fold = converted_new.value / original.value
            
            return CalculationResult(
                value=fold,
                unit="fold",
                operation=OperationType.FOLD_CHANGE,
                inputs=[original, new],
                confidence=min(original.confidence, new.confidence)
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="fold",
                operation=OperationType.FOLD_CHANGE,
                inputs=[original, new],
                confidence=0,
                error_message=str(e)
            )
    
    def ci_overlap(self, ci1: ConfidenceInterval, ci2: ConfidenceInterval) -> CalculationResult:
        """Check if two confidence intervals overlap"""
        overlaps = ci1.overlaps(ci2)
        
        # Calculate overlap fraction if they overlap
        if overlaps:
            overlap_start = max(ci1.lower, ci2.lower)
            overlap_end = min(ci1.upper, ci2.upper)
            overlap_size = overlap_end - overlap_start
            
            ci1_size = ci1.upper - ci1.lower
            ci2_size = ci2.upper - ci2.lower
            
            overlap_fraction = overlap_size / min(ci1_size, ci2_size)
        else:
            overlap_fraction = 0
        
        return CalculationResult(
            value=overlap_fraction,
            unit="ratio",
            operation=OperationType.CI_OVERLAP,
            inputs=[],  # CIs are not Quantity objects
            confidence=1.0
        )
    
    def compare(self, q1: Quantity, q2: Quantity, 
                comparison: str = "eq") -> CalculationResult:
        """Compare two quantities with unit conversion"""
        if not self.converter.check_compatibility(q1.unit, q2.unit):
            return CalculationResult(
                value=0,
                unit="boolean",
                operation=OperationType.COMPARE,
                inputs=[q1, q2],
                confidence=0,
                error_message=f"Cannot compare {q1.unit} and {q2.unit}: incompatible units"
            )
        
        try:
            # Convert q2 to q1's unit
            converted_q2 = self.converter.convert(q2, q1.unit)
            if converted_q2.error_message:
                return converted_q2
            
            comparisons = {
                "eq": q1.value == converted_q2.value,
                "ne": q1.value != converted_q2.value,
                "lt": q1.value < converted_q2.value,
                "le": q1.value <= converted_q2.value,
                "gt": q1.value > converted_q2.value,
                "ge": q1.value >= converted_q2.value
            }
            
            result = comparisons.get(comparison, False)
            
            return CalculationResult(
                value=float(result),
                unit="boolean",
                operation=OperationType.COMPARE,
                inputs=[q1, q2],
                confidence=min(q1.confidence, q2.confidence)
            )
        except Exception as e:
            return CalculationResult(
                value=0,
                unit="boolean",
                operation=OperationType.COMPARE,
                inputs=[q1, q2],
                confidence=0,
                error_message=str(e)
            )
    
    def aggregate(self, quantities: List[Quantity], 
                  operation: str = "mean") -> CalculationResult:
        """Aggregate multiple quantities (mean, median, min, max)"""
        if not quantities:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.MEAN,
                inputs=[],
                confidence=0,
                error_message="No quantities to aggregate"
            )
        
        # Check all quantities have compatible units
        base_unit = quantities[0].unit
        for q in quantities[1:]:
            if not self.converter.check_compatibility(q.unit, base_unit):
                return CalculationResult(
                    value=0,
                    unit="",
                    operation=OperationType.MEAN,
                    inputs=quantities,
                    confidence=0,
                    error_message=f"Incompatible units for aggregation: {q.unit} vs {base_unit}"
                )
        
        # Convert all to base unit
        values = []
        for q in quantities:
            if q.unit != base_unit:
                converted = self.converter.convert(q, base_unit)
                if converted.error_message:
                    return converted
                values.append(converted.value)
            else:
                values.append(q.value)
        
        # Perform aggregation
        if operation == "mean":
            result_value = np.mean(values)
            op_type = OperationType.MEAN
        elif operation == "median":
            result_value = np.median(values)
            op_type = OperationType.MEDIAN
        elif operation == "min":
            result_value = np.min(values)
            op_type = OperationType.MIN
        elif operation == "max":
            result_value = np.max(values)
            op_type = OperationType.MAX
        else:
            return CalculationResult(
                value=0,
                unit="",
                operation=OperationType.MEAN,
                inputs=quantities,
                confidence=0,
                error_message=f"Unknown aggregation operation: {operation}"
            )
        
        return CalculationResult(
            value=result_value,
            unit=base_unit,
            operation=op_type,
            inputs=quantities,
            confidence=np.mean([q.confidence for q in quantities])
        )