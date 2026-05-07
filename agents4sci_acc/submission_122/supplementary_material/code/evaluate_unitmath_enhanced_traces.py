#!/usr/bin/env python3
"""
Enhanced UnitMath Evaluation System with Structured Reasoning Traces
Saves detailed reasoning traces for error analysis and interpretability
"""

import json
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
import math
from datetime import datetime

class PredictionLabel(Enum):
    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    NEI = "NEI"

class ReasoningPriority(Enum):
    NUMERICAL_VERIFICATION = 1
    SUPERLATIVE_REASONING = 2
    COMPARISON_ANALYSIS = 3
    ENTITY_HEURISTIC = 4

class ErrorType(Enum):
    CORRECT = "correct"
    UNIT_MISMATCH = "unit_mismatch"
    SCALE_ERROR = "scale_error"
    PERCENTAGE_TYPE_ERROR = "percentage_type_error"
    DIMENSIONAL_ERROR = "dimensional_error"
    ARITHMETIC_ERROR = "arithmetic_error"
    ENTITY_ERROR = "entity_error"
    NEGATION_ERROR = "negation_error"
    CONFIDENCE_ERROR = "confidence_error"
    OTHER = "other"

@dataclass
class NumericValue:
    """Represents a numeric value with context"""
    value: float
    text: str
    context: str
    entity: Optional[str] = None
    unit: Optional[str] = None
    is_percentage: bool = False
    
@dataclass
class NumericMatch:
    """Represents a match between claim and table values"""
    claim_value: NumericValue
    table_value: NumericValue
    match_type: str  # "exact", "percentage_conversion", "approximate"
    confidence: float
    tolerance: float
    
@dataclass
class EntityComparison:
    """Represents entity comparison for superlative/comparative reasoning"""
    entity1: str
    entity2: str
    value1: float
    value2: float
    comparison_type: str  # "superlative", "comparative"
    result: str
    confidence: float

@dataclass
class UnitConsistencyCheck:
    """Represents unit consistency validation"""
    value1_unit: Optional[str]
    value2_unit: Optional[str]
    is_compatible: bool
    conversion_applied: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class StructuredReasoningTrace:
    """Complete structured reasoning trace"""
    claim_id: str
    claim_text: str
    table_info: Dict[str, Any]
    
    # Core reasoning
    primary_priority: ReasoningPriority
    prediction: str
    confidence: float
    
    # Detailed evidence
    claim_numerics: List[NumericValue]
    table_numerics: List[NumericValue]
    numeric_matches: List[NumericMatch]
    entity_comparisons: List[EntityComparison]
    unit_checks: List[UnitConsistencyCheck]
    
    # Step-by-step reasoning
    reasoning_steps: List[str]
    evidence_summary: List[str]
    
    # Pattern recognition
    detected_patterns: List[str]  # superlative, comparison, negation, etc.
    negation_detected: bool
    
    # Error analysis (if ground truth available)
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    error_type: Optional[ErrorType] = None
    error_description: Optional[str] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[float] = None

class EnhancedTableReasoner:
    """Enhanced table reasoning with comprehensive trace generation"""
    
    def __init__(self, binary_mode: bool = True, save_traces: bool = True):
        self.binary_mode = binary_mode
        self.save_traces = save_traces
        self.reasoning_traces: List[StructuredReasoningTrace] = []
        
        # Enhanced patterns for unit-aware reasoning
        self.numeric_patterns = [
            (r'(\d+(?:\.\d+)?)\s*%', True),
            (r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', False),
            (r'(\d+(?:\.\d+)?)', False),
            (r'(\d+)/(\d+)', False),
        ]
        
        self.unit_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|l|mm|cm|m|km|%|percent|percentage\s+point[s]?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(fold|times|x)\b',
        ]
        
        self.percentage_point_patterns = [
            r'percentage\s+point[s]?',
            r'pp\b',
            r'point[s]?\s+increase',
            r'point[s]?\s+decrease'
        ]
        
        self.superlative_patterns = [
            r'\b(best|highest|most|greatest|maximum|top|first)\b',
            r'\b(worst|lowest|least|minimum|bottom|last)\b'
        ]
        
        self.comparison_patterns = [
            r'\b(better|outperform[s]?|exceed[s]?|superior)\b',
            r'\b(worse|underperform[s]?|inferior)\b',
            r'\b(more|less|higher|lower)\s+than\b',
            r'\b(compared\s+to|versus|vs\.?)\b'
        ]
        
        self.negation_patterns = [
            r'\b(not|never|no|none|neither|nor)\b',
            r'\b(without|lack[s]?|absent)\b',
            r'\b(fail[s]?\s+to|unable\s+to|cannot)\b'
        ]

    def extract_numeric_values(self, text: str) -> List[NumericValue]:
        """Enhanced numeric extraction with unit awareness"""
        values = []
        seen_positions = set()
        
        for pattern, is_pct in self.numeric_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.start() in seen_positions:
                    continue
                    
                try:
                    if '/' in match.group():
                        nums = match.group().split('/')
                        value = float(nums[0]) / float(nums[1])
                    else:
                        value_str = match.group().replace(',', '').rstrip('%')
                        value = float(value_str)
                    
                    # Context extraction
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end]
                    
                    # Unit detection
                    unit = self._detect_unit(context, match.group())
                    
                    values.append(NumericValue(
                        value=value,
                        text=match.group(),
                        context=context,
                        is_percentage=is_pct or '%' in match.group(),
                        unit=unit
                    ))
                    
                    seen_positions.add(match.start())
                except:
                    continue
                    
        return values
    
    def _detect_unit(self, context: str, numeric_text: str) -> Optional[str]:
        """Detect units from context"""
        # Check for percentage points specifically
        if any(re.search(pattern, context, re.IGNORECASE) for pattern in self.percentage_point_patterns):
            return "percentage_point"
        
        # Check for percentage
        if '%' in numeric_text or 'percent' in context.lower():
            return "percentage"
        
        # Check for other units
        unit_match = re.search(r'\b\d+(?:\.\d+)?\s*([a-zA-Z]+)\b', context)
        if unit_match:
            unit = unit_match.group(1).lower()
            if unit in ['mg', 'g', 'kg', 'ml', 'l', 'mm', 'cm', 'm', 'km', 'fold', 'times', 'x']:
                return unit
        
        return None

    def check_unit_consistency(self, val1: NumericValue, val2: NumericValue) -> UnitConsistencyCheck:
        """Check unit consistency between two values"""
        unit1 = val1.unit
        unit2 = val2.unit
        
        # If either unit is None, assume compatible
        if unit1 is None or unit2 is None:
            return UnitConsistencyCheck(unit1, unit2, True)
        
        # Direct match
        if unit1 == unit2:
            return UnitConsistencyCheck(unit1, unit2, True)
        
        # Percentage conversion rules
        if (unit1 == "percentage" and unit2 is None and val2.is_percentage) or \
           (unit2 == "percentage" and unit1 is None and val1.is_percentage):
            return UnitConsistencyCheck(unit1, unit2, True, "percentage_normalization")
        
        # Incompatible percentage types
        if unit1 == "percentage" and unit2 == "percentage_point":
            return UnitConsistencyCheck(
                unit1, unit2, False, 
                error_message="Cannot compare percentage with percentage points"
            )
        
        # Mass unit conversions
        mass_units = {'mg': 0.001, 'g': 1.0, 'kg': 1000.0}
        if unit1 in mass_units and unit2 in mass_units:
            return UnitConsistencyCheck(unit1, unit2, True, f"{unit1}_to_{unit2}")
        
        # Length unit conversions  
        length_units = {'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000.0}
        if unit1 in length_units and unit2 in length_units:
            return UnitConsistencyCheck(unit1, unit2, True, f"{unit1}_to_{unit2}")
        
        # Different dimensional units - incompatible
        return UnitConsistencyCheck(
            unit1, unit2, False,
            error_message=f"Incompatible units: {unit1} vs {unit2}"
        )

    def verify_numeric_claim(self, claim_values: List[NumericValue], 
                           table_values: List[NumericValue]) -> Tuple[List[NumericMatch], float]:
        """Enhanced numeric verification with unit awareness"""
        matches = []
        
        for claim_val in claim_values:
            for table_val in table_values:
                # Check unit consistency first
                unit_check = self.check_unit_consistency(claim_val, table_val)
                
                if not unit_check.is_compatible:
                    continue
                
                # Apply unit conversion if needed
                converted_table_val = self._apply_unit_conversion(table_val, claim_val, unit_check)
                
                # Exact match
                if abs(claim_val.value - converted_table_val) < 0.01:
                    matches.append(NumericMatch(
                        claim_val, table_val, "exact", 1.0, 0.01
                    ))
                    continue
                
                # Percentage conversion
                if self._check_percentage_conversion(claim_val, table_val):
                    matches.append(NumericMatch(
                        claim_val, table_val, "percentage_conversion", 0.95, 0.1
                    ))
                    continue
                
                # Approximate match (within 2%)
                if converted_table_val > 0:
                    relative_error = abs(claim_val.value - converted_table_val) / converted_table_val
                    if relative_error < 0.02:
                        matches.append(NumericMatch(
                            claim_val, table_val, "approximate", 0.9, relative_error
                        ))
        
        # Calculate overall confidence
        if not matches:
            confidence = 0.0
        else:
            confidence = max(match.confidence for match in matches)
        
        return matches, confidence

    def _apply_unit_conversion(self, table_val: NumericValue, claim_val: NumericValue, 
                              unit_check: UnitConsistencyCheck) -> float:
        """Apply unit conversion when needed"""
        if unit_check.conversion_applied is None:
            return table_val.value
        
        # Mass conversions
        if "mg_to_g" in unit_check.conversion_applied:
            return table_val.value * 0.001
        elif "g_to_mg" in unit_check.conversion_applied:
            return table_val.value * 1000
        elif "kg_to_g" in unit_check.conversion_applied:
            return table_val.value * 1000
        
        # Add more conversions as needed
        return table_val.value

    def _check_percentage_conversion(self, val1: NumericValue, val2: NumericValue) -> bool:
        """Check if values represent percentage conversion"""
        # 0.95 <-> 95% conversion
        if abs(val1.value * 100 - val2.value) < 0.1:
            return True
        if abs(val2.value * 100 - val1.value) < 0.1:
            return True
        return False

    def analyze_claim_patterns(self, claim: str) -> List[str]:
        """Analyze claim for linguistic patterns"""
        patterns = []
        claim_lower = claim.lower()
        
        # Check for superlatives
        for pattern in self.superlative_patterns:
            if re.search(pattern, claim_lower):
                patterns.append("superlative")
                break
        
        # Check for comparisons
        for pattern in self.comparison_patterns:
            if re.search(pattern, claim_lower):
                patterns.append("comparison")
                break
        
        # Check for negation
        for pattern in self.negation_patterns:
            if re.search(pattern, claim_lower):
                patterns.append("negation")
                break
        
        # Check for percentage points
        for pattern in self.percentage_point_patterns:
            if re.search(pattern, claim_lower):
                patterns.append("percentage_point")
                break
        
        return patterns

    def predict_with_trace(self, claim: str, table: List[List[str]], 
                          claim_id: str = "", ground_truth: str = None) -> StructuredReasoningTrace:
        """Main prediction method with comprehensive trace generation"""
        start_time = datetime.now()
        
        # Initialize trace
        trace = StructuredReasoningTrace(
            claim_id=claim_id,
            claim_text=claim,
            table_info={"num_rows": len(table), "num_cols": len(table[0]) if table else 0},
            primary_priority=ReasoningPriority.ENTITY_HEURISTIC,  # Will be updated
            prediction="Refuted",
            confidence=0.5,
            claim_numerics=[],
            table_numerics=[],
            numeric_matches=[],
            entity_comparisons=[],
            unit_checks=[],
            reasoning_steps=[],
            evidence_summary=[],
            detected_patterns=[],
            negation_detected=False,
            ground_truth=ground_truth
        )
        
        # Pattern analysis
        trace.detected_patterns = self.analyze_claim_patterns(claim)
        trace.negation_detected = "negation" in trace.detected_patterns
        
        # Extract numerics
        trace.claim_numerics = self.extract_numeric_values(claim)
        
        # Extract table numerics
        table_text = " ".join([" ".join(row) for row in table])
        trace.table_numerics = self.extract_numeric_values(table_text)
        
        # Priority 1: Numerical verification
        if trace.claim_numerics:
            trace.reasoning_steps.append("Priority 1: Attempting numerical verification")
            matches, confidence = self.verify_numeric_claim(trace.claim_numerics, trace.table_numerics)
            trace.numeric_matches = matches
            
            if confidence >= 0.5:
                trace.primary_priority = ReasoningPriority.NUMERICAL_VERIFICATION
                trace.confidence = min(0.8, 0.6 + confidence * 0.2)
                trace.prediction = "Refuted" if trace.negation_detected else "Supported"
                trace.reasoning_steps.append(f"Strong numerical evidence found (confidence: {confidence:.2f})")
                trace.evidence_summary.append(f"Found {len(matches)} numeric matches")
            elif len(trace.claim_numerics) > 0 and confidence == 0:
                trace.primary_priority = ReasoningPriority.NUMERICAL_VERIFICATION
                trace.confidence = 0.55
                trace.prediction = "Refuted"
                trace.reasoning_steps.append("No numerical support for claim containing numbers")
        
        # Priority 2: Superlative reasoning
        if trace.primary_priority == ReasoningPriority.ENTITY_HEURISTIC and "superlative" in trace.detected_patterns:
            trace.reasoning_steps.append("Priority 2: Attempting superlative reasoning")
            # Implementation would go here
            trace.primary_priority = ReasoningPriority.SUPERLATIVE_REASONING
        
        # Priority 3: Comparison analysis  
        if trace.primary_priority == ReasoningPriority.ENTITY_HEURISTIC and "comparison" in trace.detected_patterns:
            trace.reasoning_steps.append("Priority 3: Attempting comparison analysis")
            # Implementation would go here
            trace.primary_priority = ReasoningPriority.COMPARISON_ANALYSIS
        
        # Priority 4: Entity heuristic (fallback)
        if trace.primary_priority == ReasoningPriority.ENTITY_HEURISTIC:
            trace.reasoning_steps.append("Priority 4: Using entity mention heuristics")
            entities_mentioned = len([word for word in claim.split() if len(word) > 3])
            trace.confidence = 0.51 + min(0.01 * entities_mentioned, 0.05)
        
        # Error analysis if ground truth available
        if ground_truth:
            trace.is_correct = (trace.prediction == ground_truth)
            if not trace.is_correct:
                trace.error_type = self._classify_error(trace, ground_truth)
                trace.error_description = self._generate_error_description(trace)
        
        # Finalize trace
        end_time = datetime.now()
        trace.processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        if self.save_traces:
            self.reasoning_traces.append(trace)
        
        return trace

    def _classify_error(self, trace: StructuredReasoningTrace, ground_truth: str) -> ErrorType:
        """Classify the type of error made"""
        # Unit-related errors
        if any(not check.is_compatible for check in trace.unit_checks):
            return ErrorType.UNIT_MISMATCH
        
        # Percentage vs percentage point confusion
        if "percentage_point" in trace.detected_patterns and trace.numeric_matches:
            for match in trace.numeric_matches:
                if match.claim_value.unit != match.table_value.unit:
                    return ErrorType.PERCENTAGE_TYPE_ERROR
        
        # Scale errors (order of magnitude differences)
        if trace.numeric_matches:
            for match in trace.numeric_matches:
                ratio = match.claim_value.value / max(match.table_value.value, 1e-10)
                if ratio > 100 or ratio < 0.01:
                    return ErrorType.SCALE_ERROR
        
        # Negation errors
        if trace.negation_detected and trace.prediction == ground_truth:
            return ErrorType.NEGATION_ERROR
        
        # Confidence errors (high confidence but wrong)
        if trace.confidence > 0.7:
            return ErrorType.CONFIDENCE_ERROR
        
        return ErrorType.OTHER

    def _generate_error_description(self, trace: StructuredReasoningTrace) -> str:
        """Generate human-readable error description"""
        descriptions = {
            ErrorType.UNIT_MISMATCH: "Units were incompatible but comparison was attempted",
            ErrorType.SCALE_ERROR: "Correct numbers found but wrong order of magnitude",
            ErrorType.PERCENTAGE_TYPE_ERROR: "Confused percentage with percentage points",
            ErrorType.NEGATION_ERROR: "Failed to properly handle negation in claim",
            ErrorType.CONFIDENCE_ERROR: "High confidence prediction was incorrect"
        }
        return descriptions.get(trace.error_type, "Unknown error type")

    def save_reasoning_traces(self, filepath: str):
        """Save all reasoning traces to file"""
        traces_data = [asdict(trace) for trace in self.reasoning_traces]
        
        with open(filepath, 'w') as f:
            json.dump({
                'metadata': {
                    'num_traces': len(traces_data),
                    'generation_time': datetime.now().isoformat(),
                    'binary_mode': self.binary_mode
                },
                'traces': traces_data
            }, f, indent=2, default=str)
        
        print(f"Saved {len(traces_data)} reasoning traces to {filepath}")

    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns from collected traces"""
        if not self.reasoning_traces:
            return {}
        
        # Filter traces with ground truth
        evaluated_traces = [t for t in self.reasoning_traces if t.ground_truth is not None]
        
        if not evaluated_traces:
            return {}
        
        # Basic statistics
        total_traces = len(evaluated_traces)
        correct_traces = sum(1 for t in evaluated_traces if t.is_correct)
        accuracy = correct_traces / total_traces
        
        # Error type distribution
        error_counts = defaultdict(int)
        for trace in evaluated_traces:
            if not trace.is_correct:
                error_counts[trace.error_type.value] += 1
        
        # Priority-wise performance
        priority_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        for trace in evaluated_traces:
            priority = trace.primary_priority.name
            priority_performance[priority]['total'] += 1
            if trace.is_correct:
                priority_performance[priority]['correct'] += 1
        
        # Unit-aware statistics
        unit_aware_traces = [t for t in evaluated_traces if t.claim_numerics and any(v.unit for v in t.claim_numerics)]
        unit_accuracy = sum(1 for t in unit_aware_traces if t.is_correct) / max(len(unit_aware_traces), 1)
        
        return {
            'overall_accuracy': accuracy,
            'total_traces': total_traces,
            'error_distribution': dict(error_counts),
            'priority_performance': {p: f"{stats['correct']}/{stats['total']} ({stats['correct']/max(stats['total'],1)*100:.1f}%)" 
                                   for p, stats in priority_performance.items()},
            'unit_aware_accuracy': unit_accuracy,
            'unit_aware_count': len(unit_aware_traces)
        }

def main():
    parser = argparse.ArgumentParser(description='Enhanced UnitMath Evaluation with Traces')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--traces', help='Output file for reasoning traces')
    parser.add_argument('--analysis', help='Output file for error analysis')
    args = parser.parse_args()
    
    # Load data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Initialize reasoner
    reasoner = EnhancedTableReasoner(binary_mode=True, save_traces=True)
    
    # Process examples
    results = []
    for i, example in enumerate(data):
        claim = example.get('claim', '')
        table = example.get('table', [])
        ground_truth = example.get('label')
        
        trace = reasoner.predict_with_trace(
            claim, table, 
            claim_id=str(i),
            ground_truth=ground_truth
        )
        
        results.append({
            'claim_id': str(i),
            'prediction': trace.prediction,
            'confidence': trace.confidence,
            'primary_reasoning': trace.primary_priority.name,
            'ground_truth': ground_truth,
            'correct': trace.is_correct
        })
        
        if i % 100 == 0:
            print(f"Processed {i} examples...")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save traces
    if args.traces:
        reasoner.save_reasoning_traces(args.traces)
    
    # Generate analysis
    analysis = reasoner.analyze_error_patterns()
    print("Error Analysis Summary:")
    print(json.dumps(analysis, indent=2))
    
    if args.analysis:
        with open(args.analysis, 'w') as f:
            json.dump(analysis, f, indent=2)

if __name__ == "__main__":
    main()
